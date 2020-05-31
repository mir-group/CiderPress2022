from mldftdat import lowmem_analyzers
from mldftdat.pyscf_utils import get_vele_mat_generator
from pyscf.dft.gen_grid import Grids
from pyscf.dft.numint import eval_ao
from pyscf.dft.libxc import eval_xc
from pyscf import gto, df
import scipy.linalg
import numpy as np
from scipy.linalg.lapack import dgetrf, dgetri
from scipy.linalg.blas import dgemm, dgemv
from mldftdat.workflow_utils import safe_mem_cap_mb

def get_aux_mat_chunks(mol, points, num_chunks, lam = 0.5):
    """
    Generate chunks of vele_mat on the fly to reduce memory load.
    """
    if lam < 0.5 or lam >= 1.00:
        raise ValueError('lambda must be in [0.5, 1.0). For lambda=1.0, calculate conventional exchange energy density.')

    phases = []
    for j in range(mol.nbas):
        l = mol._bas[j][1]
        phases += [(-1)**l] * (2*l+1)

    phase = np.array(phases)
    num_pts = points.shape[0]
    for i in range(num_chunks):
        start = (i * num_pts) // num_chunks
        end = ((i+1) * num_pts) // num_chunks
        auxmol = gto.fakemol_for_charges(points[start:end])
        mol._env[11] = lam
        auxmol._env[11] = lam
        vele_mat_chunk = df.incore.aux_e2(mol, auxmol, intor='int3c2e_xed_sph')
        vele_mat_chunk = np.ascontiguousarray(np.transpose(
                                vele_mat_chunk, axes=(2,0,1)))
        #for i in range(mol.nao_nr()):
        #    vele_mat_chunk[:,:,i] *= phases[i]
        #vele_mat_chunk[:,:,:] *= phases
        vele_mat_chunk *= phases
        #vele_mat_chunk *= mulphases
        yield vele_mat_chunk

def get_aux_mat_generator(mol, coords, num_chunks, lam=0.5):
    return lambda: get_aux_mat_chunks(mol, coords, num_chunks, lam)

def get_fx_energy_density_from_aug(aux_mat_gen, mo_to_aux,
                                    mo_occ):
    ex_dens = np.einsum('pij,qij,i,j->pq', mo_to_aux, mo_to_aux,
                                          mo_occ, mo_occ)
    ex_dens.shape = (ex_dens.shape[0] * ex_dens.shape[1])
    fx_energy_density = np.array([])
    i = 0
    print('start loc_fx_iteration')
    for aux_mat_chunk in aux_mat_gen():
        #print('iter', i)
        #i += 1
        #fx_energy_density = np.append(fx_energy_density,
        #                    np.einsum('pq,xpq->x', ex_dens, aux_mat_chunk))
        aux_mat_chunk.shape = (aux_mat_chunk.shape[0],\
            aux_mat_chunk.shape[1] * aux_mat_chunk.shape[2])
        fxed_chunk = dgemv(1, aux_mat_chunk, ex_dens)
        fx_energy_density = np.append(fx_energy_density, fxed_chunk)
    return - 0.25 * fx_energy_density

class RHFAnalyzer(lowmem_analyzers.RHFAnalyzer):

    def post_process(self):
        super(RHFAnalyzer, self).post_process()
        self.auxmol = None
        self.loc_fx_energy_density = None

    def as_dict(self):
        analyzer_dict = super(RHFAnalyzer, self).as_dict()
        analyzer_dict['data']['loc_fx_energy_density'] = self.loc_fx_energy_density
        return analyzer_dict

    def assign_num_chunks(self, ao_vals_shape, ao_vals_dtype):
        self.max_mem = safe_mem_cap_mb()

        if ao_vals_dtype == np.float32:
            nbytes = 4
        elif ao_vals_dtype == np.float64:
            nbytes = 8
        else:
            raise ValueError('Wrong dtype for ao_vals')
        num_mbytes = nbytes * ao_vals_shape[0] * ao_vals_shape[1]**2 // 1000000
        self.num_chunks = int(num_mbytes // self.max_mem) + 1
        return self.num_chunks

    def setup_etb(self, lam = 0.5):
        auxbasis = df.aug_etb(self.mol, beta=1.6)
        nao = self.mol.nao_nr()
        self.auxmol = df.make_auxmol(self.mol, auxbasis)
        naux = self.auxmol.nao_nr()
        # shape (naux, naux), symmetric
        aug_J = self.auxmol.intor('int2c2e')
        # shape (nao, nao, naux)
        aux_e2 = df.incore.aux_e2(self.mol, self.auxmol)
        print(aux_e2.shape)
        # shape (naux, nao * nao)
        aux_e2 = aux_e2.reshape((-1, aux_e2.shape[-1])).transpose()
        aux_e2 = np.ascontiguousarray(aux_e2)
        self.augJ = aug_J.copy()
        lu, piv, info = dgetrf(aug_J, overwrite_a = True)
        inv_aug_J, info = dgetri(lu, piv, overwrite_lu = True)
        self.ao_to_aux = dgemm(1, inv_aug_J, aux_e2)
        print(self.ao_to_aux.shape)
        # phi_i phi_j = \sum_{mu,nu,P} c_{i,mu} c_{j,nu} d_{P,mu,nu}
        self.mo_to_aux = np.matmul(self.mo_coeff.transpose(),
                                   np.matmul(self.ao_to_aux.reshape(naux, nao, nao),
                                             self.mo_coeff))
        aux_ao_ratio = naux // nao + 1
        self.aux_num_chunks = self.num_chunks * aux_ao_ratio * aux_ao_ratio
        print(self.aux_num_chunks)
        small_grid = Grids(self.mol)
        small_grid.level = 3
        small_grid.build()
        self.small_grid = small_grid
        self.mo_aux_mat_generator = get_aux_mat_generator(
                                    self.auxmol, small_grid.coords,
                                    self.aux_num_chunks, lam = lam)

    def get_loc_fx_energy_density(self, lam = 0.5, overwrite = False):
        #tot_loc = np.einsum('pij,qij,i,j->pq', self.mo_to_aux, self.mo_to_aux,
        #                                    self.mo_occ, self.mo_occ)
        #tot_loc = -0.25 * np.einsum('pq,pq', tot_loc, self.augJ)
        if overwrite or (self.loc_fx_energy_density is None):
            self.setup_etb(lam = lam)
            self.loc_fx_energy_density = get_fx_energy_density_from_aug(
                                            self.mo_aux_mat_generator,
                                            self.mo_to_aux, self.mo_occ
                                            )
        return self.loc_fx_energy_density

    def perform_full_analysis(self):
        super(RHFAnalyzer, self).perform_full_analysis()
        self.get_loc_fx_energy_density()


class UHFAnalyzer(lowmem_analyzers.UHFAnalyzer):

    def post_process(self):
        super(UHFAnalyzer, self).post_process()
        self.auxmol = None
        self.loc_fx_energy_density = None
        self.loc_fx_energy_density_u = None
        self.loc_fx_energy_density_d = None

    def as_dict(self):
        analyzer_dict = super(UHFAnalyzer, self).as_dict()
        analyzer_dict['data']['loc_fx_energy_density'] = self.loc_fx_energy_density
        analyzer_dict['data']['loc_fx_energy_density_u'] = self.loc_fx_energy_density_u
        analyzer_dict['data']['loc_fx_energy_density_d'] = self.loc_fx_energy_density_d
        return analyzer_dict

    def assign_num_chunks(self, ao_vals_shape, ao_vals_dtype):
        self.max_mem = safe_mem_cap_mb()

        if ao_vals_dtype == np.float32:
            nbytes = 4
        elif ao_vals_dtype == np.float64:
            nbytes = 8
        else:
            raise ValueError('Wrong dtype for ao_vals')
        num_mbytes = nbytes * ao_vals_shape[0] * ao_vals_shape[1]**2 // 1000000
        self.num_chunks = int(num_mbytes // self.max_mem) + 1
        return self.num_chunks 

    def setup_etb(self, lam = 0.5):
        auxbasis = df.aug_etb(self.mol, beta=1.6)
        nao = self.mol.nao_nr()
        self.auxmol = df.make_auxmol(self.mol, auxbasis)
        naux = self.auxmol.nao_nr()
        # shape (naux, naux), symmetric
        aug_J = self.auxmol.intor('int2c2e')
        # shape (nao, nao, naux)
        aux_e2 = df.incore.aux_e2(self.mol, self.auxmol)
        print(aux_e2.shape)
        # shape (naux, nao * nao)
        aux_e2 = aux_e2.reshape((-1, aux_e2.shape[-1])).transpose()
        aux_e2 = np.ascontiguousarray(aux_e2)
        self.augJ = aug_J.copy()
        #self.inv_aug_J = scipy.linalg.inv(aug_J, overwrite_a = True)
        # d(Q,rs) shape (naux, nao * nao)
        # \sum_P \Theta_P * d(P,rs) \approx rho_{rs}
        print(aug_J.shape, aux_e2.shape)
        #self.ao_to_aux = scipy.linalg.solve(aug_J, aux_e2, overwrite_a=True,
        #                 overwrite_b=True, debug=None, check_finite=True,
        #                 assume_a='sym', transposed=False)
        lu, piv, info = dgetrf(aug_J, overwrite_a = True)
        inv_aug_J, info = dgetri(lu, piv, overwrite_lu = True)
        #self.ao_to_aux = np.matmul(inv_aug_J, aux_e2)
        self.ao_to_aux = dgemm(1, inv_aug_J, aux_e2)
        print(self.ao_to_aux.shape)
        # phi_i phi_j = \sum_{mu,nu,P} c_{i,mu} c_{j,nu} d_{P,mu,nu}
        self.mo_to_aux = [np.matmul(self.mo_coeff[0].transpose(),
                                   np.matmul(self.ao_to_aux.reshape(naux, nao, nao),
                                             self.mo_coeff[0])),\
                          np.matmul(self.mo_coeff[1].transpose(),
                                   np.matmul(self.ao_to_aux.reshape(naux, nao, nao),
                                             self.mo_coeff[1]))]

        aux_ao_ratio = naux // nao + 1
        self.aux_num_chunks = self.num_chunks * aux_ao_ratio * aux_ao_ratio
        print(self.aux_num_chunks)
        small_grid = Grids(self.mol)
        small_grid.level = 3
        small_grid.build()
        self.small_grid = small_grid
        self.mo_aux_mat_generator = get_aux_mat_generator(
                                    self.auxmol, small_grid.coords,
                                    self.aux_num_chunks, lam = lam)

    def get_loc_fx_energy_density(self, lam = 0.5, overwrite = False):
        #tot_loc = np.einsum('pij,qij,i,j->pq', self.mo_to_aux, self.mo_to_aux,
        #                                    self.mo_occ, self.mo_occ)
        #tot_loc = -0.25 * np.einsum('pq,pq', tot_loc, self.augJ)
        if overwrite or (self.loc_fx_energy_density is None):
            self.lam = lam
            self.setup_etb(lam = lam)
            self.loc_fx_energy_density_u = 2 * get_fx_energy_density_from_aug(
                                            self.mo_aux_mat_generator,
                                            self.mo_to_aux[0], self.mo_occ[0]
                                            )
            self.loc_fx_energy_density_d = 2 * get_fx_energy_density_from_aug(
                                            self.mo_aux_mat_generator,
                                            self.mo_to_aux[1], self.mo_occ[1]
                                            )
            self.loc_fx_energy_density = self.loc_fx_energy_density_u\
                                         + self.loc_fx_energy_density_d
        return self.loc_fx_energy_density

    def perform_full_analysis(self):
        super(RHFAnalyzer, self).perform_full_analysis()
        self.get_loc_fx_energy_density()
