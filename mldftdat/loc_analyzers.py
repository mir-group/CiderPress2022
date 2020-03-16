from mldftdat import lowmem_analyzers
from mldftdat.pyscf_utils import get_vele_mat_generator
from pyscf.dft.gen_grid import Grids
from pyscf import gto, df
import scipy.linalg
import numpy as np

def invert_mol(mol, center):
    atom = mol._atom
    new_atom = []
    for i in range(mol.natm):
        new_atom.append((atom[i][0],\
            2*center - np.array(atom[i][1])))
    new_mol = mol.copy()
    new_mol.atom = new_atom
    new_mol.unit = 'au'
    new_mol.build()
    return new_mol

def get_centered_mols(mol, center):
    atom = mol._atom
    new_atom1 = []
    new_atom2 = []
    for i in range(mol.natm):
        new_atom1.append((atom[i][0],\
            center - np.array(atom[i][1])))
        new_atom2.append((atom[i][0],\
            center + np.array(atom[i][1])))
    mol1 = mol.copy()
    mol1.atom = new_atom1
    mol1.unit = 'au'
    mol1.set_common_origin([0,0,0])
    mol1.build()
    mol2 = mol.copy()
    mol2.atom = new_atom2
    mol2.unit = 'au'
    mol2.set_common_origin([0,0,0])
    mol2.build()
    return mol1, mol2

def get_eri(mol, center):
    mol1, mol2, = get_centered_mols(mol, center)
    atm, bas, env = \
        gto.mole.conc_env(mol1._atm, mol1._bas, mol1._env,
                          mol2._atm, mol2._bas, mol2._env)
    # https://github.com/pyscf/pyscf/blob/master/examples/gto/21-concatenate_molecules.py
    eri2c = np.zeros((mol1.nao_nr(), mol2.nao_nr()))
    pi = 0
    for i in range(mol1.nbas):
        pj = 0
        for j in range(mol1.nbas, mol1.nbas+mol2.nbas):
            shls = (i, j)
            buf = gto.moleintor.getints_by_shell('cint1e_rinv_sph',
                                                 shls, atm, bas, env)
            di, dj = buf.shape
            #print(buf.shape)
            eri2c[pi:pi+di,pj:pj+dj] = buf
            pj += dj
        pi += di
    return eri2c

def get_aux_mat_chunks(mol, points, num_chunks):
    naux = mol.nao_nr()
    num_pts = points.shape[0]
    for i in range(num_chunks):
        start = (i * num_pts) // num_chunks
        end = min(((i+1) * num_pts) // num_chunks, num_pts)
        chunk = np.zeros((end - start, naux, naux))
        for index in range(start, end):
            center = points[index]
            #mol2 = invert_mol(mol, center)
            #mol3 = gto.fakemol_for_charges(np.array([center]))
            #chunk[index-start,:,:] = df.incore.aux_e2(mol + mol2, mol3)[:naux,naux:,0]
            chunk[index-start,:,:] = get_eri(mol, points[index])
        yield chunk

def get_aux_mat_generator(mol, coords, num_chunks):
    return lambda: get_aux_mat_chunks(mol, coords, num_chunks)

def get_fx_energy_density_from_aug(aux_mat_gen, mo_to_aux,
                                    mo_occ):
    ex_dens = np.einsum('pij,qij,i,j->pq', mo_to_aux, mo_to_aux,
                                          mo_occ, mo_occ)
    fx_energy_density = np.array([])
    i = 0
    for aux_mat_chunk in aux_mat_gen():
        print('iter', i)
        i += 1
        fx_energy_density = np.append(fx_energy_density,
                            np.einsum('pq,xpq->x', ex_dens, aux_mat_chunk))
    return -0.25 * fx_energy_density

class RHFAnalyzer(lowmem_analyzers.RHFAnalyzer):

    def setup_etb(self):
        #auxbasis = df.aug_etb(mol, beta=1.6)
        auxbasis = df.aug_etb(self.mol, beta=2.3)
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
        self.augJ = aug_J.copy()
        #self.inv_aug_J = scipy.linalg.inv(aug_J, overwrite_a = True)
        # d(Q,rs) shape (naux, nao * nao)
        # \sum_P \Theta_P * d(P,rs) \approx rho_{rs}
        print(aug_J.shape, aux_e2.shape)
        self.ao_to_aux = scipy.linalg.solve(aug_J, aux_e2, overwrite_a=False,
                         overwrite_b=False, debug=None, check_finite=True,
                         assume_a='sym', transposed=False)
        print(self.ao_to_aux.shape)
        # phi_i phi_j = \sum_{mu,nu,P} c_{i,mu} c_{j,nu} d_{P,mu,nu}
        self.mo_to_aux = np.matmul(self.mo_coeff.transpose(),
                                   np.matmul(self.ao_to_aux.reshape(naux, nao, nao),
                                             self.mo_coeff))
        aux_ao_ratio = naux // nao + 1
        self.aux_num_chunks = self.num_chunks * aux_ao_ratio * aux_ao_ratio
        print(self.aux_num_chunks)
        #self.mo_aux_mat_generator = get_vele_mat_generator(
        #                            self.auxmol, self.grid.coords,
        #                            self.aux_num_chunks)
        small_grid = Grids(self.mol)
        small_grid.level = 0
        small_grid.build()
        self.small_grid = small_grid
        self.mo_aux_mat_generator = get_aux_mat_generator(
                                    self.auxmol, small_grid.coords,
                                    self.aux_num_chunks)

    def get_loc_fx_energy_density(self):
        #ha_pred = np.einsum('')
        tot_loc = np.einsum('pij,qij,i,j->pq', self.mo_to_aux, self.mo_to_aux,
                                            self.mo_occ, self.mo_occ)
        tot_loc = -0.25 * np.einsum('pq,pq', tot_loc, self.augJ)
        self.loc_fx_energy_density = get_fx_energy_density_from_aug(
                                        self.mo_aux_mat_generator,
                                        self.mo_to_aux, self.mo_occ
                                        )
        return self.loc_fx_energy_density, tot_loc
