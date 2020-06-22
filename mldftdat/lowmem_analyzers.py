from pyscf import scf, dft, gto, ao2mo, df, lib, fci, cc
from pyscf.dft.numint import eval_ao, eval_rho
from pyscf.dft.gen_grid import Grids
from pyscf.pbc.tools.pyscf_ase import atoms_from_ase
from mldftdat.pyscf_utils import *
from mldftdat.external import pyscf_ccsd_rdm as ext_ccsd_rdm
#from mldftdat.external import pyscf_uccsd_rdm as ext_uccsd_rdm
import numpy as np
from abc import ABC, abstractmethod, abstractproperty
from io import BytesIO
import psutil


CALC_TYPES = {
    'RHF'   : scf.hf.RHF,
    'UHF'   : scf.uhf.UHF,
    'RKS'   : dft.rks.RKS,
    'UKS'   : dft.uks.UKS,
    'CCSD'  : cc.ccsd.CCSD,
    'UCCSD' : cc.uccsd.UCCSD
}

def recursive_remove_none(obj):
    if type(obj) == dict:
        return {k: recursive_remove_none(v) for k, v in obj.items() if v is not None}
    else:
        return obj

def get_vele_mat_outcore(mol, points):
    """
    Return shape (N, nao, nao)
    """
    auxmol = gto.fakemol_for_charges(points)
    vele_mat_file = df.outcore.cholesky_eri(mol, 'vele_mat_tmp.hdf5', auxmol=auxmol)
    return vele_mat_file

def get_ee_energy_density_outcore(mol, rdm2_file, vele_mat_file, mo_vals,
                                    mo_coeff, mem_chunk_size = 100):
    """
    Get the electron-electron repulsion energy density for a system and basis set (mol),
    for a given molecular structure with basis set (mol).
    Returns the electron-electron repulsion energy.
    NOTE: vele_mat, rdm2, and orb_vals must be in the same basis! (AO or MO)
    Args:
        mol (gto.Mole)
        rdm2 (4-dimensional array shape (nao, nao, nao, nao))
        vele_mat (3-dimensional array shape (nao, nao, N))
        orb_vals (2D array shape (N, nao))

    The following script is equivalent and easier to read (but slower):

    Vele_tmp = np.einsum('ijkl,pkl->pij', rdm2, vele_mat)
    tmp = np.einsum('pij,pj->pi', Vele_tmp, orb_vals)
    Vele = np.einsum('pi,pi->p', tmp, orb_vals)
    return 0.5 * Vele
    """
    #mu,nu,lambda,sigma->i,j,k,l; r->p
    import dask.array as da 
    mb_div_rdm1_size = mem_chunk_size*1e6 // (8*mo_coeff.shape[0]**2)
    ao_vele_mat = da.from_array(vele_mat_file, chunks=(-1,-1,mb_div_rdm1_size))
    mo_vele_mat = da.einsum('ij,jkp->pik', mo_coeff, ao_vele_mat)
    sqrt_mb_div_rdm1_size = int(np.sqrt(mb_div_rdm1_size))
    mo_rdm2 = da.from_array(rdm2_file,
        chunks=(-1,-1,sqrt_mb_div_rdm1_size,sqrt_mb_div_rdm1_size))
    vele_tmp = da.einsum('ijkl,pkl->pij', mo_rdm2, mo_vele_mat)
    tmp = da.einsum('pij,pj->pi', vele_tmp, mo_vals)
    Vele = da.einsum('pi,pi->p', tmp, mo_vals)
    return 0.5 * np.asarray(Vele)

def get_ee_energy_density_outcore2(mol, rdm2, vele_mat, orb_vals):
    ee_energy_density = np.array([])
    norb = orb_vals.shape[1]
    for vele_mat_chunk, orb_vals_chunk in vele_mat(orb_vals):
        num_mbytes = 8 * norb**4 // 1000000
        num_chunks = 5#int(num_mbytes // self.max_mem) + 1
        blksize = norb // num_chunks + 1
        sub_en = 0
        for p0, p1 in lib.prange(0, norb, blksize):
            sub_en += get_ee_energy_density(mol,
                                np.ascontiguousarray(
                                    rdm2[:,:,p0:p1,:].transpose(1,0,3,2)),
                                np.ascontiguousarray(
                                    vele_mat_chunk[:,:,p0:p1]),
                                orb_vals_chunk)
        ee_energy_density = np.append(ee_energy_density, sub_en)
    return ee_energy_density


class ElectronAnalyzer(ABC):

    calc_class = None
    calc_type = None

    def __init__(self, calc, require_converged=True, max_mem=None):
        # max_mem in MB
        if not isinstance(calc, self.calc_class):
            raise ValueError('Calculation must be instance of {}.'.format(self.calc_class))
        if calc.e_tot is None:
            raise ValueError('{} calculation must be complete.'.format(self.calc_type))
        if require_converged and not calc.converged:
            raise ValueError('{} calculation must be converged.'.format(self.calc_type))
        self.calc = calc
        self.mol = calc.mol
        self.conv_tol = self.calc.conv_tol
        self.converged = calc.converged
        self.max_mem = max_mem
        self.post_process()

    def as_dict(self):
        calc_props = {
            'conv_tol' : self.conv_tol,
            'converged' : self.converged,
            'e_tot' : self.e_tot,
            'mo_coeff' : self.mo_coeff,
            'mo_occ' : self.mo_occ,
        }
        data = {
            'coords' : self.grid.coords,
            'weights' : self.grid.weights,
            'ha_total' : self.ha_total,
            'fx_total' : self.fx_total,
            'ha_energy_density' : self.ha_energy_density,
            'fx_energy_density' : self.fx_energy_density,
            'xc_energy_density' : self.xc_energy_density,
            'ee_energy_density' : self.ee_energy_density,
            'rho_data' : self.rho_data,
            'tau_data' : self.tau_data
        }
        return {
            'mol' : gto.mole.pack(self.mol),
            'calc_type' : self.calc_type,
            'calc' : calc_props,
            'data' : data
        }

    def dump(self, fname):
        h5dict = recursive_remove_none(self.as_dict())
        lib.chkfile.dump(fname, 'analyzer', h5dict)

    @classmethod
    def from_dict(cls, analyzer_dict, max_mem = None):
        if analyzer_dict['calc_type'] != cls.calc_type:
            raise ValueError('Dict is from wrong type of calc, {} vs {}!'.format(
                                analyzer_dict['calc_type'], cls.calc_type))
        mol = gto.mole.unpack(analyzer_dict['mol'])
        mol.build()
        calc = cls.calc_class(mol)
        calc.__dict__.update(analyzer_dict['calc'])
        analyzer = cls(calc, require_converged = False, max_mem = max_mem)
        analyzer_dict['data'].pop('coords')
        analyzer_dict['data'].pop('weights')
        analyzer.__dict__.update(analyzer_dict['data'])
        return analyzer

    @classmethod
    def load(cls, fname, max_mem = None):
        analyzer_dict = lib.chkfile.load(fname, 'analyzer')
        return cls.from_dict(analyzer_dict, max_mem)

    def assign_num_chunks(self, ao_vals_shape, ao_vals_dtype):
        if self.max_mem == None:
            self.num_chunks = 1
            return self.num_chunks

        if ao_vals_dtype == np.float32:
            nbytes = 4
        elif ao_vals_dtype == np.float64:
            nbytes = 8
        else:
            raise ValueError('Wrong dtype for ao_vals')
        num_mbytes = nbytes * ao_vals_shape[0] * ao_vals_shape[1]**2 // 1000000
        self.num_chunks = int(num_mbytes // self.max_mem) + 1
        return self.num_chunks

    def post_process(self):
        # The child post process function must set up the RDMs
        self.grid = get_grid(self.mol)

        self.e_tot = self.calc.e_tot
        self.mo_coeff = self.calc.mo_coeff
        self.mo_occ = self.calc.mo_occ
        ###self.mo_energy = self.calc.mo_energy
        self.ao_vals = get_ao_vals(self.mol, self.grid.coords)
        self.mo_vals = get_mo_vals(self.ao_vals, self.mo_coeff)

        self.assign_num_chunks(self.ao_vals.shape, self.ao_vals.dtype)

        self.ao_vele_mat = get_vele_mat_generator(self.mol, self.grid.coords,
                                                  self.num_chunks)

        self.rdm1 = None
        self.rdm2 = None
        self.ao_data = None
        self.rho_data = None
        self.tau_data = None

        self.ha_total = None
        self.fx_total = None
        self.ee_total = None
        self.ha_energy_density = None
        self.fx_energy_density = None
        self.xc_energy_density = None
        self.ee_energy_density = None

    def get_ao_rho_data(self):
        if self.rho_data is None or self.tau_data is None:
            ao_data, self.rho_data = get_mgga_data(
                                        self.mol, self.grid, self.rdm1)
            self.tau_data = get_tau_and_grad(self.mol, self.grid,
                                            self.rdm1, ao_data)
        return self.rho_data, self.tau_data

    def perform_full_analysis(self):
        self.get_ao_rho_data()
        self.get_ha_energy_density()
        self.get_ee_energy_density()


class RHFAnalyzer(ElectronAnalyzer):

    calc_class = scf.hf.RHF
    calc_type = 'RHF'

    def as_dict(self):
        analyzer_dict = super(RHFAnalyzer, self).as_dict()
        analyzer_dict['calc']['mo_energy'] = self.mo_energy
        return analyzer_dict

    def post_process(self):
        super(RHFAnalyzer, self).post_process()
        self.rdm1 = np.array(self.calc.make_rdm1())
        self.mo_energy = self.calc.mo_energy
        self.jmat, self.kmat = scf.hf.get_jk(self.mol, self.rdm1)
        self.ha_total, self.fx_total = get_hf_coul_ex_total2(self.rdm1,
                                                    self.jmat, self.kmat)
        self.mo_vele_mat = get_vele_mat_generator(self.mol, self.grid.coords,
                                            self.num_chunks, self.mo_coeff)

    def get_ha_energy_density(self):
        if self.ha_energy_density is None:
            self.ha_energy_density = get_ha_energy_density2(
                                    self.mol, self.rdm1,
                                    self.ao_vele_mat, self.ao_vals
                                    )
        return self.ha_energy_density

    def get_fx_energy_density(self):
        if self.fx_energy_density is None:
            self.fx_energy_density = get_fx_energy_density2(
                                    self.mol, self.mo_occ,
                                    self.mo_vele_mat, self.mo_vals
                                    )
        return self.fx_energy_density

    def get_ee_energy_density(self):
        if self.ee_energy_density is None:
            self.ee_energy_density = self.get_ha_energy_density()\
                                     + self.get_fx_energy_density()
        return self.ee_energy_density


class UHFAnalyzer(ElectronAnalyzer):

    calc_class = scf.uhf.UHF
    calc_type = 'UHF'

    def as_dict(self):
        analyzer_dict = super(UHFAnalyzer, self).as_dict()
        analyzer_dict['calc']['mo_energy'] = self.mo_energy
        analyzer_dict['data']['fx_energy_density_u'] = self.fx_energy_density_u
        analyzer_dict['data']['fx_energy_density_d'] = self.fx_energy_density_d
        return analyzer_dict

    def post_process(self):
        super(UHFAnalyzer, self).post_process()
        self.rdm1 = np.array(self.calc.make_rdm1())
        self.mo_energy = self.calc.mo_energy
        self.fx_energy_density_u = None
        self.fx_energy_density_d = None
        self.jmat, self.kmat = scf.hf.get_jk(self.mol, self.rdm1)
        self.ha_total, self.fx_total = get_hf_coul_ex_total2(self.rdm1,
                                                    self.jmat, self.kmat)
        self.mo_vele_mat = [get_vele_mat_generator(self.mol, self.grid.coords,
                                            self.num_chunks, self.mo_coeff[0]),\
                            get_vele_mat_generator(self.mol, self.grid.coords,
                                            self.num_chunks, self.mo_coeff[1])]

    def get_ha_energy_density(self):
        if self.ha_energy_density is None:
            self.ha_energy_density = get_ha_energy_density2(
                                    self.mol, np.sum(self.rdm1, axis=0),
                                    self.ao_vele_mat, self.ao_vals
                                    )
        return self.ha_energy_density

    def get_fx_energy_density(self):
        if self.fx_energy_density is None:
            self.fx_energy_density_u = 0.5 * get_fx_energy_density2(
                                        self.mol, 2 * self.mo_occ[0],
                                        self.mo_vele_mat[0], self.mo_vals[0]
                                        )
            self.fx_energy_density_d = 0.5 * get_fx_energy_density2(
                                        self.mol, 2 * self.mo_occ[1],
                                        self.mo_vele_mat[1], self.mo_vals[1]
                                        )
            self.fx_energy_density = self.fx_energy_density_u + self.fx_energy_density_d
        return self.fx_energy_density

    def get_ee_energy_density(self):
        if self.ee_energy_density is None:
            self.ee_energy_density = self.get_ha_energy_density()\
                                     + self.get_fx_energy_density()
        return self.ee_energy_density


class RKSAnalyzer(RHFAnalyzer):

    def __init__(self, calc, require_converged=True, max_mem=None):
        if type(calc) != dft.rks.RKS:
            raise ValueError('Calculation must be RKS.')
        self.dft = calc
        hf = scf.RHF(self.dft.mol)
        hf.e_tot = self.dft.e_tot
        hf.mo_coeff = self.dft.mo_coeff
        hf.mo_occ = self.dft.mo_occ
        hf.mo_energy = self.dft.mo_energy
        hf.converged = self.dft.converged
        super(RKSAnalyzer, self).__init__(hf, require_converged, max_mem)


class UKSAnalyzer(UHFAnalyzer):

    def __init__(self, calc, require_converged=True, max_mem=None):
        if type(calc) != dft.uks.UKS:
            raise ValueError('Calculation must be UKS.')
        self.dft = calc
        hf = scf.UHF(self.dft.mol)
        hf.e_tot = self.dft.e_tot
        hf.mo_coeff = self.dft.mo_coeff
        hf.mo_occ = self.dft.mo_occ
        hf.mo_energy = self.dft.mo_energy
        hf.converged = self.dft.converged
        super(UKSAnalyzer, self).__init__(hf, require_converged, max_mem)


class CCSDAnalyzer(ElectronAnalyzer):

    calc_class = cc.ccsd.CCSD
    calc_type = 'CCSD'

    def as_dict(self):
        analyzer_dict = super(CCSDAnalyzer, self).as_dict()
        analyzer_dict['calc']['t1'] = self.calc.t1
        analyzer_dict['calc']['t2'] = self.calc.t2
        analyzer_dict['calc']['l1'] = self.calc.l1
        analyzer_dict['calc']['l2'] = self.calc.l2
        analyzer_dict['calc']['e_corr'] = self.calc.e_corr
        analyzer_dict['data']['ee_total'] = self.ee_total
        analyzer_dict['data']['ecorr_dens'] = self.ecorr_dens
        analyzer_dict['data']['e_tri'] = self.e_tri
        return analyzer_dict

    @classmethod
    def from_dict(cls, analyzer_dict, max_mem = None):
        if analyzer_dict['calc_type'] != cls.calc_type:
            raise ValueError('Dict is from wrong type of calc, {} vs {}!'.format(
                                analyzer_dict['calc_type'], cls.calc_type))
        mol = gto.mole.unpack(analyzer_dict['mol'])
        mol.build()
        hf = scf.hf.RHF(mol)
        hf.e_tot = analyzer_dict['calc'].pop('e_tot') - analyzer_dict['calc']['e_corr']
        calc = cls.calc_class(hf)
        calc.__dict__.update(analyzer_dict['calc'])
        analyzer = cls(calc, require_converged = False, max_mem = max_mem)
        analyzer.__dict__.update(analyzer_dict['data'])
        return analyzer

    def post_process(self):
        super(CCSDAnalyzer, self).post_process()
        self.mo_rdm1 = np.array(self.calc.make_rdm1())

        self.ao_rdm1 = transform_basis_1e(self.mo_rdm1, self.mo_coeff.transpose())
        self.rdm1 = self.ao_rdm1
        self.mo_vele_mat = get_vele_mat_generator(self.mol, self.grid.coords,
                                            self.num_chunks, self.mo_coeff)
        self.ee_total = None

        self.jmat, self.kmat = scf.hf.get_jk(self.mol, self.rdm1)
        self.ha_total, self.fx_total = get_hf_coul_ex_total2(self.rdm1,
                                                    self.jmat, self.kmat)

        self.ecorr_dens = None
        self.e_tri = None

    def get_ha_energy_density(self):
        if self.ha_energy_density is None:
            self.ha_energy_density = get_ha_energy_density2(
                                    self.mol, self.ao_rdm1,
                                    self.ao_vele_mat, self.ao_vals
                                    )
        return self.ha_energy_density

    def get_ee_energy_density(self):
        if self.ee_energy_density is None:
            self.ee_energy_density = get_lowmem_ee_energy(self.calc,
                                            self.mo_vele_mat, self.mo_vals,
                                            dm1 = self.mo_rdm1)
            self.ee_total = integrate_on_grid(self.ee_energy_density,
                                                self.grid.weights)
        return self.ee_energy_density

    def get_corr_energy_density(self):
        if self.ecorr_dens is None:
            t1, t2 = self.calc.t1, self.calc.t2
            tau = t2 + np.einsum('ia,jb->ijab', t1, t1)
            nocc, nvir = t1.shape
            ecorr_dens = []
            for vele_mat_chunk, orb_vals_chunk in self.mo_vele_mat(self.mo_vals):
                vele_mat_ov = vele_mat_chunk[:,:nocc,nocc:]
                orbvals_occ = orb_vals_chunk[:,:nocc]
                orbvals_vir = orb_vals_chunk[:,nocc:]
                ecorr_tmp = 2 * get_corr_energy_density(self.mol,
                                    tau, vele_mat_ov, orbvals_occ,
                                    orbvals_vir, direct = True)\
                            - get_corr_energy_density(self.mol,
                                    tau, vele_mat_ov, orbvals_occ,
                                    orbvals_vir, direct = False)
                ecorr_dens = np.append(ecorr_dens, ecorr_tmp)
            self.ecorr_dens = ecorr_dens
        return self.ecorr_dens

    def calc_pert_triples(self):
        self.e_tri = self.calc.ccsd_t()
        return self.e_tri

    def perform_full_analysis(self):
        super(CCSDAnalyzer, self).perform_full_analysis()
        self.get_corr_energy_density()


class UCCSDAnalyzer(ElectronAnalyzer):

    calc_class = cc.uccsd.UCCSD
    calc_type = 'UCCSD'

    def as_dict(self):
        analyzer_dict = super(UCCSDAnalyzer, self).as_dict()
        analyzer_dict['calc']['t1'] = self.calc.t1
        analyzer_dict['calc']['t2'] = self.calc.t2
        analyzer_dict['calc']['l1'] = self.calc.l1
        analyzer_dict['calc']['l2'] = self.calc.l2
        analyzer_dict['calc']['e_corr'] = self.calc.e_corr

        analyzer_dict['data']['ee_total'] = self.ee_total
        analyzer_dict['data']['ee_energy_density_uu'] = self.ee_energy_density_uu
        analyzer_dict['data']['ee_energy_density_ud'] = self.ee_energy_density_ud
        analyzer_dict['data']['ee_energy_density_dd'] = self.ee_energy_density_dd

        analyzer_dict['data']['ecorr_dens'] = self.ecorr_dens
        analyzer_dict['data']['ecorr_dens_uu'] = self.ecorr_dens_uu
        analyzer_dict['data']['ecorr_dens_ud'] = self.ecorr_dens_ud
        analyzer_dict['data']['ecorr_dens_dd'] = self.ecorr_dens_dd

        analyzer_dict['data']['e_tri'] = self.e_tri

        return analyzer_dict

    @classmethod
    def from_dict(cls, analyzer_dict, max_mem = None):
        if analyzer_dict['calc_type'] != cls.calc_type:
            raise ValueError('Dict is from wrong type of calc, {} vs {}!'.format(
                                analyzer_dict['calc_type'], cls.calc_type))
        mol = gto.mole.unpack(analyzer_dict['mol'])
        mol.build()
        hf = scf.uhf.UHF(mol)
        hf.e_tot = analyzer_dict['calc'].pop('e_tot') - analyzer_dict['calc']['e_corr']
        calc = cls.calc_class(hf)
        calc.__dict__.update(analyzer_dict['calc'])
        analyzer = cls(calc, require_converged = False, max_mem = max_mem)
        analyzer.__dict__.update(analyzer_dict['data'])
        return analyzer

    def post_process(self):
        super(UCCSDAnalyzer, self).post_process()
        self.mo_rdm1 = np.array(self.calc.make_rdm1())
        # PySCF does not currently do outcore RDM stuff,
        # so this would not be very helpful for saving memory
        #self.mo_rdm2_file = ext_uccsd_rdm.make_rdm2(self.calc, self.calc.t1,
        #                                    self.calc.t2, self.calc.l1,
        #                                    self.calc.l2)
        self.mo_rdm2 = self.calc.make_rdm2()
        self.mo_rdm2_file = lib.H5TmpFile()
        for i, name in enumerate(['dm2aa', 'dm2ab', 'dm2bb']):
            dm2 = self.mo_rdm2_file.create_dataset(name,
                                            self.mo_rdm2[i].shape,
                                            dtype=self.mo_rdm2[i].dtype)
            dm2[:,:,:,:] = self.mo_rdm2[i].transpose(1,0,3,2)
        self.mo_rdm2 = None

        # These are all three-tuples
        trans_mo_coeff = np.transpose(self.mo_coeff, axes=(0,2,1))
        self.ao_rdm1 = transform_basis_1e(self.mo_rdm1, trans_mo_coeff)
        self.rdm1 = self.ao_rdm1
        self.ee_total = None

        self.jmat, self.kmat = scf.hf.get_jk(self.mol, self.rdm1)
        self.ha_total, self.fx_total = get_hf_coul_ex_total2(self.rdm1,
                                                    self.jmat, self.kmat)

        self.mo_vele_mat = [get_vele_mat_generator(self.mol, self.grid.coords,
                                            self.num_chunks, self.mo_coeff[0]),\
                            get_vele_mat_generator(self.mol, self.grid.coords,
                                            self.num_chunks, self.mo_coeff[1])]

        self.ee_energy_density_uu = None
        self.ee_energy_density_ud = None
        self.ee_energy_density_dd = None

        self.ecorr_dens = None
        self.ecorr_dens_uu = None
        self.ecorr_dens_ud = None
        self.ecorr_dens_dd = None

        self.e_tri = None

    def get_ha_energy_density(self):
        if self.ha_energy_density is None:
            self.ha_energy_density = get_ha_energy_density2(
                                    self.mol, np.sum(self.ao_rdm1, axis=0),
                                    self.ao_vele_mat, self.ao_vals
                                    )
        return self.ha_energy_density

    def get_ee_energy_density(self):
        if self.ee_energy_density is None:
            self.ee_energy_density_uu = get_ee_energy_density_outcore2(
                                    self.mol, self.mo_rdm2_file['dm2aa'],
                                    self.mo_vele_mat[0], self.mo_vals[0])
            self.ee_energy_density_ud = get_ee_energy_density_outcore2(
                                    self.mol, self.mo_rdm2_file['dm2ab'],
                                    self.mo_vele_mat[1], self.mo_vals[0])
            self.ee_energy_density_dd = get_ee_energy_density_outcore2(
                                    self.mol, self.mo_rdm2_file['dm2bb'],
                                    self.mo_vele_mat[1], self.mo_vals[1])
            self.ee_energy_density = self.ee_energy_density_uu\
                                    + 2 * self.ee_energy_density_ud\
                                    + self.ee_energy_density_dd
            self.ee_total = integrate_on_grid(self.ee_energy_density,
                                                self.grid.weights)
        return self.ee_energy_density

    def get_corr_energy_density(self):
        if self.ecorr_dens is None:
            t1, t2 = self.calc.t1, self.calc.t2
            tauaa, tauab, taubb = cc.uccsd.make_tau(t2, t1, t1)
            
            nocca, nvira = t1[0].shape
            noccb, nvirb = t1[1].shape

            ecorr_dens_uu = []
            for vele_mat_chunk, orb_vals_chunk in self.mo_vele_mat[0](self.mo_vals[0]):
                vele_mat_ov = vele_mat_chunk[:,:nocca,nocca:]
                orbvals_occ = orb_vals_chunk[:,:nocca]
                orbvals_vir = orb_vals_chunk[:,nocca:]
                ecorr_tmp = get_corr_energy_density(self.mol,
                                    tauaa, vele_mat_ov, orbvals_occ,
                                    orbvals_vir, direct = True)\
                            - get_corr_energy_density(self.mol,
                                    tauaa, vele_mat_ov, orbvals_occ,
                                    orbvals_vir, direct = False)
                ecorr_dens_uu = np.append(ecorr_dens_uu, ecorr_tmp)

            ecorr_dens_dd = []
            for vele_mat_chunk, orb_vals_chunk in self.mo_vele_mat[1](self.mo_vals[1]):
                vele_mat_ov = vele_mat_chunk[:,:noccb,noccb:]
                orbvals_occ = orb_vals_chunk[:,:noccb]
                orbvals_vir = orb_vals_chunk[:,noccb:]
                ecorr_tmp = get_corr_energy_density(self.mol,
                                    taubb, vele_mat_ov, orbvals_occ,
                                    orbvals_vir, direct = True)\
                            - get_corr_energy_density(self.mol,
                                    taubb, vele_mat_ov, orbvals_occ,
                                    orbvals_vir, direct = False)
                ecorr_dens_dd = np.append(ecorr_dens_dd, ecorr_tmp)

            ecorr_dens_ud = []
            for vele_mat_chunk, orb_vals_chunk in self.mo_vele_mat[1](self.mo_vals[0]):
                vele_mat_ov = vele_mat_chunk[:,:noccb,noccb:]
                orbvals_occ = orb_vals_chunk[:,:nocca]
                orbvals_vir = orb_vals_chunk[:,nocca:]
                ecorr_tmp = get_corr_energy_density(self.mol,
                                    tauab, vele_mat_ov, orbvals_occ,
                                    orbvals_vir, direct = True)
                ecorr_dens_ud = np.append(ecorr_dens_ud, ecorr_tmp)

            self.ecorr_dens = 0.25 * ecorr_dens_uu + 0.25 * ecorr_dens_dd \
                +  ecorr_dens_ud
            self.ecorr_dens_uu = ecorr_dens_uu
            self.ecorr_dens_ud = ecorr_dens_ud
            self.ecorr_dens_dd = ecorr_dens_dd

        return self.ecorr_dens

    def calc_pert_triples(self):
        self.e_tri = self.calc.ccsd_t()
        return self.e_tri

    def perform_full_analysis(self):
        super(UCCSDAnalyzer, self).perform_full_analysis()
        self.get_corr_energy_density()
