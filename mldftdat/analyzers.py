from pyscf import scf, dft, gto, ao2mo, df, lib, fci, cc
from pyscf.dft.numint import eval_ao, eval_rho, eval_rho2
from pyscf.dft.gen_grid import Grids
from pyscf.pbc.tools.pyscf_ase import atoms_from_ase
from mldftdat.pyscf_utils import *
import numpy as np
from abc import ABC, abstractmethod, abstractproperty
from io import BytesIO
import psutil
from pyscf.scf.hf import get_jk


def recursive_remove_none(obj):
    if type(obj) == dict:
        return {k: recursive_remove_none(v) for k, v in obj.items() if v is not None}
    else:
        return obj


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
        self.mol = calc.mol.build()
        self.conv_tol = self.calc.conv_tol
        self.converged = calc.converged
        self.max_mem = max_mem
        print('PRIOR TO POST PROCESS', psutil.virtual_memory().available // 1e6)
        self.post_process()
        print('FINISHED POST PROCESS', psutil.virtual_memory().available // 1e6)

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
        mol = mol_from_dict(analyzer_dict['mol'])
        calc = get_scf(analyzer_dict['calc_type'], mol, analyzer_dict['calc'])
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
        print("NUMBER OF CHUNKS", self.calc_type, self.num_chunks, self.ao_vals.dtype, psutil.virtual_memory().available // 1e6)

        if self.num_chunks > 1:
            self.ao_vele_mat = get_vele_mat_generator(self.mol, self.grid.coords,
                                                self.num_chunks)
        else:
            self.ao_vele_mat = get_vele_mat(self.mol, self.grid.coords)
            print('AO VELE MAT', self.ao_vele_mat.nbytes, self.ao_vele_mat.shape)

        print("MEM NOW", psutil.virtual_memory().available // 1e6)

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
        if self.num_chunks > 1:
            self.mo_vele_mat = get_vele_mat_generator(self.mol, self.grid.coords,
                                                self.num_chunks, self.mo_coeff)
        else:
            self.mo_vele_mat = get_mo_vele_mat(self.ao_vele_mat, self.mo_coeff)
            print("MO VELE MAT", self.mo_vele_mat.nbytes, psutil.virtual_memory().available // 1e6)

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

    def _get_rdm2(self):
        if self.rdm2 is None:
            self.eri_ao = self.mol.intor('int2e')
            self.rdm2 = make_rdm2_from_rdm1(self.rdm1)
        return self.rdm2

    def _get_ee_energy_density_slow(self):
        rdm2 = self._get_rdm2()
        return get_ee_energy_density2(self.mol, self.rdm2,
                                        self.ao_vele_mat, self.ao_vals)


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
        if self.num_chunks > 1:
            self.mo_vele_mat = [get_vele_mat_generator(self.mol, self.grid.coords,
                                                self.num_chunks, self.mo_coeff[0]),\
                                get_vele_mat_generator(self.mol, self.grid.coords,
                                                self.num_chunks, self.mo_coeff[1])]
        else:
            self.mo_vele_mat = get_mo_vele_mat(self.ao_vele_mat, self.mo_coeff)

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

    def _get_rdm2(self):
        if self.rdm2 is None:
            self.eri_ao = self.mol.intor('int2e')
            self.rdm2 = make_rdm2_from_rdm1_unrestricted(self.rdm1)
        return self.rdm2

    def _get_ee_energy_density_slow(self):
        rdm2 = self._get_rdm2()
        euu = get_ee_energy_density2(self.mol, self.rdm2[0],
                                    self.ao_vele_mat, self.ao_vals)
        eud = get_ee_energy_density2(self.mol, self.rdm2[1],
                                    self.ao_vele_mat, self.ao_vals)
        edd = get_ee_energy_density2(self.mol, self.rdm2[2],
                                    self.ao_vele_mat, self.ao_vals)
        return euu + 2 * eud + edd


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
        mol = mol_from_dict(analyzer_dict['mol'])
        calc = get_ccsd(analyzer_dict['calc_type'], mol, analyzer_dict['calc'])
        analyzer = cls(calc, require_converged = False, max_mem = max_mem)
        analyzer.__dict__.update(analyzer_dict['data'])
        return analyzer

    def post_process(self):
        super(CCSDAnalyzer, self).post_process()
        self.eri_ao = self.mol.intor('int2e')
        self.mo_rdm1 = np.array(self.calc.make_rdm1())
        self.mo_rdm2 = np.array(self.calc.make_rdm2())

        self.ao_rdm1 = transform_basis_1e(self.mo_rdm1, self.mo_coeff.transpose())
        self.ao_rdm2 = transform_basis_2e(self.mo_rdm2, self.mo_coeff.transpose())
        self.eri_mo = transform_basis_2e(self.eri_ao, self.mo_coeff)
        self.rdm1 = self.ao_rdm1
        self.rdm2 = self.ao_rdm2
        self.ee_total = get_ccsd_ee(self.mo_rdm2, self.eri_mo)

        self.jmat, self.kmat = scf.hf.get_jk(self.mol, self.rdm1)
        self.ha_total, self.fx_total = get_hf_coul_ex_total2(self.rdm1,
                                                    self.jmat, self.kmat)

        self.mo_vele_mat = get_vele_mat_generator(self.mol, self.grid.coords,
                                            self.num_chunks, self.mo_coeff)

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
            self.ee_energy_density = get_ee_energy_density2(
                                    self.mol, self.ao_rdm2,
                                    self.ao_vele_mat, self.ao_vals)
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
        mol = mol_from_dict(analyzer_dict['mol'])
        calc = get_ccsd(analyzer_dict['calc_type'], mol, analyzer_dict['calc'])
        analyzer = cls(calc, require_converged = False, max_mem = max_mem)
        analyzer.__dict__.update(analyzer_dict['data'])
        return analyzer

    def post_process(self):
        super(UCCSDAnalyzer, self).post_process()
        self.eri_ao = self.mol.intor('int2e')
        self.mo_rdm1 = np.array(self.calc.make_rdm1())
        self.mo_rdm2 = np.array(self.calc.make_rdm2())

        # These are all three-tuples
        trans_mo_coeff = np.transpose(self.mo_coeff, axes=(0,2,1))
        self.ao_rdm1 = transform_basis_1e(self.mo_rdm1, trans_mo_coeff)
        self.ao_rdm2 = transform_basis_2e(self.mo_rdm2, trans_mo_coeff)
        eri_ao_lst = [self.eri_ao] * 3
        self.eri_mo = transform_basis_2e(eri_ao_lst, self.mo_coeff)
        self.rdm1 = self.ao_rdm1
        self.rdm2 = self.ao_rdm2
        self.ee_total = get_ccsd_ee(self.mo_rdm2, self.eri_mo)

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
            self.ee_energy_density_uu = get_ee_energy_density2(
                                    self.mol, self.ao_rdm2[0],
                                    self.ao_vele_mat, self.ao_vals)
            self.ee_energy_density_ud = get_ee_energy_density2(
                                    self.mol, self.ao_rdm2[1],
                                    self.ao_vele_mat, self.ao_vals)
            self.ee_energy_density_dd = get_ee_energy_density2(
                                    self.mol, self.ao_rdm2[2],
                                    self.ao_vele_mat, self.ao_vals)
            self.ee_energy_density = self.ee_energy_density_uu\
                                    + 2 * self.ee_energy_density_ud\
                                    + self.ee_energy_density_dd
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


from mldftdat.data import density_similarity

class XCPotentialAnalyzer():

    def __init__(self, analyzer):
        """
        analyzer: CCSDAnalyzer
        """
        self.analyzer = analyzer

    def get_no_rdm(self):
        """
        Get the natural orbital coefficients, occupancies,
        2-RDM, and ERI in terms of the MOs.
        """
        no_occ, no_coeff = np.linalg.eigh(self.analyzer.mo_rdm1)
        no_rdm2 = transform_basis_2e(self.analyzer.mo_rdm2, no_coeff)
        no_eri = transform_basis_2e(self.analyzer.eri_mo, no_coeff)
        return no_occ, no_coeff, no_rdm2, no_eri

    def compute_lambda_ij(self):
        """
        Compute the wavefunction part of the XC potential formula.
        Returns the wf_term as well as mu_max (most negative
        eigenvalue of G), h1e_ao, and the Hartree potential.
        """
        from scipy.linalg import eigvalsh
        no_occ, no_coeff, no_rdm2, no_eri = self.get_no_rdm()
        print(no_occ)
        #eri_trace = np.sum(no_eri * no_rdm2, axis = (2,3))
        #eri_trace = eri_trace + eri_trace.T
        nno = no_occ.shape[0]
        eri_trace = np.zeros((nno, nno))
        for i in range(nno):
            for j in range(nno):
                eri_trace[i,j] = np.sum(no_rdm2[j,:,:,:] * no_eri[i,:,:,:])
        #eri_trace *= 2
        h1e_ao = self.analyzer.calc._scf.get_hcore(self.analyzer.mol)
        #no_ao_coeff = np.dot()
        h1e_no = transform_basis_1e(h1e_ao, self.analyzer.mo_coeff)
        h1e_no = transform_basis_1e(h1e_no, no_coeff)
        lambda_ij = np.dot(h1e_no, np.diag(no_occ)) + eri_trace
        print(np.linalg.eigvalsh(lambda_ij))
        vs_xc = self.analyzer.get_ee_energy_density()\
                - self.analyzer.get_ha_energy_density()
        ao_data, rho_data = get_mgga_data(
            self.analyzer.mol, self.analyzer.grid, self.analyzer.ao_rdm1)
        ao = ao_data[0]
        no = np.dot(np.dot(ao, self.analyzer.mo_coeff), no_coeff)
        vs_xc *= 2 / (rho_data[0] + 1e-20)
        tau_rho = rho_data[5] / (rho_data[0] + 1e-20)
        eps_wf = np.dot(no, lambda_ij)
        eps_wf = np.einsum('ni,ni->n', no, eps_wf)
        eps_wf /= (rho_data[0] + 1e-20)
        fac = np.sqrt(np.outer(no_occ, no_occ) + 1e-10)
        mu_max = np.max(np.linalg.eigvalsh(lambda_ij / fac))
        mu_max = np.max(eigvalsh(lambda_ij, np.diag(no_occ + 1e-10)))
        print('MU_MAX', mu_max)
        return vs_xc - eps_wf + tau_rho, mu_max, h1e_ao,\
            2 * self.analyzer.get_ha_energy_density() / (rho_data[0] + 1e-20)
        #return eps_wf, mu_max, h1e_ao,\
        #    2 * self.analyzer.get_ha_energy_density() / (rho_data[0] + 1e-20)

    def initial_dft_guess(self, mu_max, return_mf = False):
        from pyscf import lib
        mf = run_scf(self.analyzer.mol, 'RKS', functional = 'PBE')
        ao_data, rho_data = get_mgga_data(self.analyzer.mol,
            self.analyzer.grid, mf.make_rdm1())
        mo_data = np.dot(ao_data[0], mf.mo_coeff)**2
        ehomo = np.max(mf.mo_energy[mf.mo_occ > 1e-10])
        mo_energy = mf.mo_energy + mu_max - ehomo
        tau_rho = rho_data[5] / (rho_data[0] + 1e-20)
        print('KS', mf.mo_energy, ehomo)
        eps_ks = np.dot(mo_data, mf.mo_occ * mo_energy)
        eps_ks /= (rho_data[0] + 1e-20)
        dm = mf.make_rdm1()
        if return_mf:
            dm = lib.tag_array(dm, mo_coeff=mf.mo_coeff, mo_occ=mf.mo_occ,
                               mo_energy=mf.mo_energy)
            return rho_data, dm, mf
        else:
            return eps_ks - tau_rho, rho_data, dm

    def initialize_for_scf(self):
        self.wf_term, self.mu_max, self.h1e_ao, self.vha = \
            self.compute_lambda_ij()
        self.ao_data = eval_ao(self.analyzer.mol,
                               self.analyzer.grid.coords,
                               deriv = 2)
        self.analyzer.get_ao_rho_data()

    def solve_vxc(self):
        from scipy.linalg import eigh
        mol = self.analyzer.mol
        wf_term, mu_max, h1e_ao, vha = self.compute_lambda_ij()
        ks_term, ks_rho_data, dm = self.initial_dft_guess(mu_max)
        nelec = mol.nelectron
        ovlp = mol.get_ovlp()
        vxc = wf_term + ks_term
        vxc_old = np.zeros(vxc.shape)
        weight = self.analyzer.grid.weights
        ao_data = eval_ao(self.analyzer.mol, self.analyzer.grid.coords,
                          deriv = 2)
        ao = ao_data[0]
        iter_num = 0
        init_sim = density_similarity(self.analyzer.rho_data,
            ks_rho_data, self.analyzer.grid, mol, exponent = 1, inner_r = 0.01)
        while iter_num < 4000 and np.dot(np.abs(vxc - vxc_old), weight) > 1e-8:
            vxc_old = vxc
            vrho = vxc + vha
            #vrho = vxc
            aow = np.einsum('pi,p->pi', ao, .5*weight*vrho)
            vmat = np.dot(ao.T, aow)
            #print(np.linalg.norm(vmat), np.linalg.norm(h1e_ao))
            h1e = h1e_ao + vmat + vmat.T# + get_jk(mol, dm)[0]
            energy, coeff = eigh(h1e, ovlp)
            occ = 0 * energy
            occ[:nelec//2] = 2
            mocc = coeff[:,occ>0]
            dm = np.dot(mocc*occ[occ>0], mocc.conj().T)
            ehomo = np.max(energy[occ > 1e-10])
            #print(ehomo, mu_max, 2 * (ehomo - mu_max))
            energy += mu_max - ehomo
            rho_data = eval_rho2(mol, ao_data, coeff, occ, xctype='MGGA')
            tau_rho = rho_data[5] / (rho_data[0] + 1e-20)
            mo_data = np.dot(ao, coeff)**2
            eps_ks = np.dot(mo_data, occ * energy)
            eps_ks /= rho_data[0] + 1e-20
            ks_term = eps_ks - tau_rho
            ds = density_similarity(self.analyzer.rho_data,
                    rho_data, self.analyzer.grid, mol,
                    exponent = 1, inner_r = 0.01)
            if ds[0] > 0.01:
                vxc = (wf_term + ks_term) * 0.001 + 0.999 * vxc
            elif ds[0] > 0.004:
                vxc = (wf_term + ks_term) * 0.01 + 0.99 * vxc
            elif ds[0] > 0.0008:
                vxc = (wf_term + ks_term) * 0.1 + 0.9 * vxc
            else:
                vxc = wf_term + ks_term
            iter_num += 1
            print('iter', iter_num, np.dot(np.abs(vxc - vxc_old), weight), ds)
        print('iter', iter_num, np.dot(np.abs(vxc - vxc_old), weight))
        final_sim = density_similarity(self.analyzer.rho_data,
            rho_data, self.analyzer.grid, mol,
            exponent = 1, inner_r = 0.01)
        print(init_sim, final_sim)
        print(np.dot(rho_data[0], weight))
        print(np.dot(ks_rho_data[0], weight))
        return vxc, rho_data, np.dot(np.abs(vxc - vxc_old), weight)

    def get_veff(self, mol, dm, dm_last=None, vhf_last=None,
                 hermi=1, vhfopt=None):

        mo_occ = dm.mo_occ
        mo_coeff = dm.mo_coeff
        mo_energy = dm.mo_energy
        mu_max = self.mu_max
        wf_term = self.wf_term
        ao_data = self.ao_data
        weight = self.analyzer.grid.weights
        ao = ao_data[0]

        ehomo = np.max(mo_energy[mo_occ > 0])
        mo_energy = mo_energy + mu_max - ehomo
        rho_data = eval_rho2(mol, ao_data, mo_coeff, mo_occ, xctype='MGGA')
        tau_rho = rho_data[5] / (rho_data[0] + 1e-20)
        mo_data = np.dot(ao, mo_coeff)**2
        eps_ks = np.dot(mo_data, mo_occ * mo_energy)
        eps_ks /= rho_data[0] + 1e-20
        ks_term = eps_ks - tau_rho
        vrho = ks_term + wf_term

        aow = np.einsum('pi,p->pi', ao, .5*weight*vrho)
        vmat = np.dot(ao.T, aow)

        return vmat + vmat.T + get_jk(mol, dm)[0]

    def scf(self, **kwargs):

        from pyscf.lib import logger

        import time
        self.initialize_for_scf()
        rho_data, dm0, mf = self.initial_dft_guess(self.mu_max, return_mf=True)

        cput0 = (time.clock(), time.time())

        from mldftdat.external.pyscf_scf_vxcopt import kernel

        ds_init = density_similarity(self.analyzer.rho_data,
                        rho_data, self.analyzer.grid,
                        self.analyzer.mol,
                        exponent = 1, inner_r = 0.01)

        def check_convergence(kwargs):
            rho_data = eval_rho2(self.analyzer.mol,
                             self.ao_data, kwargs['mo_coeff'],
                             kwargs['mo_occ'], xctype='MGGA')
            ds = density_similarity(self.analyzer.rho_data,
                        rho_data, self.analyzer.grid,
                        self.analyzer.mol,
                        exponent = 1, inner_r = 0.01)[0]
            print('CONV CHECK', ds)
            return ds < 0.001

        mf.check_convergence = check_convergence

        mf.dump_flags()
        mf.build(mf.mol)
        self.converged, self.e_tot, \
                self.mo_energy, self.mo_coeff, self.mo_occ = \
                kernel(self, mf, mf.conv_tol, mf.conv_tol_grad,
                       dm0=dm0, callback=mf.callback,
                       conv_check=mf.conv_check, **kwargs)

        logger.timer(mf, 'VXCOPT', *cput0)
        rho_data = eval_rho2(self.analyzer.mol,
                             self.ao_data, self.mo_coeff,
                             self.mo_occ, xctype='MGGA')
        self.ds = density_similarity(self.analyzer.rho_data,
                    rho_data, self.analyzer.grid,
                    self.analyzer.mol,
                    exponent = 1, inner_r = 0.01)
        mf._finalize()
        return self.ds, ds_init
