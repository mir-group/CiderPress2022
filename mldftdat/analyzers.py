from pyscf import scf, dft, gto, ao2mo, df, lib, fci, cc
from pyscf.dft.numint import eval_ao, eval_rho
from pyscf.dft.gen_grid import Grids
from pyscf.pbc.tools.pyscf_ase import atoms_from_ase
from mldftdat.pyscf_utils import *
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
    def load(cls, fname):
        analyzer_dict = lib.chkfile.load(fname, 'analyzer')
        return cls.from_dict(analyzer_dict)

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
        print("NUMBER OF CHUNKS", self.num_chunks, self.ao_vals.dtype, psutil.virtual_memory().available // 1e6)

        if self.num_chunks > 1:
            self.ao_vele_mat = get_vele_mat_generator(self.mol, self.grid.coords,
                                                self.num_chunks, self.ao_vals)
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
                                                self.num_chunks, self.mo_vals,
                                                self.mo_coeff)
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
                                                self.num_chunks, self.mo_vals[0],
                                                self.mo_coeff[0]),\
                                get_vele_mat_generator(self.mol, self.grid.coords,
                                                self.num_chunks, self.mo_vals[1],
                                                self.mo_coeff[1])]
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

        self.ee_energy_density_uu = None
        self.ee_energy_density_ud = None
        self.ee_energy_density_dd = None

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
