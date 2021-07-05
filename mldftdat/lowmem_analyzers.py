from pyscf import scf, dft, gto, ao2mo, df, lib
from pyscf.dft.numint import eval_ao, eval_rho
from pyscf.dft.gen_grid import Grids
from pyscf.pbc.tools.pyscf_ase import atoms_from_ase
from mldftdat.pyscf_utils import *
import numpy as np
from abc import ABC, abstractmethod, abstractproperty
from io import BytesIO
import psutil

"""
Module for loading, storing, and analyzing PySCF HF/DFT calculations.
"""

CALC_TYPES = {
    'RHF'   : scf.hf.RHF,
    'UHF'   : scf.uhf.UHF,
    'RKS'   : dft.rks.RKS,
    'UKS'   : dft.uks.UKS
}

def recursive_remove_none(obj):
    if type(obj) == dict:
        return {k: recursive_remove_none(v) for k, v in obj.items() if v is not None}
    else:
        return obj


class ElectronAnalyzer(ABC):
    """
    A class for storing a PySCF electronic structure calculation
    and computing various distributions on a real-space grid,
    such as density, exchange (correlation) energy density, Coulomb
    energy density, etc.

    Attributes:

        ## IN INTIALIZER ##
        calc: PySCF calculation object
        mol: pyscf.gto.Mole object on which calculation was performed
        conv_tol: convergence tolerance of SCF calculation in Ha
        converged (bool): Whether the PySCF calculation is converged
        max_mem: Maximum memory chunk size for use in calculating
            energy density quantities, in MB.

        ## IN POST PROCESS ##
        grid: pyscf.dft.gen_grid.Grids object onto which distributions
            such as density and exchange energy density are projected
        e_tot: total energy of the calculation
        mo_coeff: Molecular orbital coefficients for SCF calculation
        mo_occ: Occupations of molecular orbitals
        ao_vals: Atomic orbitals projected onto grid
        mo_vals: Molecular orbitals projected onto grid
        num_chunks (int): Number of chunks for memory-intensive routines
        ao_vele_mat: Projection of the Coulomb repulsion tensor into
            real space on one side.

        ## COMPUTABLE DATA ## (initialized to None)
        rho_data (array (6, ngrid)): The density (0),
            density gradient (1,2,3),
            density Laplacian (4), and kinetic energy density (5) for the
            1-RDM of the system.
        tau_data (array, (4, ngrid)): The kinetic energy density (0)
            and kinetic energy density gradient (1,2,3) for the 1-RDM
            of the system.
    """

    calc_class = None
    calc_type = None

    def __init__(self, calc, require_converged=True, max_mem=None):
        """
        Args:
            calc: A PySCF object of type calc_type
            require_converged: Check and require that the PySCF
                calculation is electronically converged.
            max_mem: Maximum memory in MB that the class is allowed
                to use in performing calculations. This is necessary
                because many routines performed here are memory intensive
                and must be split into chunks for large systems.
        """
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
        """
        Save all the propeties of self to a dict.
        """
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
        """
        Dump self to an hdf5 file called fname.

        Args:
            fname (str): Name of file to dump to
        """
        h5dict = recursive_remove_none(self.as_dict())
        lib.chkfile.dump(fname, 'analyzer', h5dict)

    @classmethod
    def from_dict(cls, analyzer_dict, max_mem = None):
        """
        Initialize an instance of cls from analyzer_dict, which should be
        generated by the as_dict method.

        Args:
            analyzer_dict (dict): dict form of cls
            max_mem: See __init__

        Returns:
            cls instance initialized from analyzer_dict
        """
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
        """
        Load instance of cls from hdf5
        Args:
            fname (str): Name of file from which to load
            max_mem: See __init__
        """
        analyzer_dict = lib.chkfile.load(fname, 'analyzer')
        return cls.from_dict(analyzer_dict, max_mem)

    def assign_num_chunks(self, ao_vals_shape, ao_vals_dtype):
        """
        Internal method for assigning the number of chunks into
        which to split the vele_mat array, which is the (nao, nao, ngrid)
        shaped matrix projecting the Coulomb repulsion tensor into
        real space on one side.

        Args:
            ao_vals_shape: Shape of the ao_vals object
                (ngrid, nao, ...)
            ao_vals_dtype: Should be np.float32 or np.float64

        Returns:
            Sets self.num_chunks and also returns it.
        """
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
        """
        Called by the initializer to set up all the basic data needed
        to perform computations within the ElectronAnalyzer.
        Sets values that may be calculated to None. Generally speaking,
        computationally expensive routines first check if a given quantity
        has already been saved to the analyzer. If so, it is returned without
        further computation. If not, it is calculated. post_process is overridden
        by subclasses to add more data that can be calcualted.
        """
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
        """
        Calculate, set, and return the rho_data and tau_data objects
        Returns:
            rho_data (array (6, ngrid)): The density (0),
                density gradient (1,2,3),
                density Laplacian (4), and kinetic energy density (5) for the
                1-RDM of the system.
            tau_data (array, (4, ngrid)): The kinetic energy density (0)
                and kinetic energy density gradient (1,2,3) for the 1-RDM
                of the system.
        """
        if self.rho_data is None or self.tau_data is None:
            ao_data, self.rho_data = get_mgga_data(
                                        self.mol, self.grid, self.rdm1)
            self.tau_data = get_tau_and_grad(self.mol, self.grid,
                                            self.rdm1, ao_data)
        return self.rho_data, self.tau_data

    def perform_full_analysis(self):
        """
        Perform all the main distribution calculations provided by this analyzer.
        Overridden by subclasses to compute additional data.
        """
        self.get_ao_rho_data()
        self.get_ha_energy_density()
        self.get_ee_energy_density()


class RHFAnalyzer(ElectronAnalyzer):
    """
    Analyzer for restricted Hartree Fock. This can also be used for
    restricted Kohn Sham for computating exact exchange energy densities
    and other orbital-based quantities, but BE AWARE that when loaded
    from hdf5, the calc object will be set to an RHF and no longer
    return the correct DFT total energy or XC energy.

    Attributes:

        ## ALL ATTRIBUTES OF ELECTRON ANALYZER ##
        See ElectronAnalyzer

        ## IN POST PROCESS ##
        rdm1: 1-electron reduced density matrix in atomic orbital space
        mo_energy: Single-particle energies of molecular orbitals
        jmat, kmat: Direct and Exchange Coulomb contributions to the
            1-particle Hamiltonian
        ha_total, fx_total: Direct and exchange Coulomb energies of the
            Slater determinant
        mo_vele_mat: Same as ao_vele_mat but in molecular orbital basis

        ## COMPUTABLE DATA ## (initialized to None)
        ha_energy_density: Classical Hartree energy density on grid
        fx_energy_density: HF exchange energy density on grid
        ee_energy_density: Sum of ha_energy_density and fx_energy_density
    """

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
        """
        Return the classical Hartree energy density
        e_{Ha} (r_1) = (1/2) \int dr_2 n(r_1) n(r_2) / |r_1 - r_2|.
        """
        if self.ha_energy_density is None:
            self.ha_energy_density = get_ha_energy_density2(
                                    self.mol, self.rdm1,
                                    self.ao_vele_mat, self.ao_vals
                                    )
        return self.ha_energy_density

    def get_fx_energy_density(self):
        """
        Return the HF exchange energy density
        e_X (r_1) = -(1/2) \int dr_2 |n_1(r_1, r_2)|^2 / |r_1 - r_2|.
        """
        if self.fx_energy_density is None:
            self.fx_energy_density = get_fx_energy_density2(
                                    self.mol, self.mo_occ,
                                    self.mo_vele_mat, self.mo_vals
                                    )
        return self.fx_energy_density

    def get_ee_energy_density(self):
        """
        Returns the sum of E_{Ha} and E_{X}, i.e. the Coulomb repulsion
        energy of the HF or KS Slater determinant.
        """
        if self.ee_energy_density is None:
            self.ee_energy_density = self.get_ha_energy_density()\
                                     + self.get_fx_energy_density()
        return self.ee_energy_density


class UHFAnalyzer(ElectronAnalyzer):
    """
    Analyzer for Unrestricted Hartree Fock. This can also be used for
    Unrestricted Kohn Sham for computating exact exchange energy densities
    and other orbital-based quantities, but BE AWARE that when loaded
    from hdf5, the calc object will be set to a UHF and no longer
    return the correct DFT total energy or XC energy.

    Attributes:

        ## ALL ATTRIBUTES OF ELECTRON ANALYZER ##
        See ElectronAnalyzer

        ## IN POST PROCESS ##
        rdm1: 1-electron reduced density matrix in atomic orbital space
        mo_energy: Single-particle energies of molecular orbitals
        jmat, kmat: Direct and Exchange Coulomb contributions to the
            1-particle Hamiltonian
        ha_total, fx_total: Direct and exchange Coulomb energies of the
            Slater determinant
        mo_vele_mat: Same as ao_vele_mat but in molecular orbital basis

        ## COMPUTABLE DATA ## (initialized to None)
        ha_energy_density: Classical Hartree energy density on grid
        fx_energy_density: HF exchange energy density on grid
            (_u and _d spin components are also stored)
        ee_energy_density: Sum of ha_energy_density and fx_energy_density
    """

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
        """
        Return the classical Hartree energy density
        e_{Ha} (r_1) = (1/2) \int dr_2 n(r_1) n(r_2) / |r_1 - r_2|.
        """
        if self.ha_energy_density is None:
            self.ha_energy_density = get_ha_energy_density2(
                                    self.mol, np.sum(self.rdm1, axis=0),
                                    self.ao_vele_mat, self.ao_vals
                                    )
        return self.ha_energy_density

    def get_fx_energy_density(self):
        """
        Return the HF exchange energy density
        e_X (r_1) = -(1/2) \int dr_2 |n_1(r_1, r_2)|^2 / |r_1 - r_2|.
        Saves the spin-separated contributions, which can be accessed
        directly.
        """
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
        """
        Returns the sum of E_{Ha} and E_{X}, i.e. the Coulomb repulsion
        energy of the HF or KS Slater determinant.
        """
        if self.ee_energy_density is None:
            self.ee_energy_density = self.get_ha_energy_density()\
                                     + self.get_fx_energy_density()
        return self.ee_energy_density


class RKSAnalyzer(RHFAnalyzer):
    """
    Same as RHFAnalyzer, but input must be an RKS object.
    """

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
    """
    Same as UHFAnalyzer, but input must be an RKS object.
    """

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
