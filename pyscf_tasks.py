from fireworks import FiretaskBase
from fireworks.utilities.fw_utilities import explicit_serialize
import ase.io
from pyscf.pbc.tools.pyscf_ase import atoms_from_ase
from pyscf import gto, scf, dft, cc, fci


@explicit_serialize
class HFCalc(FiretaskBase):

    required_params = ['name', 'struct']
    optional_params = ['calc', 'basis']

    calc_opts = {
                'RHF': scf.RHF,
                'UHF': scf.UHF,
                'RKS': dft.RKS,
                'UKS': dft.UKS,
                'CCSD': cc.CCSD,
                'FCI': fci.FCI
            }

    def run_task(self, fw_spec):
        atoms = ase.io.read(self.struct, format='xyz')
        mol = gto.Mole()
        mol.atom = atoms_from_ase(atoms)
        mol.basis = 'ccpvdz'
        mol.build()
        if self.calc == None:
            self.calc = 'RHF'
        calc = self.calc_opts[self.calc](mol)
        calc.kernel()
        return FWAction(update_spec={
                'mol'  :  mol,
                'calc' :  calc
            })

        
@explicit_serialize
class PostHFCalc(FiretaskBase):

    optional_params = ['calc']

    calc_opts = {
                'CCSD': cc.CCSD,
                'FCI': fci.FCI
            }

    def run_task(self, fw_spec):
        mol = fw_spec['mol']
        hfcalc = fw_spec['calc']
        if self.calc == None:
            self.calc = 'CCSD'
        calc = calc_opts[self.calc](mol, calc.mo_coeff)
        calc.kernel()
        return FWAction(update_spec={
                'calc': calc
            })


@explicit_serialize
class TrainingDataCollection(FiretaskBase):

    def run_task(self, fw_spec):
        calc = fw_spec['calc']
        dm1 = calc.make_rdm1()
        dm2 = calc.make_rdm2()

