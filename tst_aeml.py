import os
from mldftdat.data import calculate_atomization_energy
from mldftdat.data import density_similarity, rho_data_from_calc
from pyscf.dft.gen_grid import Grids
from joblib import load

mol, ae, en, atoms, calc_pbe, acalcs_pbe = calculate_atomization_energy(os.environ['MLDFTDB'],
        'RKS', 'aug-cc-pvtz', 'qm9/3-H2O', FUNCTIONAL='SCAN')
print('RKS:', ae, en, atoms)

"""
mol, ae, en, atoms = calculate_atomization_energy(os.environ['MLDFTDB'],
        'RKS', 'aug-cc-pvtz', 'qm9/3-H2O', FUNCTIONAL='wB97M-V')
print('wB97M-V:', ae, en, atoms)

mol, ae, en, atoms = calculate_atomization_energy(os.environ['MLDFTDB'],
        'RKS', 'aug-cc-pvtz', 'qm9/3-H2O', FUNCTIONAL='B97M-V')
print('B97M-V:', ae, en, atoms)
"""

mol, ae, en, atoms, calc_rhf, acalcs_rhf = calculate_atomization_energy(os.environ['MLDFTDB'],
        'RHF', 'aug-cc-pvtz', 'qm9/3-H2O')
print('RHF:', ae, en, atoms)

mol, ae, en, atoms, calc_rhf, acalcs_rhf = calculate_atomization_energy(os.environ['MLDFTDB'],
        'RKS', 'aug-cc-pvtz', 'qm9/3-H2O', FUNCTIONAL='B3LYP', mol = mol)
print('B3LYP:', ae, en, atoms)

mol, ae, en, atoms, calc_ccsdt, acalcs_ccsdt = calculate_atomization_energy(os.environ['MLDFTDB'],
        'CCSD', 'aug-cc-pvtz', 'qm9/3-H2O', mol = mol)
print('CCSD T-zeta:', ae, en, atoms)

mol, ae, en, atoms, calc_ccsdt, acalcs_ccsdt = calculate_atomization_energy(os.environ['MLDFTDB'],
        'CCSD_T', 'aug-cc-pvtz', 'qm9/3-H2O', mol = mol)
print('CCSD_T T-zeta:', ae, en, atoms)

grid = Grids(mol)
grid.build()
rho_pbe = rho_data_from_calc(calc_pbe, grid, is_ccsd = False)
rho_ccsd = rho_data_from_calc(calc_ccsdt, grid, is_ccsd = True)
rho_hf = rho_data_from_calc(calc_rhf, grid, is_ccsd = False)
print(density_similarity(rho_pbe, rho_ccsd, grid, mol))

mol, ae, en, atoms, _, _ = calculate_atomization_energy(os.environ['MLDFTDB'],
        'CCSD_T', 'def2-qzvppd', 'qm9/3-H2O', mol = mol)
        #'CCSD_T', 'aug-cc-pvqz', 'qm9/3-H2O', mol = mol)
print('CCSD_T Q-zeta:', ae, en, atoms)

mol, ae, en, atoms, calc_ml, acalcs_ml = calculate_atomization_energy(os.environ['MLDFTDB'],
        'RKS', 'aug-cc-pvtz', 'qm9/3-H2O', FUNCTIONAL='ARBF_B3LYP', mol=mol)
print('ML:', ae, en, atoms)

mol, ae, en, atoms, calc_pbex, acalcs_pbex = calculate_atomization_energy(os.environ['MLDFTDB'],
                'RKS', 'aug-cc-pvtz', 'qm9/3-H2O', FUNCTIONAL='GGA_X_PBE,GGA_C_PBE', mol = mol)

rho_pbex = rho_data_from_calc(calc_pbex, grid, is_ccsd = False)
rho_pbe = rho_data_from_calc(calc_pbe, grid, is_ccsd = False)
rho_ccsd = rho_data_from_calc(calc_ccsdt, grid, is_ccsd = True)
rho_ml = rho_data_from_calc(calc_ml, grid, is_ccsd = False)
print('PBEX:', ae, en, atoms)

print(density_similarity(rho_pbe, rho_ccsd, grid, mol))
print(density_similarity(rho_hf, rho_ccsd, grid, mol))
print(density_similarity(rho_hf, rho_ml, grid, mol))
print(density_similarity(rho_ml, rho_ccsd, grid, mol))
print(density_similarity(rho_pbe, rho_ml, grid, mol))
print(density_similarity(rho_pbe, rho_pbex, grid, mol))

