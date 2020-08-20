import numpy as np
from mldftdat.pyscf_utils import *
from mldftdat.data import density_similarity
from pyscf.dft.numint import eval_rho2, eval_ao, eval_rho
from pyscf.scf.hf import get_jk
from pyscf.lib import logger
import time
from mldftdat.external.pyscf_scf_vxcopt import kernel
from scipy.linalg import eigvalsh, eigh
from pyscf import lib

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
                 hermi=1, vhfopt=None, realspace=False):

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

        if realspace:
            return vrho

        return vmat + vmat.T + get_jk(mol, dm)[0]

    def scf(self, **kwargs):

        self.initialize_for_scf()
        rho_data, dm0, mf = self.initial_dft_guess(self.mu_max, return_mf=True)

        cput0 = (time.clock(), time.time())

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
        #mf.DIIS = scf.ADIIS
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
        self.rdm1 = mf.make_rdm1()
        self.rdm1 = lib.tag_array(self.rdm1, mo_coeff=self.mo_coeff,
                                  mo_occ=self.mo_occ,
                                  mo_energy=self.mo_energy)
        self.mf = mf
        self.gs_density = rho_data
        self.rho_data = rho_data
        self.vrho = self.get_veff(mf.mol, self.rdm1, realspace=True)
        return self.ds, ds_init


class FPXCPotentialAnalyzer(XCPotentialAnalyzer):

    def compute_lambda_ij(self):
        """
        Compute the wavefunction part of the XC potential formula.
        Returns the wf_term as well as mu_max (most negative
        eigenvalue of G), h1e_ao, and the Hartree potential.
        """
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
        dno = np.zeros(ao_data[1:4].shape)
        dno[0] = np.dot(np.dot(ao_data[1], self.analyzer.mo_coeff), no_coeff)
        dno[1] = np.dot(np.dot(ao_data[2], self.analyzer.mo_coeff), no_coeff)
        dno[2] = np.dot(np.dot(ao_data[3], self.analyzer.mo_coeff), no_coeff)
        tsum = 0
        for k in range(nno):
            for l in range(k+1, nno):
                if no_occ[k] > 0 and no_occ[l] > 0:
                    diff = no[:,k] * dno[:,:,l] - no[:,l] * dno[:,:,k]
                    diff = np.einsum('rn,rn->n', diff, diff)
                    tsum += no_occ[k] * no_occ[l] * diff
        tau_rho = 0.5 * tsum / (rho_data[0] + 1e-20)**2
        vs_xc *= 2 / (rho_data[0] + 1e-20)
        eps_wf = np.dot(no, lambda_ij)
        eps_wf = np.einsum('ni,ni->n', no, eps_wf)
        eps_wf /= (rho_data[0] + 1e-20)
        fac = np.sqrt(np.outer(no_occ, no_occ) + 1e-10)
        mu_max = np.max(np.linalg.eigvalsh(lambda_ij / fac))
        mu_max = np.max(eigvalsh(lambda_ij, np.diag(no_occ + 1e-10)))
        print('MU_MAX', mu_max)
        return vs_xc - eps_wf + tau_rho, mu_max, h1e_ao,\
            2 * self.analyzer.get_ha_energy_density() / (rho_data[0] + 1e-20)

    def get_veff(self, mol, dm, dm_last=None, vhf_last=None,
                 hermi=1, vhfopt=None, realspace=False):

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
        nmo = mo_occ.shape[0]
        mo = np.dot(ao, mo_coeff)
        dmo = np.zeros(ao_data[1:4].shape)
        dmo[0] = np.dot(ao_data[1], mo_coeff)
        dmo[1] = np.dot(ao_data[2], mo_coeff)
        dmo[2] = np.dot(ao_data[3], mo_coeff)
        tsum = 0
        for k in range(nmo):
            for l in range(k+1, nmo):
                if mo_occ[k] > 0 and mo_occ[l] > 0:
                    diff = mo[:,k] * dmo[:,:,l] - mo[:,l] * dmo[:,:,k]
                    diff = np.einsum('rn,rn->n', diff, diff)
                    tsum += mo_occ[k] * mo_occ[l] * diff
        tau_rho = 0.5 * tsum / (rho_data[0] + 1e-20)**2
        mo_data = np.dot(ao, mo_coeff)**2
        eps_ks = np.dot(mo_data, mo_occ * mo_energy)
        eps_ks /= rho_data[0] + 1e-20
        ks_term = eps_ks - tau_rho
        vrho = ks_term + wf_term

        if realspace:
            return vrho

        aow = np.einsum('pi,p->pi', ao, .5*weight*vrho)
        vmat = np.dot(ao.T, aow)

        return vmat + vmat.T + get_jk(mol, dm)[0]
        
