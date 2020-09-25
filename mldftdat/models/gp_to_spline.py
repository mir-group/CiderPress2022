import yaml
import numpy as np
from interpolation.splines import UCGrid, CGrid, nodes
from interpolation.splines import filter_cubic, eval_cubic
from mldftdat.models.matrix_rbf import PartialQARBF, qarbf_args
from interpolation.splines.eval_cubic_numba import vec_eval_cubic_splines_G_1,\
                                                   vec_eval_cubic_splines_G_2,\
                                                   vec_eval_cubic_splines_G_3
from sklearn.gaussian_process.kernels import RBF


def get_vec_eval(grid, coeffs, X, N):
    coeffs = np.expand_dims(coeffs, coeffs.ndim)
    y = np.zeros((X.shape[0], 1))
    dy = np.zeros((X.shape[0], N, 1))
    a_, b_, orders = zip(*grid)
    if N == 1:
        vec_eval_cubic_splines_G_1(a_, b_, orders,
                                   coeffs, X, y, dy)
    if N == 2:
        vec_eval_cubic_splines_G_2(a_, b_, orders,
                                   coeffs, X, y, dy)
    if N == 3:
        vec_eval_cubic_splines_G_3(a_, b_, orders,
                                   coeffs, X, y, dy)
    return np.squeeze(y, -1), np.squeeze(dy, -1)

class Evaluator():

    def __init__(self, scale, ind_sets, spline_grids, coeff_sets,
                 y_to_xed, get_descriptors, num):
        self.scale = scale
        self.nterms = len(self.scale)
        self.ind_sets = ind_sets
        self.spline_grids = spline_grids
        self.coeff_sets = coeff_sets
        self.y_to_xed = y_to_xed
        self.get_descriptors = get_descriptors
        self.num = num

    def predict_from_desc(self, X, max_order = 3, vec_eval = False, subind = 0):
        res = np.zeros(X.shape[0])
        if vec_eval:
            dres = np.zeros(X.shape)
        #print(np.max(X, axis=0))
        #print(np.min(X, axis=0))
        for t in range(self.nterms):
            if (type(self.ind_sets[t]) == int) or len(self.ind_sets[t]) <= max_order:
                #print(self.scale[t], self.ind_sets[t])
                if subind > 0:
                    ind_set = [ind-subind for ind in self.ind_sets[t]]
                else:
                    ind_set = self.ind_sets[t]
                if vec_eval:
                    y, dy = get_vec_eval(self.spline_grids[t],
                                          self.coeff_sets[t],
                                          X[:,ind_set],
                                          len(ind_set))
                    res += y * self.scale[t]
                    dres[:,ind_set] += dy * self.scale[t]
                else:
                    #y = eval_cubic(self.spline_grids[t],
                    #                  self.coeff_sets[t],
                    #                  X[:,ind_set])
                    #print('RANGE', t, np.min(y), np.max(y)) 
                    res += eval_cubic(self.spline_grids[t],
                                      self.coeff_sets[t],
                                      X[:,ind_set])\
                           * self.scale[t]
        if vec_eval:
            return res, dres
        else:
            return res

    def predict(self, X, rho_data, vec_eval = False):
        desc = self.get_descriptors(X, rho_data, num = self.num)
        F = self.predict_from_desc(desc, vec_eval = vec_eval)
        if vec_eval:
            F = F[0]
        return self.y_to_xed(F, rho_data)

    def dump(self, fname):
        state_dict = {
                        'scale': np.array(self.scale),
                        'ind_sets': self.ind_sets,
                        'spline_grids': self.spline_grids,
                        'coeff_sets': self.coeff_sets,
                        'y_to_xed': self.y_to_xed,
                        'get_descriptors': self.get_descriptors,
                        'num': self.num
                     }
        with open(fname, 'w') as f:
            yaml.dump(state_dict, f)

    @classmethod
    def load(cls, fname):
        with open(fname, 'r') as f:
            state_dict = yaml.load(f)
        return cls(state_dict['scale'],
                   state_dict['ind_sets'],
                   state_dict['spline_grids'],
                   state_dict['coeff_sets'],
                   state_dict['y_to_xed'],
                   state_dict['get_descriptors'],
                   state_dict['num'])


class EvaluatorCorr(Evaluator):

    def predict(self, X, rho_data, vec_eval = False):
        rho_data_u, rho_data_d = spinpol_data(rho_data)
        xdescu, xdescd = spinpol_data(X)
        desc = self.get_descriptors(xdescu, xdescd, rho_data_u, rho_data_d, num=self.num)
        F = self.predict_from_desc(desc[:,2:], vec_eval = vec_eval)
        if vec_eval:
            F = F[0]
        #F *= desc[:,1]
        return self.y_to_xed(F, rho_data_u, rho_data_d)


def spinpol_data(data_arr):
    if data_arr.ndim == 2:
        return data_arr, data_arr
    else:
        return data_arr[0], data_arr[1]


class Evaluator2(Evaluator):

    def __init__(self, scale, ind_sets, spline_grids, coeff_sets,
                 y_to_xed, get_descriptors, num, const = 0.0):
        self.scale = scale
        self.nterms = len(self.scale)
        self.ind_sets = ind_sets
        self.spline_grids = spline_grids
        self.coeff_sets = coeff_sets
        self.y_to_xed = y_to_xed
        self.get_descriptors = get_descriptors
        self.num = num
        self.const = const

    def predict_from_desc(self, X, max_order = 3, vec_eval = False, subind = 0):
        res = super(Evaluator2, self).predict_from_desc(
                                X, max_order, vec_eval, subind)
        if vec_eval:
            return self.const + res[0], res[1]
        else:
            return self.const + res

    def predict(self, X, rho_data, vec_eval = False): 
        rho_data_u, rho_data_d = spinpol_data(rho_data)
        xdescu, xdescd = spinpol_data(X)
        desc = self.get_descriptors(xdescu, xdescd, rho_data_u, rho_data_d, num=self.num)
        F = self.predict_from_desc(desc[:,2:], vec_eval = vec_eval)
        if vec_eval:
            F = F[0]
        F *= desc[:,1]
        return self.y_to_xed(F, rho_data_u, rho_data_d)


class LinearEvaluator():

    def __init__(self, alpha, d0):
        self.const = np.dot(alpha, d0)

    def predict_from_desc(self, X, max_order=3, vec_eval=False, subind=0):
        return X[:,0] * self.const


class EvaluatorSum():

    def __init__(self, eval_ss, eval_os, NSS, NOS,
                 y_to_xed, get_descriptors, num):
        self.eval_ss = eval_ss
        self.eval_os = eval_os
        self.NOS = NOS
        self.NSS = NSS
        self.y_to_xed = y_to_xed
        self.get_descriptors = get_descriptors
        self.num = num

    def predict_from_desc(self, X, max_order = 3, vec_eval = False, subind = 0,
                          sum_terms = True):
        NOS, NSS = self.NOS, self.NSS
        os_term = self.eval_os.predict_from_desc(X[:,2:NOS+2],
                                            max_order, vec_eval, subind)
        u_term = self.eval_ss.predict_from_desc(X[:,NOS+3:NOS+NSS+3],
                                           max_order, vec_eval = vec_eval, subind = subind)
        d_term = self.eval_ss.predict_from_desc(X[:,NOS+NSS+4:NOS+2*NSS+4],
                                           max_order, vec_eval = vec_eval, subind = subind)
        if vec_eval:
            os_term, dos_term = os_term
            u_term, du_term = u_term
            d_term, dd_term = d_term
            return (os_term, u_term, d_term), (dos_term, du_term, d_term)
        elif sum_terms:
            return os_term * X[:,1] + u_term * X[:,NOS+2] + d_term * X[:,NOS+NSS+3]
        else:
            return os_term, u_term, d_term

    def predict(self, X, rho_data, vec_eval = False): 
        rho_data_u, rho_data_d = spinpol_data(rho_data)
        xdescu, xdescd = spinpol_data(X)
        desc = self.get_descriptors(xdescu, xdescd, rho_data_u, rho_data_d, num=self.num)
        F = self.predict_from_desc(desc, sum_terms = True)
        return self.y_to_xed(F, rho_data_u, rho_data_d)


def get_dim(x, length_scale, density = 6, buff = 0.0, bound = None):
    print(length_scale, bound)
    mini = np.min(x) - buff
    maxi = np.max(x) + buff
    if bound is not None:
        mini, maxi = bound[0], bound[1]
    ran = maxi - mini
    ngrid = max(int(density * ran / length_scale) + 1, 3)
    return (mini, maxi, ngrid)

def get_mapped_gp_evaluator(gpr, test_x = None, test_y = None, test_rho_data = None,
                            version = 'a'):
    """
    version a is for RBF(s) * ARBF(other desc) with X[:,0]=rho (used for exchange)
    version b is for Linear(SCAN) * ARBF(other desc) with X[:,0]=rho
    (used for correlation)
    """
    X = gpr.X
    alpha = gpr.gp.alpha_
    if version == 'a':
        d0 = X[:,1]
        d1 = X[:,2:]
    elif version == 'b':
        d0 = X[:,1]
        d1 = X[:,2:]
    else:
        raise ValueError('version must be a or b')
    print("SHAPES", d0.shape, d1.shape)
    y = gpr.y
    print(gpr.gp.kernel_)
    aqrbf = gpr.gp.kernel_.k1.k1
    gradk = gpr.gp.kernel_.k1.k2
    if version == 'a':
        dims = [get_dim(d0, gradk.length_scale, density = 4, bound = (0,1))]
    else:
        dims = [get_dim(d0, gradk.length_scale, density = 4, bound = (0,np.max(d0)*2))]
    if isinstance(aqrbf, RBF):
        ndim, length_scale, scale = qarbf_args(gpr.gp.kernel_.k1.k1)
        scale = np.array(scale)
    else:
        scale = aqrbf.scale
    #dims = [(0, 1, 60)]
    bounds = [(0, 1),\
              (-1,1),\
              #(-2*0.5,1),\
              #(-2*0.1002,1),\
              (-2*0.2167,1),\
              (0,1),\
              (0,1),\
              (-1,1),\
              (-1,1),\
              #(-8*0.125,1),\
              #(-8*0.8350,1),\
              (-8*1.0937,1),\
              #(-0.5*2.0,1),\
              #(-0.5*0.2483,1),\
              (-0.5*0.2365,1),\
              (-1,1),
              (-1,1)][:d1.shape[1]+1]
    if version != 'a':
        bounds[0] = (0, np.max(d0) * 2)
    for i in range(X.shape[1] - 2):
        dims.append( get_dim(d1[:,i], aqrbf.length_scale[i], density = 4, bound=bounds[i+1]) )
    grid = [np.linspace(dims[i][0], dims[i][1], dims[i][2])\
            for i in range(X.shape[1]-1)]
    k0s = []
    print(gradk.length_scale)
    diff = (d0[:,np.newaxis] - grid[0][np.newaxis,:]) / gradk.length_scale
    k0s.append(np.exp(-0.5 * diff**2))
    for i in range(X.shape[1] - 2):
        print(aqrbf.length_scale[i])
        diff = (d1[:,i:i+1] - grid[i+1][np.newaxis,:]) / aqrbf.length_scale[i]
        k0s.append(np.exp(-0.5 * diff**2))
    funcps = [np.dot(alpha, k0s[0])]
    spline_grids = [UCGrid(dims[0])]
    ind_sets = [(1,)]
    for i in range(X.shape[1] - 2):
        print(i, dims[0], dims[i+1])
        k = np.einsum('ni,nj->nij', k0s[0], k0s[i+1])
        funcps.append(np.einsum('n,nij->ij', alpha, k))
        print('range', i, np.min(funcps[-1]), np.max(funcps[-1]))
        spline_grids.append(UCGrid(dims[0], dims[i+1]))
        if i == 0:
            print(nodes(spline_grids[-1]))
            print(grid[0], grid[1])
        ind_sets.append((1,i+2))
    for i in range(X.shape[1] - 3):
        for j in range(i+1, X.shape[1] - 2):
            print(i, j, dims[i+1], dims[j+1])
            k = np.einsum('ni,nj,nk->nijk', k0s[0], k0s[i+1], k0s[j+1])
            funcps.append(np.einsum('n,nijk->ijk', alpha, k))
            print('range', i, j, np.min(funcps[-1]), np.max(funcps[-1]))
            spline_grids.append(UCGrid(dims[0], dims[i+1], dims[j+1]))
            ind_sets.append((1,i+2,j+2))
    print(spline_grids)
    coeff_sets = []
    for i in range(len(funcps)):
        coeff_sets.append(filter_cubic(spline_grids[i], funcps[i]))
    evaluator = Evaluator(scale, ind_sets, spline_grids, coeff_sets,
                          gpr.y_to_xed, gpr.get_descriptors, gpr.num)
    if not isinstance(aqrbf, RBF):
        return evaluator

    res, en = aqrbf(X, get_sub_kernels = True)
    resg = gradk(X)
    res = np.dot(alpha, aqrbf.scale[0] * resg)
    print("Comparing K and Kspline!!!")
    print(en[0])
    tsp = eval_cubic(spline_grids[0],
                     coeff_sets[0],
                     X[:,1:2])
    diff = (d0[:,np.newaxis] - d0[np.newaxis,:]) / gradk.length_scale
    tk = np.exp(-0.5 * diff**2)
    print(np.mean(np.abs(gradk(X) - tk)))
    print(np.mean(np.abs(res - evaluator.predict_from_desc(X, max_order = 1))))
    print(np.mean(np.abs(res - aqrbf.scale[0] * tsp)))
    print(evaluator.scale[0], scale[0], aqrbf.scale[0])
    print(res[::1000])
    print(tsp[::1000])
    print("checked 1d")
    print(np.mean(res - evaluator.predict_from_desc(X, max_order = 1)))
    res += np.dot(alpha, aqrbf.scale[1] * en[1] * resg)
    print(np.mean(np.abs(res - evaluator.predict_from_desc(X, max_order = 2))))
    res += np.dot(alpha, aqrbf.scale[2] * en[2] * resg)
    print(np.mean(np.abs(res - evaluator.predict_from_desc(X, max_order = 3))))

    ytest = gpr.gp.predict(X)
    ypred = evaluator.predict_from_desc(X)
    test_y = gpr.xed_to_y(test_y, test_rho_data)
    print(np.mean(np.abs(ytest - ypred)))
    print(np.mean(np.abs(ytest - y)))
    print(np.mean(np.abs(y - ypred)))
    print(np.linalg.norm(ytest - y))
    print(np.linalg.norm(ypred - y))
    print(ytest.shape)

    if test_x is not None:
        ytest = gpr.xed_to_y(gpr.predict(test_x, test_rho_data), test_rho_data)
        ypred = gpr.xed_to_y(evaluator.predict(test_x, test_rho_data), test_rho_data)
        print(np.max(np.abs(ytest - ypred)))
        print(np.max(np.abs(ytest - test_y)))
        print(np.max(np.abs(test_y - ypred)))
        print()
        print(np.linalg.norm(ytest - test_y))
        print(np.mean(np.abs(ytest - test_y)))
        print(np.mean(ytest - test_y))
        print(np.linalg.norm(ypred - test_y))
        print(np.mean(np.abs(ypred - test_y)))
        print(np.mean(ypred - test_y))
        print(np.linalg.norm(ypred - ytest))
        print(np.mean(np.abs(ypred - ytest)))
        print(np.mean(ypred - ytest))

    return evaluator


def get_mapped_gp_evaluator_corr(gpr, test_x = None, test_y = None,
                                 test_rho_data = None, triples = False):
    """
    version a is for RBF(s) * ARBF(other desc) with X[:,0]=rho (used for exchange)
    version b is for Linear(SCAN) * ARBF(other desc) with X[:,0]=rho
    (used for correlation)
    """
    X = gpr.X
    d0 = np.ones(X[:,1].shape[0])
    d1 = X[:,2:]
    NFEAT = d1.shape[1]
    alpha = gpr.gp.alpha_ * d0
    y = gpr.y
    print(gpr.gp.kernel_)
    aqrbf = gpr.gp.kernel_.k1.k2
    bounds = [(0, 1),\
              (-1,1),\
              (-2*0.64772,1),\
              (0,1),\
              (0,1),\
              (-1,1),\
              (-1,1),\
              (-8*0.44065,1),\
              (-0.5*0.6144,1),\
              (-1,1),\
              (0,1.2 * np.max(X[:,-1]))]#,\
    #(0,1)]
    print('minmax', np.max(X[:,-1]), np.min(X[:,-1]))
    dims = []
    if isinstance(aqrbf, RBF):
        ndim, length_scale, scale = qarbf_args(aqrbf)
        scale = np.array(scale)
    else:
        scale = aqrbf.scale
    for i in range(NFEAT):
        dims.append( get_dim(d1[:,i], aqrbf.length_scale[i], density = 4, bound = bounds[i]) )
    grid = [np.linspace(dims[i][0], dims[i][1], dims[i][2])\
            for i in range(NFEAT)]
    k0s = []
    for i in range(NFEAT):
        print(aqrbf.length_scale[i])
        diff = (d1[:,i:i+1] - grid[i][np.newaxis,:]) / aqrbf.length_scale[i]
        k0s.append(np.exp(-0.5 * diff**2))
    funcps = []
    spline_grids = []
    ind_sets = []
    for i in range(NFEAT):
        funcps.append(np.einsum('n,ni->i', alpha, k0s[i]))
        spline_grids.append(UCGrid(dims[i]))
        ind_sets.append((i,))
    for i in range(NFEAT - 1):
        for j in range(i+1, NFEAT):
            k = np.einsum('ni,nj->nij', k0s[i], k0s[j])
            funcps.append(np.einsum('n,nij->ij', alpha, k))
            spline_grids.append(UCGrid(dims[i], dims[j]))
            ind_sets.append((i,j))
    if triples:
        for i in range(NFEAT - 2):
            for j in range(i+1, NFEAT - 1):
                for l in range(j+1, NFEAT):
                    k = np.einsum('ni,nj,nk->nijk', k0s[i], k0s[j], k0s[l])
                    funcps.append(np.einsum('n,nijk->ijk', alpha, k))
                    spline_grids.append(UCGrid(dims[i], dims[j], dims[l]))
                    ind_sets.append((i,j,l))
    print(spline_grids)
    coeff_sets = []
    for i in range(len(funcps)):
        coeff_sets.append(filter_cubic(spline_grids[i], funcps[i]))
    evaluator = Evaluator2(scale[1:], ind_sets, spline_grids, coeff_sets,
                           gpr.y_to_xed, gpr.get_descriptors, gpr.num,
                           const = scale[0] * np.sum(alpha))
    if not isinstance(aqrbf, RBF):
        return evaluator

    res, en = aqrbf(X, get_sub_kernels = True)
    res = np.sum(alpha) * aqrbf.scale[0] * d0
    print("Comparing K and Kspline!!!")
    print(en[0])
    print(np.mean(np.abs(res - evaluator.const * d0)))
    print(evaluator.scale[0], scale[0], aqrbf.scale[0])
    print(res[::1000])
    print("checked 1d")
    print(np.mean(res - d0 * evaluator.predict_from_desc(d1, max_order = 1)))
    res += np.dot(alpha, aqrbf.scale[1] * en[1]) * d0
    print(np.mean(np.abs(res - d0 * evaluator.predict_from_desc(d1, max_order = 1))))
    res += np.dot(alpha, aqrbf.scale[2] * en[2]) * d0
    print(np.mean(np.abs(res - d0 * evaluator.predict_from_desc(d1, max_order = 2))))

    ytest = gpr.gp.predict(X)
    ypred = d0 * evaluator.predict_from_desc(d1)
    print(np.mean(np.abs(ytest - ypred)))
    print(np.mean(np.abs(ytest - y)))
    print(np.mean(np.abs(y - ypred)))
    print(np.linalg.norm(ytest - y))
    print(np.linalg.norm(ypred - y))
    print(ytest.shape)

    if test_x is not None:
        test_y = gpr.xed_to_y(test_y, test_rho_data)
        ytest = gpr.xed_to_y(gpr.predict(test_x, test_rho_data), test_rho_data)
        ypred = gpr.xed_to_y(evaluator.predict(test_x, test_rho_data), test_rho_data)
        print(np.max(np.abs(ytest - ypred)))
        print(np.max(np.abs(ytest - test_y)))
        print(np.max(np.abs(test_y - ypred)))
        print()
        print(np.linalg.norm(ytest - test_y))
        print(np.mean(np.abs(ytest - test_y)))
        print(np.mean(ytest - test_y))
        print(np.linalg.norm(ypred - test_y))
        print(np.mean(np.abs(ypred - test_y)))
        print(np.mean(ypred - test_y))
        print(np.linalg.norm(ypred - ytest))
        print(np.mean(np.abs(ypred - ytest)))
        print(np.mean(ypred - ytest))

    return evaluator


def get_sub_evaluator(kernel, X, alpha, bounds):
    d0 = X[:,0]
    d1 = X[:,1:]
    triples = False
    NFEAT = d1.shape[1]
    linear = kernel.k1
    aqrbf = kernel.k2
    print(alpha.shape, d0.shape)
    alpha = alpha * d0
    dims = []
    if isinstance(aqrbf, RBF):
        ndim, length_scale, scale = qarbf_args(aqrbf)
        scale = np.array(scale)
    else:
        scale = aqrbf.scale
    for i in range(NFEAT):
        dims.append( get_dim(d1[:,i], aqrbf.length_scale[i], density = 6, bound = bounds[i]) )
    grid = [np.linspace(dims[i][0], dims[i][1], dims[i][2])\
            for i in range(NFEAT)]
    k0s = []
    for i in range(NFEAT):
        print(aqrbf.length_scale[i])
        diff = (d1[:,i:i+1] - grid[i][np.newaxis,:]) / aqrbf.length_scale[i]
        k0s.append(np.exp(-0.5 * diff**2))
    funcps = []
    spline_grids = []
    ind_sets = []
    for i in range(NFEAT):
        funcps.append(np.einsum('n,ni->i', alpha, k0s[i]))
        spline_grids.append(UCGrid(dims[i]))
        ind_sets.append((i,))
    for i in range(NFEAT - 1):
        for j in range(i+1, NFEAT):
            k = np.einsum('ni,nj->nij', k0s[i], k0s[j])
            funcps.append(np.einsum('n,nij->ij', alpha, k))
            spline_grids.append(UCGrid(dims[i], dims[j]))
            ind_sets.append((i,j))
    if triples:
        for i in range(NFEAT - 2):
            for j in range(i+1, NFEAT - 1):
                for l in range(j+1, NFEAT):
                    k = np.einsum('ni,nj,nk->nijk', k0s[i], k0s[j], k0s[l])
                    funcps.append(np.einsum('n,nijk->ijk', alpha, k))
                    spline_grids.append(UCGrid(dims[i], dims[j], dims[l]))
                    ind_sets.append((i,j,l))
    print(spline_grids)
    coeff_sets = []
    for i in range(len(funcps)):
        coeff_sets.append(filter_cubic(spline_grids[i], funcps[i]))
    evaluator = Evaluator2(scale[1:], ind_sets, spline_grids, coeff_sets,
                           None, None, NFEAT,
                           const = scale[0] * np.sum(alpha))
    return evaluator

def get_sub_evaluator2(kernel, X, alpha, bounds, version = 'c'):
    if version == 'c':
        d2 = np.ones(X[:,0].shape)
        d1 = X[:,1:-1]
        d0 = X[:,-1]
        alpha = alpha
        NFEAT = d1.shape[1]
        aqrbf, srbf = kernel.k1, kernel.k2
    else:
        d2 = X[:,0]
        d1 = X[:,1:-1]
        d0 = X[:,-1]
        alpha = alpha * d2
        NFEAT = d1.shape[1]
        linear = kernel.k1
        aqrbf, srbf = kernel.k2.k1, kernel.k2.k2
    print(type(aqrbf), type(srbf), d1.shape)
    ndim, length_scale, scale = qarbf_args(aqrbf)
    scale = np.array(scale)
    dims = []
    for i in range(NFEAT):
        dims.append( get_dim(d1[:,i], aqrbf.length_scale[i], density=6, bound=bounds[i]) )
    dims.append(get_dim(d0, srbf.length_scale, density=6, bound = bounds[-1]))
    grid = [np.linspace(dims[i][0], dims[i][1], dims[i][2])\
            for i in range(NFEAT + 1)]
    
    k0s = []
    for i in range(NFEAT):
        diff = (d1[:,i:i+1] - grid[i][np.newaxis,:]) / aqrbf.length_scale[i]
        k0s.append(np.exp(-0.5 * diff**2))
    diff = (d0[:,np.newaxis] - grid[-1][np.newaxis,:]) / srbf.length_scale
    k0s.append(np.exp(-0.5 * diff**2))

    funcps = [np.dot(alpha, k0s[-1])]
    spline_grids = [UCGrid(dims[-1])]
    ind_sets = [(-1,)]
    for i in range(NFEAT):
        k = np.einsum('ni,nj->nij', k0s[-1], k0s[i])
        funcps.append(np.einsum('n,nij->ij', alpha, k))
        spline_grids.append(UCGrid(dims[-1], dims[i]))
        ind_sets.append((-1,i))
    for i in range(NFEAT - 1):
        for j in range(i+1, NFEAT):
            k = np.einsum('ni,nj,nk->nijk', k0s[-1], k0s[i], k0s[j])
            funcps.append(np.einsum('n,nijk->ijk', alpha, k))
            spline_grids.append(UCGrid(dims[-1], dims[i], dims[j]))
            ind_sets.append((-1,i,j))

    coeff_sets = []
    for i in range(len(funcps)):
        coeff_sets.append(filter_cubic(spline_grids[i], funcps[i]))
    if version == 'c':
        evaluator = EvaluatorCorr(scale, ind_sets, spline_grids, coeff_sets,
                                  None, None, NFEAT)
    else:
        evaluator = EvaluatorCorr(scale, ind_sets, spline_grids, coeff_sets,
                                  None, None, NFEAT)

    return evaluator


def get_mapped_gp_evaluator_corr_sp(gpr, triples = False, version = 'a'):
    X = gpr.X
    NOS = 11
    NSS = 4
    print('MAX', np.max(X[:,NOS+1]))
    bounds_os = [(0, 1),\
                 (-1,1),\
                 (-2*0.64772,1),\
                 (0,1),\
                 (0,1),\
                 (-1,1),\
                 (-1,1),\
                 (-8*0.44065,1),\
                 (-0.5*0.6144,1),\
                 (-1,1),\
                 (-4,1.2 * np.max(X[:,NOS+1]))]
    bounds_ss = [(0,1),\
                 (-2*0.64772,1),\
                 (-1,1),\
                 (-4,1.2 * np.max(X[:,NOS+1]))]
    if version == 'a':
        eval_os = get_sub_evaluator2(gpr.gp.kernel_.k1.k1, X[:,1:NOS+2],
                                     gpr.gp.alpha_, bounds_os)
        eval_ss = LinearEvaluator(gpr.gp.alpha_, X[:,NOS+2])
        #eval_ss = get_sub_evaluator2(gpr.gp.kernel_.k1.k2.k, X[:,NOS+2:NOS+NSS+3],
        #                             gpr.gp.alpha_, bounds_ss)
    elif version == 'b':
        eval_os = get_sub_evaluator2(gpr.gp.kernel_.k1.k1, X[:,1:NOS+2],
                                     gpr.gp.alpha_, bounds_os)
        eval_ss = get_sub_evaluator2(gpr.gp.kernel_.k1.k2.k, X[:,NOS+2:NOS+NSS+3],
                                     gpr.gp.alpha_, bounds_ss)
    elif version == 'c':
        evaluator = get_sub_evaluator2(gpr.gp.kernel.k1, X[:,1:NOS+2],
                                       gpr.gp.alpha_, bounds_os)
        evaluator.y_to_xed = gpr.y_to_xed
        evaluator.get_descriptors = gpr.get_descriptors
        evaluator.num = gpr.num
    else:
        raise ValueError('version must be a, b, or c')

    if version != 'c':
        evaluator = EvaluatorSum(eval_ss, eval_os, NSS, NOS,
                                 gpr.y_to_xed, gpr.get_descriptors, gpr.num)

    y = gpr.y

    #res, en = aqrbf(X, get_sub_kernels = True)
    print("checked 1d")
    #print(np.mean(np.abs(res - evaluator.predict_from_desc(X, sum_terms = True))))

    ytest = gpr.gp.predict(X)
    if version == 'c':
        ypred = evaluator.predict_from_desc(X[:,2:NOS+2])
        #ypred = X[:,1] * evaluator.predict_from_desc(X[:,2:NOS+2])
    else:
        ypred = evaluator.predict_from_desc(X, sum_terms = True)
    print(np.mean(np.abs(ytest - ypred)))
    print(np.mean(np.abs(ytest - y)))
    print(np.mean(np.abs(y - ypred)))
    print(np.linalg.norm(ytest - y))
    print(np.linalg.norm(ypred - y))
    print(ytest.shape)

    return evaluator
