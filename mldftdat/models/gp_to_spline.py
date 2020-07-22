import yaml
import numpy as np
from interpolation.splines import UCGrid, CGrid, nodes
from interpolation.splines import filter_cubic, eval_cubic
from mldftdat.models.matrix_rbf import PartialQARBF, qarbf_args

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

    def predict_from_desc(self, X):
        res = 0
        for t in range(self.nterms):
            res += eval_cubic(self.spline_grids[t],
                              self.coeff_sets[t],
                              X[:,self.ind_sets[t]])\
                   * self.scale[t]
        return res

    def predict(self, X, rho_data):
        desc = self.get_descriptors(X, rho_data, num = self.num)
        F = self.predict_from_desc(desc)
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

def get_mapped_gp_evaluator(gpr):
    X = gpr.X
    alpha = gpr.gp.alpha_
    d0 = X[:,1]
    d1 = X[:,2:]
    y = gpr.y
    aqrbf = gpr.gp.kernel_.k1.k1
    gradk = gpr.gp.kernel_.k1.k2
    dims = [(0, 1, int(8 / gradk.length_scale))]
    #dims = [(0, 1, 60)]
    for i in range(X.shape[1] - 2):
        dims.append((np.min(X[:,i+2]),\
                     np.max(X[:,i+2]),\
                     #60)
                     int(8 / aqrbf.length_scale[i]) + 1)
        )
    grid = [np.linspace(dims[i][0], dims[i][1], dims[i][2])\
            for i in range(X.shape[1]-1)]
    k0s = []
    diff = (d0[:,np.newaxis] - grid[0][np.newaxis,:]) / gradk.length_scale
    k0s.append(np.exp(-0.5 * diff**2))
    for i in range(X.shape[1] - 2):
        diff = (d1[:,np.newaxis,i] - grid[i+1][np.newaxis,:]) / aqrbf.length_scale[i]
        k0s.append(np.exp(-0.5 * diff**2))
    funcps = [np.dot(alpha, k0s[0])]
    spline_grids = [UCGrid(dims[0])]
    ind_sets = [1]
    for i in range(X.shape[1] - 2):
        k = np.einsum('ni,nj->nij', k0s[0], k0s[i+1])
        funcps.append(np.einsum('n,nij->ij', alpha, k))
        spline_grids.append(UCGrid(dims[0], dims[i+1]))
        ind_sets.append((1,i+2))
    for i in range(X.shape[1] - 3):
        for j in range(i+1, X.shape[1] - 2):
            k = np.einsum('ni,nj,nk->nijk', k0s[0], k0s[i+1], k0s[j+1])
            funcps.append(np.einsum('n,nijk->ijk', alpha, k))
            spline_grids.append(UCGrid(dims[0], dims[i+1], dims[j+1]))
            ind_sets.append((1,i+2,j+2))
    print(spline_grids)
    coeff_sets = []
    for i in range(len(funcps)):
        coeff_sets.append(filter_cubic(spline_grids[i], funcps[i]))
    ndim, length_scale, scale = qarbf_args(gpr.gp.kernel_.k1.k1)
    evaluator = Evaluator(scale, ind_sets, spline_grids, coeff_sets,
                          gpr.y_to_xed, gpr.get_descriptors, gpr.num)
    ytest = gpr.gp.predict(X)
    ypred = evaluator.predict_from_desc(X)
    print(np.linalg.norm(ytest - ypred))
    print(ytest.shape)
    return evaluator
