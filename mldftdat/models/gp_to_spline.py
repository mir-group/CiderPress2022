import yaml

class Evaluator():

    def __init__(self, scale, ind_sets, spline_grids, coeff_sets):
        self.scale = scale
        self.nterms = len(self.scale)
        self.ind_sets = ind_sets
        self.spline_grids = spline_grids
        self.coeff_sets = coeff_sets

    def __call__(self, X):
        res = 0
        for t in range(self.nterms):
            res += eval_cubic(self.spline_grids[t],
                              self.coeff_sets[t],
                              X[:,ind_sets[t]])\
                   * self.scale[t]
        return res

    def dump(self, fname):
        state_dict = {
                        'scale': np.array(scale),
                        'ind_sets': ind_sets,
                        'spline_grids': spline_grids,
                        'coeff_sets': coeff_sets
                      }
        with open(fname, 'w') as f:
            yaml.dump(state_dict, f)

    @classmethod
    def load(cls, fname):
        with open(fname 'r') as f:
            state_dict = yaml.load(f)
        return cls(state_dict['scale'],
                   state_dict['ind_sets'],
                   state_dict['spline_grids'],
                   state_dict['coeff_sets'])

def get_mapped_gp_evaluator(gpr):
    X = gpr.X
    alpha = gpr.gp.alpha_
    d0 = X[:,1]
    d1 = X[:,2:]
    y = gpr.y
    aqrbf = gpr.gp.kernel_.k1.k1
    gradk = gpr.gp.kernel_.k1.k2
    funcs = aqrbf.get_funcs_for_spline_conversion()
    dims = [(0, 1, int(4 / gradk.length_scale))]
    for i in range(X.shape[1] - 2):
        dims.append((np.min(X[:,i+2]),\
                     np.max(X[:,i+2]),\
                     int(4 / aqrbf.length_scale[i]) + 1)
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
        spline_grids.append(UCGrid(dims[0], dims[1]))
        ind_sets.append((1,i+2))
    for i in range(X.shape[1] - 3):
        for j in range(i+1, X.shape[1] - 2):
            k = np.einsum('ni,nj,nk->nijk', k0s[0], k0s[i+1], k0s[j+1])
            funcps.append(np.einsum('n,nijk->ijk', alpha, k))
            ind_sets.append((1,i+2,j+2))
    coeff_sets = []
    for i in range(len(funcps)):
        coeff_sets.append(filter_cubic(spline_grids[i], funcps[i]))
    evaluator = Evaluator(aqrbf.scale, ind_sets, spline_grids, coeff_sets)