import yaml
import numpy as np
from interpolation.splines import UCGrid, CGrid, nodes
from interpolation.splines import filter_cubic, eval_cubic
from interpolation.splines.eval_cubic_numba import vec_eval_cubic_splines_G_1,\
                                                   vec_eval_cubic_splines_G_2,\
                                                   vec_eval_cubic_splines_G_3,\
                                                   vec_eval_cubic_splines_G_4


def get_vec_eval(grid, coeffs, X, N):
    coeffs = np.expand_dims(coeffs, coeffs.ndim)
    y = np.zeros((X.shape[0], 1))
    dy = np.zeros((X.shape[0], N, 1))
    a_, b_, orders = zip(*grid)
    if N == 1:
        vec_eval_cubic_splines_G_1(a_, b_, orders,
                                   coeffs, X, y, dy)
    elif N == 2:
        vec_eval_cubic_splines_G_2(a_, b_, orders,
                                   coeffs, X, y, dy)
    elif N == 3:
        vec_eval_cubic_splines_G_3(a_, b_, orders,
                                   coeffs, X, y, dy)
    elif N == 4:
        vec_eval_cubic_splines_G_4(a_, b_, orders,
                                   coeffs, X, y, dy)
    else:
        raise ValueError('invalid dimension N')
    return np.squeeze(y, -1), np.squeeze(dy, -1)


class Evaluator():

    def __init__(self, scale, ind_sets, spline_grids, coeff_sets,
                 xed_y_converter, feature_list, desc_order, const=0):
        self.scale = scale
        self.nterms = len(self.scale)
        self.ind_sets = ind_sets
        self.spline_grids = spline_grids
        self.coeff_sets = coeff_sets
        self.xed_y_converter = xed_y_converter
        self.feature_list = feature_list
        self.nfeat = feature_list.nfeat
        self.desc_order = desc_order
        self.const = const

    def _xedy(self, y, x, code):
        if self.xed_y_converter[-1] == 1:
            return self.xed_y_converter[code](y, x[:,0])
        elif self.xed_y_converter[-1] == 2:
            return self.xed_y_converter[code](y, x[:,0], x[:,1])
        else:
            return self.xed_y_converter[code](y)

    def xed_to_y(self, y, x):
        return self._xedy(y, x, 0)

    def y_to_xed(self, y, x):
        return self._xedy(y, x, 1)

    def get_descriptors(self, x):
        return self.feature_list(x[:,self.desc_order])

    def predict_from_desc(self, X, vec_eval=False,
                          max_order=3, min_order=0):
        res = np.zeros(X.shape[0]) + self.const
        if vec_eval:
            dres = np.zeros(X.shape)
        for t in range(self.nterms):
            if type(self.ind_sets[t]) != int and len(self.ind_sets[t]) < min_order:
                continue
            if (type(self.ind_sets[t]) == int) or len(self.ind_sets[t]) <= max_order:
                ind_set = self.ind_sets[t]
                if vec_eval:
                    y, dy = get_vec_eval(self.spline_grids[t],
                                         self.coeff_sets[t],
                                         X[:,ind_set],
                                         len(ind_set))
                    res += y * self.scale[t]
                    dres[:,ind_set] += dy * self.scale[t]
                else:
                    res += eval_cubic(self.spline_grids[t],
                                      self.coeff_sets[t],
                                      X[:,ind_set])\
                           * self.scale[t]
        if vec_eval:
            return res, dres
        else:
            return res

    def predict(self, X, vec_eval=False):
        desc = self.get_descriptors(X)
        F = self.predict_from_desc(desc, vec_eval=vec_eval)
        if vec_eval:
            F = F[0]
        return self.y_to_xed(F, X)

    def dump(self, fname):
        state_dict = {
                        'scale': np.array(self.scale),
                        'ind_sets': self.ind_sets,
                        'spline_grids': self.spline_grids,
                        'coeff_sets': self.coeff_sets,
                        'xed_y_converter': self.xed_y_converter,
                        'feature_list': self.feature_list,
                        'desc_order': self.desc_order,
                        'const': self.const
                     }
        with open(fname, 'w') as f:
            yaml.dump(state_dict, f)

    @classmethod
    def load(cls, fname):
        with open(fname, 'r') as f:
            state_dict = yaml.load(f, Loader=yaml.Loader)
        return cls(state_dict['scale'],
                   state_dict['ind_sets'],
                   state_dict['spline_grids'],
                   state_dict['coeff_sets'],
                   state_dict['xed_y_converter'],
                   state_dict['feature_list'],
                   state_dict['desc_order'],
                   const=state_dict['const'])
