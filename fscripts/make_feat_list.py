from mldftdat.models.transform_data import *

sprefac = 2 * (6 * np.pi * np.pi)**(1.0/3)
gammax = 0.004 * sprefac**2
gamma0a = 0.5
gamma1 = 0.025
gamma2 = 0.015
center0a = get_vmap_heg_value(2.0, gamma0a)

lst = [
        UMap(0, 1, gammax),
        ZMap(1, 2, 1, scale=2.0, center=1.0),
        VMap(2, 3, gamma0a, scale=1.0, center=center0a),
        UMap(3, 4, gamma1),
]
"""
lst = [
        UMap(0, 1, gammax),
        ZMap(1, 2, 1, scale=2.0, center=1.0),
        VMap(2, 3, gamma0a, scale=1.0, center=center0a),
        UMap(3, 6, gamma2),
]

lst = [
        UMap(0, 1, gammax),
        ZMap(1, 2, 1, scale=2.0, center=1.0),
        VMap(2, 11, gamma0a, scale=1.0, center=center0a),
        WMap(3, 4, 6, 9, gamma1, gamma2)
]

lst = [
        UMap(0, 1, gammax),
        ZMap(1, 2, 1, scale=2.0, center=1.0),
        VMap(2, 11, gamma0a, scale=1.0, center=center0a),
        XMap(3, 1, 4, 5, gammax, gamma1)
]
"""
flst = FeatureList(lst)
flst.dump('test_files/feat_example2.yaml')