from mldftdat.models.transform_data import *

sprefac = 2 * (6 * np.pi * np.pi)**(1.0/3)
gammax = 0.004 * sprefac**2
gamma0a = 0.5
gamma1 = 0.025
center0a = get_vmap_heg_value(2.0, gamma0a)
lst = [
        UMap(0, 1, gammax),
        ZMap(1, 2, 1, scale=2.0, center=1.0),
        VMap(2, 3, gamma0a, scale=1.0, center=center0a),
        UMap(3, 4, gamma1),
]
flst = FeatureList(lst)
flst.dump('test_files/feat_example.yaml')