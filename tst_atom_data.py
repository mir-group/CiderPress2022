from mldftdat.analyzers import CCSDAnalyzer, RHFAnalyzer
from mldftdat.data import *
from matplotlib.scale import FuncScale
from mpl_toolkits import mplot3d

analyzer = CCSDAnalyzer.load('test_files/CCSD_He.hdf5')
indexes = get_unique_coord_indexes_spherical(analyzer.grid.coords)
coords = analyzer.grid.coords
values = analyzer.ee_energy_density - analyzer.ha_energy_density
value_name = 'e-e corr density'
rmax = 4
units = 'Ha/(Bohr$^3$)'
plot_data_atom(analyzer.mol, coords, values, value_name, rmax, units)
values = analyzer.get_corr_energy_density()
value_name = 'corr en density'
plot_data_atom(analyzer.mol, coords, values, value_name, rmax, units)
coords = coords[indexes]
values = values[indexes]
value_name = 'coor en density 2'
plot_data_atom(analyzer.mol, coords, values, value_name, rmax, units)

plt.show()

def forward(x):
    return np.log(1 + 0.2 * x)

def inverse(x):
    return (np.exp(x) - 1) * 5

print(inverse(forward(2.5)))
print(forward(inverse(2.5)))

fig, ax = plt.subplots()

analyzer = RHFAnalyzer.load('test_files/RHF_HF.hdf5')
inverse(forward(-1 * analyzer.fx_energy_density))
inverse(forward(analyzer.ha_energy_density))

values = analyzer.ha_energy_density
value_name = 'Ha en density'
plot_data_diatomic(analyzer.mol, analyzer.grid.coords, values,
                    value_name, units, (-1.5, 4))
values = -1 * analyzer.fx_energy_density
value_name = 'X en density'
plot_data_diatomic(analyzer.mol, analyzer.grid.coords, values,
                    value_name, units, (-1.5, 4))
ax.set_ylim(0, np.max(analyzer.ha_energy_density * 1.1))
ax.set_yscale('function', functions=(forward, inverse))
plt.show()

zs, rs = get_zr_diatomic(analyzer.mol, analyzer.grid.coords)
plot_surface_diatomic(analyzer.mol, zs, rs, values, value_name, units,
                        bounds = (-1.5, 4.0, 3.0), scales=(forward, inverse))

plt.show()
