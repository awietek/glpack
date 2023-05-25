from glpack.density import Density
from glpack.phases import Phases, init_phases
from glpack.io import read_density, read_phases
from glpack.ginzburg_landau import solve_ginzburg_landau, ginzburg_landau_functional, solve_ginzburg_landau_ext, ginzburg_landau_functional_ext
from glpack.metric import metric_plain, min_angle, metric


from matplotlib import rc
rc('font',**{'family':'Times New Roman', 'size': 13})
rc('text', usetex=True)
params= {'text.latex.preamble' : r'\usepackage{amsmath}'}
plt.rcParams.update(params)
