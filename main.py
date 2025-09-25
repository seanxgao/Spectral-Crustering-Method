from matrix_master import *
from tests import *
from SCOPE import *

L = generate_layers_groups_graph(1,250,20,0.8,0.3,0.05)
print(measure_time_and_memory(treebuilder, L, thre=10, parallel=False, manager=None))