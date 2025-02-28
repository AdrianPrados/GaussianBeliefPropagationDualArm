import numpy as np
from factor_graphs.gaussian import Gaussian
from factor_graphs.factor_graph import VNode, FNode, FactorGraph


g = FactorGraph()

#* 2D example
v0 = VNode('v0', ['v0.0', 'v0.1'])
v1 = VNode('v1', ['v1.0', 'v1.1'])
f0 = FNode('f0', [v0], factor=Gaussian(['v0.0', 'v0.1'], [[1], [1]], np.diag([1, 1])))
f01 = FNode('f01', [v0, v1], factor=Gaussian(['v0.0', 'v0.1', 'v1.0', 'v1.1'], [[2], [1], [3], [3]], np.diag([1, 1, 1, 1])))

g.connect(f0, v0)
g.connect(v0, f01)
g.connect(f01, v1)

beliefs = g.loopy_propagate()
print("Example in 2D")
for v, p in beliefs.items():
    print(v, p.mean)
print("------------------")

#* 3D example

g = FactorGraph()

# Definir nodos con tres variables
v0 = VNode('v0', ['v0.0', 'v0.1', 'v0.2'])
v1 = VNode('v1', ['v1.0', 'v1.1', 'v1.2'])

# Definir factores con distribuciones gaussianas en 3D
f0 = FNode('f0', [v0], factor=Gaussian(['v0.0', 'v0.1', 'v0.2'], [[1], [1], [1]], np.diag([1, 1, 1])))
f01 = FNode('f01', [v0, v1], factor=Gaussian(['v0.0', 'v0.1', 'v0.2', 'v1.0', 'v1.1', 'v1.2'], [[2], [1], [3], [3], [2], [1]], np.diag([1, 1, 1, 1, 1, 1])))

# Conectar nodos y factores en el grafo
g.connect(f0, v0)
g.connect(v0, f01)
g.connect(f01, v1)

# Ejecutar la propagación de creencias
beliefs = g.loopy_propagate()
print("Example in 3D")
for v, p in beliefs.items():
    print(v, p.mean)
print("------------------")