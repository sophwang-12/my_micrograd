import random
from meat.oven import Value

class Neuron:
  def __init__(self, nin, unlinearizer):
    self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
    self.b = Value(random.uniform(-1,1))

  def __call__(self, x):
    a = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
    out = a.tanh()
    print(self.w)
    print(self.b)
    return out

  def parameters(self):
    return self.w + [self.b]

class Layer:
  def __init__(self, ninputs, nneurm, unlinearizer):
    self.neurons = [Neuron(ninputs, unlinearizer=1) for _ in range(nneur)]

  def __call__(self, x):
    outs = [n(x) for n in self.neurons]
    return outs[0] if len(outs) == 1 else outs

  def parameters(self):
    return [p for n in self.neurons for p in n.parameters()]

class MLP:
  def __init__(self, ninputs, nneur_per_layer, unlinearizer=1):
    sz = [ninputs] + nneur_per_layer
    self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nneur_per_layer))]

  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x

  def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]
