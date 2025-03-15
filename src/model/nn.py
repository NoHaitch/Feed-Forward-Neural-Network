# Reference: https://github.com/karpathy/micrograd
from .value import Value
import random


class Module:
    """ Base class for all neural network modules. """

    def zero_grad(self):
        """ Set all gradients to zero. Used before backpropagation. """
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        """ Return all the parameters of the Neural Network. """
        return []

class Neuron(Module):
    """ Reperesent a Neuron in a neural network.

    Attributes:
        w (list): Weights of the neuron.
        b (Value): Bias of the neuron.
    """

    def __init__(self, nin, nonlin=True):
        """
        Args:
            nin (int): Number of inputs to the neuron.
            nonlin (bool): Whether the neuron is linear or not.
        """
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        """ Forward pass of the neuron. """
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"


class Layer(Module):
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    """ Multi-layer perceptron.

    Attributes:
        layers (list): List of layers in the network.
    """

    def __init__(self, nin, nouts):
        """
        Args:
            nin (int): Number of inputs to the network.
            nouts (list): Number of outputs from each layer.
        """
        sz = [nin] + nouts
        self.layers = [
            Layer(sz[i], sz[i + 1], nonlin=i != len(nouts) - 1)
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        """Forward pass of the network."""
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
