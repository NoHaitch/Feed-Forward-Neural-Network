# Reference: https://github.com/karpathy/micrograd
from .value import Value
from src.utils.converter import Converter


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
    """ Represents a Neuron in a Neural Network.

        Attributes:
            w (list[Value]): Weights of the neuron.
            b (Value): Bias of the neuron.
            active_func (callable[[Value], Value]): Activation function used.
    """

    def __init__( self, nin: int, weights: list = None, bias: int | float = 0.0, same_weight: int | float = 0,  active_func: callable[[Value], Value] = lambda x: x.relu()):
        """
        Args:
            nin (int): Number of inputs to the neuron.
            weights (list, optional): Predefined weights. If None, random values are used.
            bias (int | float, optional): Bias of the neuron.
            same_weight (int | float, optional): If True, all inputs use the same weight.
            active_func (callable[[Value], Value], optional): Activation function.
        """
        self.b: Value = Converter.to_Value(bias)
        self.active_func = active_func

        if weights is not None:
            self.w = weights
        elif same_weight != 0:
            self.w = [Converter.to_Value(same_weight) for _ in nin]
        else:
            self.w = [Converter.to_Value(random.uniform(-1, 1)) for _ in range(nin)]

    def __call__(self, x: list[Value]) -> Value:
        """ Forward pass of the neuron. """
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return self.active_func(act)

    def parameters(self) -> list[Value]:
        """ Returns all trainable parameters of the neuron. """
        return self.w + [self.b]

    def __repr__(self) -> str:
        return f"Neuron({len(self.w)}, activation={self.active_func.__name__})"


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
