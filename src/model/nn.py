# Reference: https://github.com/karpathy/micrograd
import numpy as np
from .value import Value
from src.model.matrix import Matrix
from src.func.activations import ActiveFunction
from typing import Any


class Module:
    """Base class for all neural network modules."""

    def zero_grad(self):
        """Set all gradients to zero. Used before backpropagation."""
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        """Return all the parameters of the Neural Network."""
        return []


class Neuron(Module):
    """Represents a Neuron in a Neural Network.

    Attributes:
        w (list[Value]): Weights of the neuron.
        b (Value): Bias of the neuron.
        active_func (callable[[Value], Value]): Activation function used.
        label (str): Label of the neuron.
    """

    def __init__( self, nin: int, mode: str = "zero",  active_func = "relu", label=None, seed:int=42, **kwargs):
        """
        Args:
            nin (int): Number of inputs to the neuron.
            mode (str, optional): Weight and bias initialization mode. enum: ['zero', 'uniform', 'normal']
            active_func (str, optional): Activation function.
            label (str, optional): Label of the neuron.
        """
        valid_mode = {"zero", "uniform", "normal"}
        assert (
            mode in valid_mode
        ), f"Weight initialization mode '{mode}' not recognized. Choose from {valid_mode}."

        # Weight Initialization
        np.random.seed(seed)
        if mode == "zero":
            # Zero Initialization
            self.w = [Value(0, label=f"{label}_w{i+1}") for i in range(nin)]
            self.b = Value(0, label=f"{label}_b")

        elif mode == "uniform":
            lower_bound = self._validate_param(kwargs, 'lower_bound')
            upper_bound = self._validate_param(kwargs, 'upper_bound')
            random_weights = np.random.uniform(low=lower_bound, high=upper_bound, size=nin)
            self.w = [Value(float(w), label=f"{label}_w{i+1}") for i, w in enumerate(random_weights)]
            self.b = Value(float(np.random.uniform(lower_bound, upper_bound)), label="{label}_b")

        else:   # mode == "normal"
            mean = self._validate_param(kwargs, 'mean')
            variance = self._validate_param(kwargs, 'variance')
            std_dev = np.sqrt(variance) 
            random_weights = np.random.normal(loc=mean, scale=std_dev, size=nin)
            self.w = [Value(float(w), label=f"{label}_w{i+1}") for i, w in enumerate(random_weights)]
            self.b = Value(float(np.random.normal(loc=mean, scale=std_dev)), label="{label}_b")
        self.active_func = getattr(ActiveFunction, active_func)
        self.label = label

    def _validate_param(self, kwargs: dict, name: str) -> Any:
        """Helper to validate required parameters"""
        if name not in kwargs:
            raise ValueError(f"Parameter '{name}' required when mode={self.mode}")
        return kwargs[name]


    def __call__(self, X: Matrix) -> Matrix:
        """Forward pass for a batch (Matrix)."""
        assert isinstance(X, Matrix), "Layer input must be a Matrix."

        # Z = wT . x + b
        weighted_sum = X.dot(Matrix([self.w]).transpose())

        activated_output = self.active_func(weighted_sum.add_scalar(self.b))

        return activated_output

    def parameters(self) -> list[Value]:
        """Returns all trainable parameters of the neuron."""
        return self.w + [self.b]

    def __repr__(self) -> str:
        return f"Neuron(nin={len(self.w)}, activation={self.active_func.__name__}, label={self.label})"


class Layer(Module):
    """Represent a Layer

    Attributes:
        neurons (list[Neuron]): Neurons in the layer.
        label (str): Label of the layer.
    """

    def __init__(self, nin, nout, mode: str = "zero", active_func="relu", label=None, seed:int=42, **kwargs):
        """ 
        Args:
            nin (int): Number of inputs.
            nout (int): Number of outputs.
            mode (str): Weight and Bias Initialization mode.
            active_func (str): Activation function.
            label (str, optional): Label of the layer.
        """
        self.label = label
        self.neurons = [
            Neuron(nin, mode, active_func, label=f"{label}_N{i+1}", seed=seed,**kwargs) for i in range(nout)
        ]

    def __call__(self, X: Matrix):
        """Call the forward pass of each neuron"""

        assert isinstance(X, Matrix), "Layer input must be a Matrix."

        out_data = [
            [neuron(Matrix([row])).data[0][0] for neuron in self.neurons]
            for row in X.data
        ]

        out = Matrix(out_data)

        return out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer(label={self.label}, neurons=[\n\t{'\n\t'.join(str(n) for n in self.neurons)}])"


class MLP(Module):
    """Multi-layer perceptron.

    Attributes:
        layers (list): List of layers in the network.
    """

    def __init__(self, nin, nouts, active_funcs = "relu", mode="zero", seed:int=42, **kwargs):
        """
        Args:
            nin (int): Number of inputs to the network.
            nouts (list): Number of outputs from each layer.
            mode (str): Weight and Bias Initialization mode.
            active_funcs (str | list[str]): Activation function(s). Either a function for all layers or a list of functions for each layer.
        """
        sz = [nin] + nouts 

        if type(active_funcs) != list:
            active_funcs = [active_funcs] * (len(nouts) - 1) + ["linier"]

        assert len(active_funcs) == len(
            nouts
        ), "Number of activation functions must match number of layers."

        self.layers = [
            Layer(
                sz[i],
                sz[i + 1],
                mode=mode,
                active_func=active_funcs[i],
                label=(
                    "Hidden-Layer-1"
                    if i == 0 
                    else (
                        "Output-Layer" if i == len(nouts) - 1 else f"Hidden-Layer-{i+1}"
                    )
                ),
                seed=seed, **kwargs
            )
            for i in range(len(nouts))
        ]

    def __call__(self, x: Matrix):
        """Forward pass of the network."""
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of {len(self.layers)} Layers [\n\t{'\n\t'.join(str(layer) for layer in self.layers)}]"
