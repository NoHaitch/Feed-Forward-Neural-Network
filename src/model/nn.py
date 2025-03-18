# Reference: https://github.com/karpathy/micrograd
import random
from .value import Value
from src.model.matrix import Matrix
from src.func.activations import ActiveFunction


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

    def __init__( self, nin: int, mode: str = "zero",  active_func = "relu"):
        """
        Args:
            nin (int): Number of inputs to the neuron.
            mode (str, optional): Weight and bias initialization mode. enum: ['zero', 'uniform', 'normal']
            active_func (str, optional): Activation function.
        """
        valid_mode = {'zero', 'uniform', 'normal'}
        assert mode in valid_mode, f"Weight initialization mode '{mode}' not recognized. Choose from {valid_mode}."

        # Weight Initialization
        if mode == "zero":
            # Zero Initialization
            self.w = [Value(0, label="w") for _ in range(nin)]
            self.b = Value(0, label="b")
            
        elif mode == "uniform":
            # TODO: Random Uniform Distribution with lower bound, upper bound, and seed
            pass

        else:
            # TODO: Random Normal Distribution with mean, variance, and seed
            pass
        
        self.active_func = getattr(ActiveFunction, active_func)


    def __call__(self, X: Matrix) -> Matrix:
        """ Forward pass for a batch (Matrix). """
        assert isinstance(X, Matrix), "Layer input must be a Matrix."

        # Z = wT . x + b
        weighted_sum = X.dot(Matrix([self.w]).transpose())  

        activated_output = self.active_func(weighted_sum.add_scalar(self.b))

        return activated_output
    
    def parameters(self) -> list[Value]:
        """ Returns all trainable parameters of the neuron. """
        return self.w + [self.b]

    def __repr__(self) -> str:
        return f"Neuron(nin={len(self.w)}, activation={self.active_func.__name__})"


class Layer(Module):
    """ Represent a Layer 
    
        Attributes:
            neuron (list[Neuron]): Neuron in the layer.
    """

    def __init__(self, nin, nout, mode: str = "zero", active_func = "relu"):
        """ 
        Args:
            nin (int): Number of input
            nout (int): Number of output
            mode (str): Weight and Bias Initialization mode
            active_funcs (str): Activation function.
        """ 

        self.neurons = [Neuron(nin, mode, active_func) for _ in range(nout)]


    def __call__(self, X: Matrix):
        """ Call the forward pass of each neuron """

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
        return f"Layer of {len(self.neurons)} Neuron: [\n\t{'\n\t'.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    """ Multi-layer perceptron.

        Attributes:
            layers (list): List of layers in the network.
    """

    def __init__(self, nin, nouts, active_funcs = "relu", mode="zero"):
        """
        Args:
            nin (int): Number of inputs to the network.
            nouts (list): Number of outputs from each layer.
            mode (str): Weight and Bias Initialization mode.
            active_funcs (str | list[str]): Activation function(s). Either a function for all layers or a list of functions for each layer.
        """
        sz = [nin] + nouts  # Number of neurons per layer, including input and output
        
        if type(active_funcs) != list:  
            active_funcs = [active_funcs] * (len(nouts) - 1) + ["linier"] 
            # last layer is output layer thus needing to use linier activation function
        
        assert len(active_funcs) == len(nouts), "Number of activation functions must match number of layers."

        # Create layers
        self.layers = [
            Layer(sz[i], sz[i + 1], mode=mode, active_func=active_funcs[i])
            for i in range(len(nouts))
        ]

    def __call__(self, x: Matrix):
        """ Forward pass of the network. """
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of {len(self.layers)} Layers [\n\t{'\n\t'.join(str(layer) for layer in self.layers)}]"
