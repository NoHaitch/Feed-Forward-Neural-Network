import math

""" Activation functions for neural networks. + for math expresion
Implemented: 
    - Linear 
    - ReLU
    - Sigmoid
    - Tanh
    - Softmax
    - Exponential
    - Logarithmic
"""


def linier(val):
    """ Linear activation function. Linear(x) = x """
    return val


def relu(val):
    """ ReLU activation function. ReLu(x) = max(0, x) """
    from src.model.value import Value

    out = Value(0 if val.data < 0 else val.data, (val,), "ReLU")

    def _backward():
        # Derivative of ReLU = 1 if x > 0 else 0
        val.grad += (out.data > 0) * out.grad

    out._backward = _backward

    return out


def sigmoid(val):
    """ Sigmoid activation function. Sigmoid(x) = 1 / (1 + exp(-x)) """
    from src.model.value import Value

    x = val.data
    sigmoid = 1 / (1 + math.exp(-x))
    out = Value(sigmoid, (val,), "sigmoid")

    def _backward():
        # Derivative of sigmoid = sigmoid * (1 - sigmoid)
        val.grad += (sigmoid * (1 - sigmoid)) * out.grad

    out._backward = _backward

    return out


def tanh(val):
    """ Hyperbolic tangent activation function. tanh(x) = (exp(2x) - 1) / (exp(2x) + 1) """
    from src.model.value import Value

    x = val.data
    t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
    out = Value(t, (val,), "tanh")

    def _backward():
        # based on the reference: https://github.com/karpathy/micrograd
        # Derivative of tanh = 1 - tanh^2
        # val.grad += (1 - t**2) * out.grad

        # based on spesification
        # Derivative of tanh = (2 / (exp(x) - exp(-x)) )^2
        val.grad += (2 / (math.exp(x) - math.exp(-x))) ** 2 * out.grad

    out._backward = _backward

    return out


def softmax(val):
    """ Softmax activation function. for vector x, softmax(x)i = exp(xi) / sigma j=1 to n (exp(xj)) """
    from src.model.value import Value

    x = val.data
    # Calculate the softmax of a vector x
    exps = [math.exp(i) for i in x]
    out = Value(exps / sum(exps), (val,), "softmax")

    def _backward():
        # Derivative of softmax = softmax * (1 - softmax)
        val.grad += (out.data * (1 - out.data)) * out.grad

    out._backward = _backward

    return out


def exp(val):
    """ Exponential activation function. exp(x) = e^x """
    from src.model.value import Value

    x = val.data
    out = Value(math.exp(x), (val,), "exp")

    def _backward():
        # Derivative of exp = exp
        val.grad += out.data * out.grad

    out._backward = _backward

    return out


def log(val):
    """ Logarithmic activation function. log(x) = ln(x) """
    from src.model.value import Value

    x = val.data
    out = Value(math.log(x), (val,), "log")

    def _backward():
        # Derivative of log = 1/x
        val.grad += (1/x) * out.grad

    out._backward = _backward

    return out