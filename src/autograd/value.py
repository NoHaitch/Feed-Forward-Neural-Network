import math

class Value:
    """ Class to represent a value in a mathematical expression. Used for automatic differentiation.

        Args:
            data (_type_): Numerical value of node. 
            _children (tuple, optional): Children nodes.
            _op (str, optional): Operation performed to get this value.
            label (str, optional): Label for the node. Used to vizualize the graph.
    """

    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data                # the value    
        self.grad = 0.0                 # the gradient of the value
        self._backward = lambda: None   # the function to compute the gradient. This will contain the chain rule for derivatives.
        self._prev = set(_children)     # children nodes
        self._op = _op                  # operation origin
        self.label = label              # node label

    def __repr__(self):
        """ String representation of the object. """
        return f"Value data={self.data} grad={self.grad} op={self._op} label={self.label}"

    ### ===== Base operations ===== ###
    def __add__(self, other):
        """ Add two values. """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out
 
    def __mul__(self, other):
        """ Multiply two values. """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        """ Exponentiate a value. """
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    ### ===== Other operations ===== ###
    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    ### ===== Complex functions ===== ###
    def exp(self):
        """ Exponential function. """
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')
        
        def _backward():
            self.grad += out.data * out.grad 
            out._backward = _backward
        
        return out

    def tanh(self):
        """ Hyperbolic tangent function. """
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward

        return out

    def backward(self):
        """ Backpropagate the gradient through the graph. Using Chain Rule to compute the gradient of the value."""
        
        # Topological sort all the nodes in the graph
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # initialize the gradient of the value to 1.0
        self.grad = 1.0
        
        # backpropagate the gradient through the graph using chain rule
        for v in reversed(topo):
            v._backward()
