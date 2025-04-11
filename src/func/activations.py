import math
from src.model.value import Value
from src.model.matrix import Matrix
import numpy as np


class ActiveFunction:
    """ Activation Functions for Neural Networks. Supports Matrix input. """

    @staticmethod
    def linier(X: Matrix) -> Matrix:
        """ Linear activation function. Linear(X) = X """
        out_data = [[Value(val.data, (val,), "linier") for val in row] for row in X.data]

        def _backward():
            for i, row in enumerate(X.data):
                for j, val in enumerate(row):
                    val.grad += out_data[i][j].grad  

        for row in out_data:
            for val in row:
                val._backward = _backward  

        return Matrix(out_data)

    @staticmethod
    def relu(X: Matrix) -> Matrix:
        """ ReLU activation function. ReLU(X) = max(0, X) """
        out_data = []
        for row in X.data:
            out_data.append([
                Value(0 if val.data < 0 else val.data, (val,), "ReLU") for val in row
            ])

        def _backward():
            for i, row in enumerate(X.data):
                for j, val in enumerate(row):
                    # Derivative of ReLU = 1 if x > 0 else 0
                    val.grad += (out_data[i][j].data > 0) * out_data[i][j].grad

        for row in out_data:
            for val in row:
                val._backward = _backward

        return Matrix(out_data)

    @staticmethod
    def sigmoid(X: Matrix) -> Matrix:
        """ Sigmoid activation function. Sigmoid(X) = 1 / (1 + exp(-X)) """
        out_data = []
        for row in X.data:
            out_data.append([
                Value(1 / (1 + math.exp(-val.data)), (val,), "sigmoid") for val in row
            ])

        def _backward():
            for i, row in enumerate(X.data):
                for j, val in enumerate(row):
                    sigmoid_val = out_data[i][j].data
                    # Derivative of sigmoid = sigmoid * (1 - sigmoid)
                    val.grad += (sigmoid_val * (1 - sigmoid_val)) * out_data[i][j].grad

        for row in out_data:
            for val in row:
                val._backward = _backward
        return Matrix(out_data)

    @staticmethod
    def tanh(X: Matrix) -> Matrix:
        """ Hyperbolic tangent activation function. tanh(X) = (exp(2X) - 1) / (exp(2X) + 1) """
        out_data = []
        for row in X.data:
            out_data.append([
                Value((math.exp(2 * val.data) - 1) / (math.exp(2 * val.data) + 1), (val,), "tanh") for val in row
            ])

        def _backward():
            for i, row in enumerate(X.data):
                for j, val in enumerate(row):
                    # Derivative of tanh = (2 / (exp(x) - exp(-x)) )^2
                    val.grad += (1 - out_data[i][j].data ** 2) * out_data[i][j].grad

        for row in out_data:
            for val in row:
                val._backward = _backward

        return Matrix(out_data)

    # @staticmethod
    # def softmax(X: Matrix) -> Matrix:
    #     """Numerically stable softmax"""
    #     out_data = []
    #     for row in X.data:
    #         # 1. Find max logit (crucial for stability)
    #         max_logit = max(v.data for v in row)
            
    #         # 2. Compute shifted exponentials
    #         exps = [(v - max_logit).exp() for v in row]  # Uses Value.exp()
    #         sum_exp = sum(exps)
            
    #         # 3. Create softmax outputs
    #         out_row = []
    #         for k in range(len(row)):
    #             # Division creates new Value node
    #             out_val = exps[k] / sum_exp
    #             out_val._op = 'softmax'
                
    #             # 4. Define backward pass
    #             def _backward():
    #                 for j in range(len(row)):
    #                     if j == k:
    #                         row[j].grad += out_val.data * (1 - out_val.data) * out_val.grad
    #                     else:
    #                         row[j].grad += -out_val.data * (exps[j]/sum_exp).data * out_val.grad
                
    #             out_val._backward = _backward
    #             out_row.append(out_val)
            
    #         out_data.append(out_row)
        
    #     return Matrix(out_data)

    # @staticmethod
    # def softmax(X: Matrix) -> Matrix:
    #     """
    #     Stable row-wise softmax for batches.
    #     Input: 50x10 Matrix of Values, Output: 50x10 Matrix (rows sum to 1.0).
    #     """
    #     out_data = []
        
    #     for row in X.data:  # Process each sample independently (50 rows)
    #         # === Step 1: Stable Softmax ===
    #         max_val = max(val.data for val in row)  # Avoid overflow
    #         exp_vals = [math.exp(val.data - max_val) for val in row]
    #         sum_exp = sum(exp_vals)  # Normalization factor
            
    #         # Softmax probabilities (guaranteed sum=1.0)
    #         softmax_vals = [
    #             Value(exp_val / sum_exp, (row[i],), "softmax") 
    #             for i, exp_val in enumerate(exp_vals)
    #         ]
    #         out_data.append(softmax_vals)

    #         # === Step 2: Correct Gradient (Jacobian) ===
    #         def _backward():
    #             n = len(softmax_vals)
    #             for i in range(n):
    #                 for j in range(n):
    #                     # ∂L/∂x_j = Σ_i (∂L/∂y_i * ∂y_i/∂x_j)
    #                     # ∂y_i/∂x_j = y_i * (1_{i=j} - y_j)
    #                     row[j].grad += (
    #                         softmax_vals[i].data * 
    #                         ((1 if i == j else 0) - softmax_vals[j].data) * 
    #                         softmax_vals[i].grad
    #                     )

    #         # Assign backward to all outputs in this row
    #         for val in softmax_vals:
    #             val._backward = _backward
        
    #     return Matrix(out_data)  # 50x10 output

    @staticmethod
    def softmax(X: Matrix) -> Matrix:
        """
        Softmax activation function. 
        Softmax(X_i) = exp(X_i) / ∑_j exp(X_j) (applied row-wise for batches).
        """
        out_data = []
        # Cache intermediate Values for backprop
        exp_values = []  
        sums = []  

        # Step 1: Compute exp(X) and row-wise sums
        for row in X.data:
            exp_row = [val.exp() for val in row]  # exp(X_ij)
            sum_exp = exp_row[0]
            for val in exp_row[1:]:
                sum_exp = sum_exp + val  # ∑_j exp(X_j)
            exp_values.append(exp_row)
            sums.append(sum_exp)

        # Step 2: Compute softmax = exp(X) / sum(exp(X))
        for i, row in enumerate(exp_values):
            out_data.append([val / sums[i] for val in row])  # Softmax(X_i)

        # Step 3: Define backward pass
        def _backward():
            for i, row in enumerate(out_data):
                for j, softmax_val in enumerate(row):
                    # Gradient of softmax: ∂L/∂X_i = softmax_i * (∂L/∂Y_i - ∑_k softmax_k * ∂L/∂Y_k)
                    grad = softmax_val.data * (softmax_val.grad - sum(
                        s.data * s.grad for s in row
                    ))
                    X.data[i][j].grad += grad

        # Attach backward to all output Values
        for row in out_data:
            for val in row:
                val._backward = _backward

        return Matrix(out_data)
        

    @staticmethod
    def exp(X: Matrix) -> Matrix:
        """ Exponential activation function. exp(X) = e^X """
        out_data = []
        for row in X.data:
            out_data.append([Value(math.exp(val.data), (val,), "exp") for val in row])

        def _backward():
            for i, row in enumerate(X.data):
                for j, val in enumerate(row):
                    val.grad += out_data[i][j].data * out_data[i][j].grad

        for row in out_data:
            for val in row:
                val._backward = _backward

        return Matrix(out_data)

    @staticmethod
    def log(X: Matrix) -> Matrix:
        """ Logarithmic activation function. log(X) = ln(X) """
        out_data = []
        for row in X.data:
            out_data.append([Value(math.log(val.data), (val,), "log") for val in row])

        def _backward():
            for i, row in enumerate(X.data):
                for j, val in enumerate(row):
                    val.grad += (1 / val.data) * out_data[i][j].grad

        for row in out_data:
            for val in row:
                val._backward = _backward

        return Matrix(out_data)
