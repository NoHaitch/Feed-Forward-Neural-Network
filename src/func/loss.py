# import numpy as np
# from src.model.value import Value
# from src.utils.converter import Converter
# from src.model.matrix import Matrix


# class LossFunction:
#     """ Class to wrap all Loss functions. """

#     MINIMUM_LOG_VALUE:float = 1e-15
    
#     @staticmethod
#     def mse(y_true: Matrix, y_pred: Matrix) -> Value:
#         """ Mean Squared Error (MSE) loss.

#             Args:
#                 y_true (Matrix): True values (Matrix).
#                 y_pred (Matrix): Predicted values (Matrix).
#         """
#         assert isinstance(y_true, Matrix) and isinstance(y_pred, Matrix), "Inputs must be Matrix instances."
#         assert y_true.rows == y_pred.rows and y_true.cols == y_pred.cols, "Matrix dimensions must match."

#         # y_true = Matrix([[Converter.to_Value(val) for val in row] for row in y_true.data])

#         # n = y_true.rows * y_true.cols

#         # loss = sum((yt - yp) ** 2 for yt_row, yp_row in zip(y_true.data, y_pred.data) for yt, yp in zip(yt_row, yp_row)) / n

#         # value_loss = Value(loss.data, (y_pred, y_true), "mse")

#         # value_loss._y_true = y_true
#         # value_loss._y_pred = y_pred

#         # def _backward():
#         #     # Compute gradient ∂L/∂y_pred for each prediction
#         #     for yt_row, yp_row in zip(loss._y_true.data, loss._y_pred.data):
#         #         for yt, yp in zip(yt_row, yp_row):
#         #             # Gradient of MSE:
#         #             # ∂L/∂y_pred = 2 * (y_pred - y_true) / N
#         #             grad = 2 * (yp.data - yt.data) / n
#         #             yp._grad += grad
        
#         # value_loss._backward = _backward
        
#         # return value_loss

#         squared_errors = []
#         for i in range(y_true.rows):
#             for j in range(y_true.cols):
#                 yt = Converter.to_Value(y_true.data[i][j])
#                 yp = y_pred.data[i][j]
#                 squared_errors.append((yp - yt)**2)
        
#         # Sum all squared errors
#         total = sum(squared_errors)
#         loss = total / (y_true.rows * y_true.cols)
        
#         # Store references for backward pass
#         loss._y_pred = y_pred  # Store reference to prediction matrix
        
#         def _backward():
#             # Only need to compute gradients for y_pred (y_true is constant)
#             n = y_true.rows * y_true.cols
#             for i in range(y_pred.rows):
#                 for j in range(y_pred.cols):
#                     # ∂L/∂yp = 2(yp - yt)/n
#                     grad = 2 * (y_pred.data[i][j].data - y_true.data[i][j]) / n
#                     y_pred.data[i][j].grad += grad
        
#         loss._backward = _backward
#         return loss
    

#     @staticmethod
#     def BCE(y_true: Matrix, y_pred: Matrix) -> Value:
#         """ Binary-Cross Entropy

#             Args:
#                 y_true (Matrix): True values (Matrix).
#                 y_pred (Matrix): Predicted values (Matrix).
#         """
#         assert isinstance(y_true, Matrix) and isinstance(y_pred, Matrix), "Inputs must be Matrix instances."
#         assert y_true.rows == y_pred.rows and y_true.cols == y_pred.cols, "Matrix dimensions must match."

#         y_true = Matrix([[Converter.to_Value(val) for val in row] for row in y_true.data])

#         n = y_true.rows * y_true.cols

#         loss = sum(
#         - (yt * np.log(max(yp, LossFunction.MINIMUM_LOG_VALUE)) + (1 - yt) * np.log(max(1 - yp, LossFunction.MINIMUM_LOG_VALUE)))
#         for yt_row, yp_row in zip(y_true.data, y_pred.data)
#         for yt, yp in zip(yt_row, yp_row)
#         ) / n

#         value_loss = Value(loss.data, (y_pred, y_true), "bce")

#         value_loss._y_true = y_true
#         value_loss._y_pred = y_pred

#         def _backward():
#             # Compute gradient ∂L/∂y_pred for each prediction
#             for yt_row, yp_row in zip(loss._y_true, loss._y_pred):
#                 for yt, yp in zip(yt_row, yp_row):
#                     # Gradient of BCE loss with respect to y_pred:
#                     # ∂L/∂y_pred = (y_pred - y_true) / (y_pred * (1 - y_pred)) / N
#                     denominator = max(yp.data, LossFunction.MINIMUM_LOG_VALUE) * (1 - yp.data)
#                     grad = (yp.data - yt) / denominator / n
#                     yp._grad += grad
        
#         value_loss._backward = _backward
        
#         return value_loss
    
#     @staticmethod
#     def CCE(y_true: Matrix, y_pred: Matrix) -> Value:
#         """ Categorical-Cross Entropy

#             Args:
#                 y_true (Matrix): True values (Matrix).
#                 y_pred (Matrix): Predicted values (Matrix).
#         """
#         assert isinstance(y_true, Matrix) and isinstance(y_pred, Matrix), "Inputs must be Matrix instances."
#         assert y_true.rows == y_pred.rows and y_true.cols == y_pred.cols, "Matrix dimensions must match."

#         y_true = Matrix([[Converter.to_Value(val) for val in row] for row in y_true.data])

#         n = y_true.rows * y_true.cols

#         loss = sum(
#         -np.log(max(y_pred[i][y_true[i]].data, LossFunction.MINIMUM_LOG_VALUE))
#         for i in range(n)
#         ) / n

#         value_loss = Value(loss.data, (y_pred, y_true), "cce")

#         value_loss._y_true = y_true
#         value_loss._probs = y_pred

#         def _backward():
#             # Gradient when softmax is separate:
#             # ∂L/∂p_j = -1/p_j if j == y_true, else 0
#             for i in range(n):
#                 true_class = loss._y_true[i]
#                 for j, prob in enumerate(loss._probs[i]):
#                     grad = (-1.0 / max(prob.data, LossFunction.MINIMUM_LOG_VALUE) 
#                         if j == true_class else 0.0) / n
#                     prob._grad += grad
    
#         value_loss._backward = _backward
        
#         return value_loss

import numpy as np
from src.model.value import Value
from src.utils.converter import Converter
from src.model.matrix import Matrix


class LossFunction:
    """ Class to wrap all Loss functions. """
    
    MINIMUM_LOG_VALUE: float = 1e-15
    
    @staticmethod
    def mse(y_true: Matrix, y_pred: Matrix) -> Value:
        """ Mean Squared Error (MSE) loss with proper autograd support.
        
        Args:
            y_true (Matrix): True values (Matrix).
            y_pred (Matrix): Predicted values (Matrix).
            
        Returns:
            Value: Loss value with backward() method for gradient computation
        """
        assert isinstance(y_true, Matrix) and isinstance(y_pred, Matrix), "Inputs must be Matrix instances."
        assert y_true.rows == y_pred.rows and y_true.cols == y_pred.cols, "Matrix dimensions must match."

        y_true_values = Matrix([[Converter.to_Value(val) for val in row] for row in y_true.data])
        
        n = y_true.rows * y_true.cols
        squared_errors = []
        
        for i in range(y_true.rows):
            for j in range(y_true.cols):
                squared_errors.append((y_pred.data[i][j] - y_true_values.data[i][j])**2)
        
        loss = sum(squared_errors) / n
        
        loss._y_true = y_true_values
        loss._y_pred = y_pred
        
        def _backward():
            grad_scale = 2.0 / n
            for i in range(y_pred.rows):
                for j in range(y_pred.cols):
                    # ∂L/∂yp = 2(yp - yt)/n
                    grad = grad_scale * (y_pred.data[i][j].data - y_true_values.data[i][j].data)
                    y_pred.data[i][j].grad += grad
        
        loss._backward = _backward
        
        return loss
    
    @staticmethod
    def bce(y_true: Matrix, y_pred: Matrix) -> Value:
        """ Binary Cross-Entropy loss for multiple output neurons.
        
        Args:
            y_true (Matrix): True binary labels (0 or 1) as Matrix
            y_pred (Matrix): Predicted probabilities (Matrix of Values between 0-1)
            
        Returns:
            Value: Loss value with backward() method for gradient computation
        """

        assert isinstance(y_true, Matrix) and isinstance(y_pred, Matrix), "Inputs must be Matrix instances."
        assert y_true.rows == y_pred.rows and y_true.cols == y_pred.cols, "Matrix dimensions must match."
        
        batch_size = y_true.rows
        num_outputs = y_true.cols
        
        y_true_values = Matrix([[Converter.to_Value(val) for val in row] for row in y_true.data])
        
        loss_terms = []
        for i in range(batch_size):
            for j in range(num_outputs):
                yt = y_true_values.data[i][j]
                yp = y_pred.data[i][j]
                
                term = - (yt * yp.log() + (1 - yt) * (1 - yp).log())
                loss_terms.append(term)
        
        loss = sum(loss_terms) / (batch_size * num_outputs)
        
        loss._y_true = y_true_values
        loss._y_pred = y_pred
        
        def _backward():
            norm_factor = 1.0 / (batch_size * num_outputs)
            for i in range(batch_size):
                for j in range(num_outputs):
                    yt = loss._y_true.data[i][j].data
                    yp = loss._y_pred.data[i][j]

                    grad = (yp.data - yt) / (yp.data * (1 - yp.data)) * norm_factor
                    yp.grad += grad
        
        loss._backward = _backward
        
        return loss
    
    # @staticmethod
    # def cce(y_true: Matrix, y_pred: Matrix) -> Value:
    #     """ Categorical Cross-Entropy for pre-softmax probabilities.
        
    #     Args:
    #         y_true (Matrix): True class INDICES (integer values, not one-hot)
    #         y_pred (Matrix): Predicted probabilities (post-softmax values)
            
    #     Returns:
    #         Value: Loss value with backward() method
    #     """
    #     assert isinstance(y_true, Matrix) and isinstance(y_pred, Matrix), "Inputs must be Matrix instances."
    #     assert y_true.rows == y_pred.rows, "Batch sizes must match."
    #     assert y_true.cols == 1, "y_true should be single-column matrix of class indices"
        
    #     batch_size = y_true.rows
    #     true_classes = [int(y_true.data[i][0].data) for i in range(batch_size)]

    #     loss_terms = []
    #     for i in range(batch_size):
    #         true_class = true_classes[i]
    #         p_true = y_pred.data[i][true_class]
    #         loss_terms.append(-p_true.log())
        
    #     loss = sum(loss_terms) / batch_size

    #     loss._y_true = true_classes
    #     loss._y_pred = y_pred
        
    #     def _backward():
    #         for i in range(batch_size):
    #             true_class = loss._y_true[i]
    #             for j in range(y_pred.cols):
    #                 # ∂L/∂p_j = -1/p_j if j == true_class else 0
    #                 grad = (-1.0 / y_pred.data[i][j].data if j == true_class else 0.0) / batch_size
    #                 y_pred.data[i][j].grad += grad
        
    #     loss._backward = _backward
        
    #     return loss

    @staticmethod
    def cce(y_true: Matrix, y_pred: Matrix) -> Value:
        """ Categorical Cross-Entropy for pre-softmax probabilities """
        assert y_true.rows == y_pred.rows, "Batch sizes must match"
        assert y_true.cols == 1, "y_true should be class indices"

        l = len(y_true.data)
        print("Sample predictions:", [[y_pred.data[i][j].data for j in range(10)] for i in range(l)])  
        print("Sample targets:", [y_true.data[i][0].data for i in range(l)]) 
        
        batch_size = y_true.rows
        true_classes = [int(y_true.data[i][0].data) for i in range(batch_size)]
        
        # Compute loss
        loss_terms = []
        for i in range(batch_size):
            true_class = true_classes[i]
            p_true = y_pred.data[i][true_class]
            loss_terms.append(-p_true.log())
        
        loss = sum(loss_terms) / batch_size
        
        # Store references
        loss._y_true = true_classes
        loss._y_pred = y_pred
        
        def _backward():
            for i in range(batch_size):
                true_class = loss._y_true[i]
                for j in range(y_pred.cols):
                    # ∂L/∂p_j = -1/p_j if j == true_class else 0
                    grad = (-1.0 / y_pred.data[i][j].data if j == true_class else 0.0) / batch_size
                    y_pred.data[i][j].grad += grad
        
        loss._backward = _backward
        return loss