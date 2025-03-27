import numpy as np
from src.model.value import Value
from src.utils.converter import Converter
from src.model.matrix import Matrix


class LossFunction:
    """ Class to wrap all Loss functions. """
    
    @staticmethod
    def mse(y_true: Matrix, y_pred: Matrix) -> Value:
        """ Mean Squared Error (MSE) loss.

            Args:
                y_true (Matrix): True values (Matrix).
                y_pred (Matrix): Predicted values (Matrix).
        """
        assert isinstance(y_true, Matrix) and isinstance(y_pred, Matrix), "Inputs must be Matrix instances."
        assert y_true.rows == y_pred.rows and y_true.cols == y_pred.cols, "Matrix dimensions must match."

        y_true = Matrix([[Converter.to_Value(val) for val in row] for row in y_true.data])

        n = y_true.rows * y_true.cols

        loss = sum((yt - yp) ** 2 for yt_row, yp_row in zip(y_true.data, y_pred.data) for yt, yp in zip(yt_row, yp_row)) / n
        
        return loss
    
    