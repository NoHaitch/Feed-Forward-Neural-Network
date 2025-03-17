import numpy as np
from src.model.value import Value
from src.utils.converter import Converter


class LossFunction:
    """ Class to wrap all Loss functions. """
    
    @staticmethod
    def mse(y_true: list[Value] | list[int | float], y_pred: list[Value]) -> Value:
        """ Mean Squared Error (MSE) loss.

            Args:
                y_true (list[Value]): True values.
                y_pred (list[Value]): Predicted values.
        """

        assert len(y_true) == len(y_pred), "Length of input must be the same."
        
        y_true = Converter.to_Values(y_true)

        n = len(y_true)
        loss = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / n
        return loss
        
    
    