import numpy as np
from src.model.value import Value
from src.utils.converter import Converter


class Matrix:
    """ A Matrix of Values 
    
        Attributes:
            data (list[list[Value]]): Matrix data.
            rows (int): Number of rows.
            cols (int): Number of columns.
    """

    def __init__(self, data):
        assert data is not None, "Matrix cannot be initialized with None."

        if isinstance(data, np.ndarray):
            data = data.tolist() 

        assert isinstance(data, list) and len(data) > 0, "Matrix must be initialized with a non-empty list or numpy array."

        if isinstance(data[0], (int, float, Value)):  
            data = [data]  

        self.data = [[Converter.to_Value(val) for val in row] for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0])

        assert isinstance(self.data[0][0], Value), f"Wrong Type. {type(self.data[0][0])}"


    def __repr__(self) -> str:
        """ String representation of the matrix. Displays shape and 5 sample rows. """
        sample_rows = self.data[:5] 
        sample_str = "\n".join([str([val.data for val in row]) for row in sample_rows])
        return f"Matrix with {self.rows} rows and {self.cols} cols\nSample:\n{sample_str}"

    def transpose(self) -> 'Matrix':
        """ Returns the transpose of the matrix. """
        transposed_data = [[self.data[row][col] for row in range(self.rows)] for col in range(self.cols)]
        return Matrix(transposed_data)

    def dot(self, other: 'Matrix') -> 'Matrix':
        """ Performs matrix multiplication (dot product) with another matrix. """
        assert self.cols == other.rows, f"Matrix dimensions do not match for multiplication. {self.cols} != {other.rows}"

        result_data = [
            [
                sum(self.data[i][k] * other.data[k][j] for k in range(self.cols)) 
                for j in range(other.cols)
            ]
            for i in range(self.rows)
        ]

        return Matrix(result_data)

    def add_scalar(self, scalar) -> 'Matrix':
        """ Adds a scalar Value to each element in the matrix. """

        new_data = [[val + scalar for val in row] for row in self.data]

        return Matrix(new_data)
