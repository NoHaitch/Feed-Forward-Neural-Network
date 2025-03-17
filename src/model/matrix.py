from .value import Value
from src.utils.converter import Converter


class Matrix:
    """ A Matrix of Values 
    
        Attributes:
            data (list[list[Value]]): Matrix data.
            rows (int): Number of rows.
            cols (int): Number of columns.
    """

    def __init__(self, data):
        assert len(data) or len(data[0]) > 0, "Empty Matrix"

        self.data = [[Converter.to_Value(vals) for vals in row] for row in data]
        self.rows = len(data)
        self.cols = len(data[0])

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
            [sum(self.data[i][k] * other.data[k][j] for k in range(self.cols)) for j in range(other.cols)]
            for i in range(self.rows)
        ]

        return Matrix(result_data)
