from src.model.value import Value


class Converter:

    @staticmethod
    def to_Value(val: float|int|Value) -> Value:
        """ Act as a Guard to make sure the type is a Value """
        assert isinstance(val, (int, float, Value)), "Value data must be int or float."

        if(type(val) == Value):
            return val
       
        return Value(val)

    @staticmethod
    def to_Values(values: list[float|int|Value] ) -> list[Value]:
        """ Act as a Guard to make sure the list elements type is Value """
        assert isinstance(values[0], (int, float, Value)), "Value data must be int or float."
        
        if(type(values[0]) == Value):
            return values
        
        return [Converter.to_Value(val) for val in values]