from src.model.value import Value


class RegularizationFunctions:
    """ Class to wrap all Regularization functions. """

    @staticmethod
    def l1(params : list[float]) -> Value:
        """
        Apply L1 regularization (Lasso) to model weights
        Returns a Value representing the L1 penalty
        """
        LAMBDA_VALUE = 0.05

        l1_penalty = Value(0)
        
        for p in params:
            l1_penalty = l1_penalty + Value(p).abs()

        l1_penalty = l1_penalty * LAMBDA_VALUE
        
        return l1_penalty
    
    @staticmethod
    def l2(params : list[float]) -> Value:
        """
        Apply L2 regularization (Ridge) to model weights
        Returns a Value representing the L2 penalty
        """
        LAMBDA_VALUE = 0.005
        
        l2_penalty = Value(0)
    
        for p in params:
            l2_penalty = l2_penalty + Value(p) * Value(p)

        l2_penalty = l2_penalty * (LAMBDA_VALUE / 2)
        
        return l2_penalty
    
    @staticmethod
    def none(params : list[float]) -> Value:
        """
        Apply no regularization
        """
    
        return Value(0.0)