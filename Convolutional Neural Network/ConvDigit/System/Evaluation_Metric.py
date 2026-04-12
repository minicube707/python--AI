
from .Evaluation_Metric_CPU import BinaryCrossEntropy_CPU, CrossEntropyLoss_CPU, MSE_CPU
from .Evaluation_Metric_GPU import BinaryCrossEntropy_GPU, CrossEntropyLoss_GPU, MSE_GPU
"""
============================
Evaluation Metrics Function
============================
"""

class BinaryCrossEntropy:
    
    @staticmethod
    def add_layer(support):
        
        support = support.lower()

        if support == "cpu":
            return BinaryCrossEntropy_CPU()
        
        elif support == "gpu":
            return BinaryCrossEntropy_GPU()
        
        else:
            raise ValueError(f"Unknown support: {support}")

    
class CrossEntropyLoss:
    
    @staticmethod
    def add_layer(support):
        
        support = support.lower()

        if support == "cpu":
            return  CrossEntropyLoss_CPU()
        
        elif support == "gpu":
            return CrossEntropyLoss_GPU()
        
        else:
            raise ValueError(f"Unknown support: {support}")


class MSE:
    
    @staticmethod
    def add_layer(support):
        
        support = support.lower()

        if support == "cpu":
            return  MSE_CPU()
        
        elif support == "gpu":
            return MSE_GPU()
        
        else:
            raise ValueError(f"Unknown support: {support}")
