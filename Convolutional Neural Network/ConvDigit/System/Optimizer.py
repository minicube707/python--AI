
from .Optimizer_CPU import Adam_CPU
from .Optimizer_GPU import Adam_GPU

class Adam:
    
    @staticmethod
    def add_layer(hyperparams):
        
        support = hyperparams.support.lower()

        if support == "cpu":
            return  Adam_CPU(hyperparams)
        
        elif support == "gpu":
            return Adam_GPU(hyperparams)
        
        else:
            raise ValueError(f"Unknown support: {support}")