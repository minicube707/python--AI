
from dataclasses import dataclass

@dataclass
class Hyperparams:

    nb_epoch: int = 1
    batch_size: int = 32

    lr: float = 0.001
    beta1: float = 0.9
    beta2: float = 0.999
    alpha: float = 0.0

    padding_mode: str = "auto"

    loss_metric : str = ""
    output_layer: str = ""
    optimizer: str = ""

    input_shape: tuple = ()
    output_shape: tuple = ()

    contamination : float = 0.1
    
    def add_training_parameters(self, loss_metric, output_layer, optimizer):

        self.loss_metric = loss_metric.__class__.__name__
        self.output_layer = output_layer.__class__.__name__
        self.optimizer = optimizer.__class__.__name__

    def  add_shape(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
    
    def print_info(self):

        print("\n============================")
        print("  Hyperparameters Setting")
        print("============================")

        print("")
        print("Number epoch: ", self.nb_epoch)
        print("Batch Size: ", self.batch_size)

        print("")
        print("Learning Rate: ", self.lr)
        print("Beta1: ", self.beta1)
        print("Beta2: ", self.beta2)
        print("Alpha: ", self.alpha)

        print("")
        print("Padding Mode: ", self.padding_mode)

        print("")
        print("Loss Metric: ", self.loss_metric)
        print("Ouput Layer: ", self.output_layer)
        print("Optimizer: ", self.optimizer)

        print("")
        print("Input Shape: ", self.input_shape)
        print("Output Shape: ", self.output_shape)


@dataclass
class Dataset:
    dataset_size: int = -1
    ratio_test: float = 0.2
    size_test_set: int = dataset_size * ratio_test
    size_training_set: int = dataset_size - size_test_set
    
    validation_size: int = -1
    validation_frequency: int = -1

    def completion_value(self, y):
        
        if (self.dataset_size == -1 or self.dataset_size > len(y)):

            self.dataset_size = len(y)
            self.size_test_set = int(self.dataset_size * self.ratio_test)
            self.size_training_set = int(self.dataset_size - self.size_test_set)

        if (self.validation_size > self.size_test_set or self.validation_size == -1):
            self.validation_size = self.size_test_set

        if (self.validation_frequency == -1):
            self.validation_frequency = self.size_training_set



    def print_info(self):

        print("\n============================")
        print("    Dataset Setting")
        print("============================")

        print("dataset_size: ", self.dataset_size)
        print("ratio_test: ", self.ratio_test)
        print("size_test_set: ", self.size_test_set)
        print("size_training_set: ", self.size_training_set)

        print("validation_size: ", self.validation_size)
        print("validation_frequency: ", self.validation_frequency)
