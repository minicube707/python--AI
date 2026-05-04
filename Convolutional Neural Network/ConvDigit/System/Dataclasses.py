
import cupy as cp
import numpy as np
from dataclasses import dataclass, field

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
    transition_layer: str = ""

    input_shape: tuple = ()
    output_shape: tuple = ()

    contamination : float = 0.0
    
    support: str = "CPU"


    def __post_init__(self):
        # input_shape
        if isinstance(self.input_shape, (list, tuple)):
            self.input_shape = tuple(self.input_shape)
        elif self.input_shape:
            self.input_shape = (self.input_shape,)
        else:
            self.input_shape = ()

    def add_training_parameters(self, loss_metric, output_layer, optimizer, transition_layer):

        self.loss_metric = loss_metric.__class__.__name__
        self.output_layer = output_layer.__class__.__name__
        self.optimizer = optimizer.__class__.__name__
        self.transition_layer = transition_layer.__class__.__name__

    def  add_shape(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
    
    def check_support(self):

        if self.support not in ["CPU", "GPU"]:
            print(f"ERROR: support '{self.support}' is not defined. Please correct with 'CPU' or 'GPU'.")
            exit(0)

        if self.support == "GPU":
            try:
                gpu_count = cp.cuda.runtime.getDeviceCount()
            except cp.cuda.runtime.CUDARuntimeError:
                gpu_count = 0

            if gpu_count == 0:
                print("\nERROR: No GPU found. Switching to CPU mode.")
                self.support = "CPU"

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
        print("Output Layer: ", self.output_layer)
        print("Optimizer: ", self.optimizer)
        print("Transition Layer: ", self.transition_layer)

        print("")
        print("Input Shape: ", self.input_shape)
        print("Output Shape: ", self.output_shape)

        print("")
        print("Support: ", self.support)

        if (self.support ==  "GPU"):
            print("Num GPUs:", cp.cuda.runtime.getDeviceCount())
            print("GPU name:", cp.cuda.runtime.getDeviceProperties(0)['name'])
            print("Memory bytes (free, total): ", cp.cuda.Device().mem_info) 


@dataclass
class Dataset:
    dataset_size: int = -1
    ratio_test: float = 0.0
    size_test_set: int = dataset_size * ratio_test
    size_training_set: int = dataset_size - size_test_set
    
    validation_size: int = -1
    validation_frequency: int = -1

    class_to_idx : dict = field(default_factory=dict)
    
    def completion_value(self, dataset_size, train_size, test_size, batch_size, is_full_data):
        
        if (self.dataset_size == -1 or self.dataset_size > dataset_size):
            
            self.dataset_size = dataset_size
            
            if (is_full_data):
                self.size_test_set = int(self.dataset_size * self.ratio_test)
                self.size_training_set = int(self.dataset_size - self.size_test_set)

            else:
                self.size_training_set = train_size
                self.size_test_set = test_size
                self.ratio_test = float(self.size_test_set / self.dataset_size)
                
        if (self.validation_size > self.size_test_set or self.validation_size == -1):
            self.validation_size = self.size_test_set
            
        if (self.validation_frequency == -1):
            self.validation_frequency = int(np.ceil(self.size_training_set / batch_size))



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
        
        print("\nclass_to_idx:")
        for key, value in self.class_to_idx.items():
            print(f"Class: {key:<20}  Index: {value}")
