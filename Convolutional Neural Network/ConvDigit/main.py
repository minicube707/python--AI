
import os

from System.Evaluation_Metric import CrossEntropyLoss, BinaryCrossEntropy, MSE
from System.Mathematical_function import Softmax, Sigmoide, Linear
from System.Layers import Flatten, GlobalAveragePooling

from System.Optimizer import Adam
from System.Run_model import run_training_pipeline
from System.Dataclasses import Hyperparams, Dataset

module_dir = os.path.dirname(__file__)
os.chdir(module_dir)


# ============================
#         PARAMÈTRES
# ============================

hyperparams = {
    "nb_epoch": 15,
    "batch_size": 64,

    "lr": 0.001,
    "beta1": 0.9,
    "beta2": 0.999,
    "alpha" : 0.0005,

    "contamination" : 0.1,

    "padding_mode": "same",

    "support": "GPU"
}

dataset = {
    "validation_size" : -1,
    "validation_frequency" : -1,

    "dataset_size": -1,
    "ratio_test" : 0.2,
}


# ============================
#      PARAMÈTRES  CNN
# ============================

# Structure CNN : (kernel size, stride, padding, nb kernels, type_layer, activations function)
structure_CNN = {

    "1": (3, 1, 0, 32,  "conv", "leaky relu", 0.0),
    "2": (3, 1, 0, 32,  "conv", "leaky relu", 0.0),
    "3": (2, 2, 0, 1,   "pool", "max", 0.25),

    "4": (3, 1, 0, 64,  "conv", "leaky relu", 0.0),
    "5": (3, 1, 0, 64,  "conv", "leaky relu", 0.0),
    "6": (2, 2, 0, 1,   "pool", "max", 0.25),
    
    "7": (3, 1, 0, 128,  "conv", "leaky relu", 0.0),
    "8": (3, 1, 0, 128,  "conv", "leaky relu", 0.0),
    "9": (2, 2, 0, 1,   "pool", "max", 0.3),
   
}

# ============================
#      PARAMÈTRES  DNN
# ============================

# Structure DNN : (number of neurone, activations function) 
structure_DNN = {
    "1": (128, "leaky relu", 0.5),
    "2": (1,   "leaky relu", 0.0)
}

hyperparams = Hyperparams(**hyperparams)

loss_metric = CrossEntropyLoss()
output_layer = Softmax()    
optimizer = Adam()
transition_layer = GlobalAveragePooling()

hyperparams.add_training_parameters(loss_metric, output_layer, optimizer, transition_layer)

dataset = Dataset(**dataset)
structure = (structure_CNN, structure_DNN)

run_training_pipeline(module_dir, hyperparams, structure, loss_metric, output_layer, optimizer, transition_layer, dataset)