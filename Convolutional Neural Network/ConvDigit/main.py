
import os

from System.Evaluation_Metric import CrossEntropyLoss, BinaryCrossEntropy
from System.Mathematical_function import Softmax, Sigmoide
from System.Optimizer import Adam
from System.Run_model import run_training_pipeline
from System.Dataclasses import Hyperparams, Dataset

module_dir = os.path.dirname(__file__)
os.chdir(module_dir)

# ============================
#         PARAMÈTRES
# ============================

hyperparams = {
    "nb_epoch": 5,
    "batch_size": 2,

    "lr": 0.001,
    "beta1": 0.9,
    "beta2": 0.999,
    "alpha" : 0.001,

    "contamination" : 0.1,

    "padding_mode": "auto",

    "support": "CPU"
}

dataset = {
    "validation_size" : 1_000,
    "validation_frequency" : 50,

    "dataset_size": -1,
    "ratio_test" : 0.2,
}


# ============================
#      PARAMÈTRES  CNN
# ============================

# Structure CNN : (kernel size, stride, padding, nb kernels, type_layer, activations function)
structure_CNN = {
    "1": (5, 1, 0, 32, "conv", "leaky relu", 0.0),
    "2": (2, 2, 0, 1, "pool", "max", 0.0),
    "3": (3, 1, 0, 64, "conv", "leaky relu", 0.0),
    "4": (2, 2, 0, 1, "pool", "max", 0.0),
    "5": (3, 1, 0, 64, "conv", "leaky relu", 0.0)
}


# ============================
#      PARAMÈTRES  DNN
# ============================

# Structure DNN : (number of neurone, activations function) 
structure_DNN = {
    "1": (128, "leaky relu", 0.2),
    "2": (64, "leaky relu", 0.2),
    "3": (64, "leaky relu", 0.2),
    "4": (0,  "leaky relu", 0.2)
}

hyperparams = Hyperparams(**hyperparams)

loss_metric = CrossEntropyLoss()
output_layer = Softmax()    
optimizer = Adam(hyperparams)

hyperparams.add_training_parameters(loss_metric, output_layer, optimizer)

dataset = Dataset(**dataset)
structure = (structure_CNN, structure_DNN)


run_training_pipeline(module_dir, hyperparams, structure, loss_metric, output_layer, optimizer, dataset)