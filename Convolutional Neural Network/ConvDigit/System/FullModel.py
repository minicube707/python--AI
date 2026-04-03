
import numpy as np
import cupy as cp

from .Convolution_Neuron_Network import CNN, calcul_output_shape
from .Deep_Neuron_Network import DNN

from .Layer import Flatten

from .Evaluation_Metric import CrossEntropyLoss, BinaryCrossEntropy
from .Evaluation_Metric_GPU import CrossEntropyLoss_GPU, BinaryCrossEntropy_GPU

from .Mathematical_function import Softmax, Sigmoide
from .Mathematical_function_GPU import Softmax_GPU, Sigmoide_GPU

class FullModel():

    def __init__(self, hyperparams, structure, loss_metric, output_layer, optimizer):
        
        input_shape = hyperparams.input_shape
        output_shape = hyperparams.output_shape
        alpha = hyperparams.alpha
        padding_mode = hyperparams.padding_mode

        structure_CNN = structure[0]
        structure_DNN = structure[1]

        self.loss_metric = loss_metric
        self.output_layer = output_layer
        self.optimizer = optimizer

        if hyperparams.support == "CPU":
            gpu_mode = False
        else:
            gpu_mode = True

        self.cnn_model = CNN(structure_CNN, input_shape, padding_mode, alpha, optimizer, gpu_mode)

        input_size = input_shape[1]
        for val in self.cnn_model.structure.values():
            o_size = calcul_output_shape(input_size, val[0], val[1], val[2])
            input_size = o_size

        last_CNN_layer = structure_CNN[str(len(structure_CNN))]
        flattened_size = np.int32((np.int32(input_size)**2 * last_CNN_layer[3]))
       
        if isinstance(self.loss_metric, BinaryCrossEntropy):
            output_shape = 1
            hyperparams.output_shape = 1
            
        self.dnn_model= DNN(flattened_size, output_shape, structure_DNN, alpha, optimizer, gpu_mode)
        self.flatten = Flatten()
        
        self.y_pred =  None

        self.show_information(input_shape)
        hyperparams.print_info()

    def forward_propagation(self, X, training):


        self.cnn_model.forward_propagation(X, training)
        res_CNN = self.cnn_model.logits

        self.dnn_model.forward_propagation(self.flatten.forward(res_CNN), training)
        res_DNN = self.dnn_model.logits

        self.y_pred = self.output_layer.forward(res_DNN)
        return self.y_pred
    

    def backward_propagation(self, y):
        
        if isinstance(self.output_layer, Softmax) and isinstance(self.loss_metric, CrossEntropyLoss):
            dZ = self.y_pred - y

        if isinstance(self.output_layer, Softmax_GPU) and isinstance(self.loss_metric, CrossEntropyLoss_GPU):
            dZ = self.y_pred - y

        elif isinstance(self.output_layer, Sigmoide) and isinstance(self.loss_metric, BinaryCrossEntropy):
            dZ = self.y_pred - y[:, np.newaxis]  

        elif isinstance(self.output_layer, Sigmoide_GPU) and isinstance(self.loss_metric, BinaryCrossEntropy_GPU):
            dZ = self.y_pred - y[:, cp.newaxis] 

        else:
            self.loss_metric.forward(self.y_pred, y)
            dZ = self.loss_metric.backward()

        dZ = self.output_layer.backward(dZ)
        dZ = self.dnn_model.backward_propagation(dZ)
        self.cnn_model.backward_propagation(self.flatten.backward(dZ))

    def update(self):
        self.cnn_model.update()
        self.dnn_model.update()

    def show_information(self, input_size):
        self.cnn_model.show_information(input_size)
        self.dnn_model.show_information()

        print("Optimizer: ", self.optimizer.__class__.__name__)
        print("Loss Metric: ", self.loss_metric.__class__.__name__)
        print("Output Layer: ", self.output_layer.__class__.__name__)

    def save(self, path):
        save = {}

        save.update(self.cnn_model.save())
        save.update(self.dnn_model.save())

        np.savez(path, **save)

    def load(self, parameters):
        
        self.cnn_model.set_parameters(parameters)
        self.dnn_model.set_parameters(parameters) 

    def set_alpha(self, alpha):
        self.cnn_model.set_alpha(alpha)
        self.dnn_model.set_alpha(alpha)