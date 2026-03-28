
import numpy as np

from .Convolution_Neuron_Network import CNN, calcul_output_shape
from .Deep_Neuron_Network import DNN
from .Evaluation_Metric import CrossEntropyLoss
from .Layer import Flatten
from .Mathematical_function import Softmax

class FullModel():

    def __init__(self, X, y, 
                 dimensions_CNN, dimensions_DNN, 
                 optimizer, loss_metric, output_layer, 
                 input_shape, alpha, padding_mode):
        
        input_size = X.shape[2]
        for val in dimensions_CNN.values():
            o_size = calcul_output_shape(input_size, val[0], val[1], val[2])
            input_size = o_size

        last_CNN_layer = dimensions_CNN[str(len(dimensions_CNN))]
        flattened_size = np.int32((np.int32(input_size)**2 * last_CNN_layer[3]))

        self.cnn_model = CNN(dimensions_CNN, input_shape, padding_mode, alpha, optimizer)
        self.dnn_model= DNN(y, flattened_size, dimensions_DNN, alpha, optimizer)
        self.loss_metric = loss_metric
        self.output_layer = output_layer
        self.flatten = Flatten()
        self.y_pred =  None

        self.show_information(input_shape)

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

        else:
            self.loss_metric.forward(self.y_pred, y)
            dA = self.loss_metric.backward()
            dZ = self.output_layer.backward(dA)

        dZ = self.dnn_model.backward_propagation(dZ)
        self.cnn_model.backward_propagation(self.flatten.backward(dZ))

    def update(self):
        self.cnn_model.update()
        self.dnn_model.update()

    def show_information(self, input_size):
        self.cnn_model.show_information(input_size)
        self.dnn_model.show_information()