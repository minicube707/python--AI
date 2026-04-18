
import numpy as np

from .Convolution_Neuron_Network import CNN, calcul_output_shape
from .Deep_Neuron_Network import DNN

class FullModel():

    def __init__(self, hyperparams, structure, loss_metric, output_layer, optimizer, transition_layer):

        # =========================
        # 🔧 Hyperparameters
        # =========================
        self.support = hyperparams.support
        input_shape = hyperparams.input_shape
        output_shape = hyperparams.output_shape
        alpha = hyperparams.alpha
        padding_mode = hyperparams.padding_mode

        structure_CNN, structure_DNN = structure

        # =========================
        # 🧱 Layers init
        # =========================
        self.loss_metric = loss_metric.add_layer(self.support)
        self.output_layer = output_layer.add_layer(self.support)
        self.optimizer = optimizer.add_layer(hyperparams)
        self.transition_layer = transition_layer.add_layer(self.support)

        # =========================
        # 🧠 CNN
        # =========================
        self.cnn_model = CNN(
            structure_CNN,
            input_shape,
            padding_mode,
            alpha,
            self.optimizer,
            self.support
        )

        # =========================
        # 🔄 Transition layer
        # =========================
        layer_type = self.transition_layer.class_

        last_layer = structure_CNN[str(len(structure_CNN))]
        channels = last_layer[3]

        if layer_type == "Flatten":
            spatial_dim = int(self.get_output_shape(input_shape))
            flattened_size = spatial_dim ** 2 * channels

        elif layer_type == "GlobalAveragePooling":
            flattened_size = channels

        else:
            raise ValueError(f"Unknown transition_layer: {layer_type}")

        # =========================
        # 🎯 Output adaptation
        # =========================
        if self.loss_metric.class_ == "BinaryCrossEntropy":
            output_shape = 1
            hyperparams.output_shape = 1

        # =========================
        # 🧠 DNN
        # =========================
        self.dnn_model = DNN(
            flattened_size,
            output_shape,
            structure_DNN,
            alpha,
            self.optimizer,
            self.support
        )

        # =========================
        # 📦 Misc
        # =========================
        self.y_pred = None

        self.show_information(input_shape)
        hyperparams.print_info()

    def get_output_shape(self, input_shape):
        
        input_size = input_shape[1]
        for val in self.cnn_model.structure.values():
            o_size = calcul_output_shape(input_size, val[0], val[1], val[2])
            input_size = o_size
        
        return o_size
    
    
    def forward_propagation(self, X, training):

        self.cnn_model.forward_propagation(X, training)
        res_CNN = self.cnn_model.logits

        res_flat = self.transition_layer.forward(res_CNN)
        
        self.dnn_model.forward_propagation(res_flat, training)
        res_DNN = self.dnn_model.logits

        self.y_pred = self.output_layer.forward(res_DNN)
        return self.y_pred
    

    def backward_propagation(self, y):
        
        m = y.shape[0]  # batch size
        
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        assert self.y_pred.shape == y.shape, f"{self.y_pred.shape} vs {y.shape}"
                
        if self.output_layer.class_ == "Softmax" and self.loss_metric.class_ == "CrossEntropyLoss":
            dZ = (self.y_pred - y) / m
            
        elif self.output_layer.class_ == "Sigmoide" and self.loss_metric.class_ == "BinaryCrossEntropy":
            dZ = (self.y_pred - y) / m
            
        else:
            self.loss_metric.forward(y, self.y_pred)
            dZ = self.loss_metric.backward()
            dZ = self.output_layer.backward(dZ)
        
        dZ = self.dnn_model.backward_propagation(dZ)
        dZ = self.transition_layer.backward(dZ)
        self.cnn_model.backward_propagation(dZ)

    def update(self):
        self.cnn_model.update()
        self.dnn_model.update()

    def show_information(self, input_size):
        self.cnn_model.show_information(input_size)
        self.dnn_model.show_information()

        total_number_parameter = self.cnn_model.get_nb_parameter() + self.dnn_model.get_nb_parameter()
        print("\nTotal Number of parameter:", f"{total_number_parameter:,}".replace(",", " "))
        
    def save(self, path):
        save = {}

        save.update(self.cnn_model.save())
        save.update(self.dnn_model.save())

        np.savez(path, **save)

    def load(self, parameters):
        
        self.cnn_model.set_parameters(parameters)
        self.dnn_model.set_parameters(parameters) 
