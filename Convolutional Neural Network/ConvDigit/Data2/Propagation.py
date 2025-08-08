
import  numpy as np

from Deep_Neuron_Network import foward_propagation_DNN, back_propagation_DNN, update_DNN
from Convolution_Neuron_Network import foward_propagation_CNN, back_propagation_CNN, update_CNN, add_padding, reshape

def forward_propagation(X, parametres_CNN, parametres_DNN, tuple_size_activation, dimensions_CNN, C_CNN):
    
     #Ajout une dimmension pour chaque image fasse 1*n*n
    X = X.reshape(1, 28, 28)

    new_X = []
    if len(dimensions_CNN) > 1:
            layer = add_padding(X, dimensions_CNN["2"][2])
            tmp = reshape(layer, dimensions_CNN["1"][0], X.shape[2], dimensions_CNN["1"][1], dimensions_CNN["2"][2])
            new_X.append(tmp)

    else:
        tmp = reshape(X, dimensions_CNN["1"][0], X.shape[1], dimensions_CNN["1"][1], 0)
        new_X.append(tmp)

    X = np.concatenate(new_X)

    activations_CNN = foward_propagation_CNN(X, parametres_CNN, tuple_size_activation, dimensions_CNN)
    A = activations_CNN["A" + str(C_CNN)].reshape(activations_CNN["A" + str(C_CNN)].size, 1)
    activation_DNN = foward_propagation_DNN(A, parametres_DNN)

    return activations_CNN, activation_DNN

def back_propagation(activation_DNN, activations_CNN, parametres_DNN, parametres_CNN, dimensions_CNN, tuple_size_activation, C_DNN, y_train):
    
    gradients_DNN, dZ = back_propagation_DNN(activation_DNN, parametres_DNN, y_train)
    gradients_CNN = back_propagation_CNN(activations_CNN, parametres_CNN, dimensions_CNN, dZ, tuple_size_activation)

    return gradients_DNN, gradients_CNN

def update(gradients_CNN, gradients_DNN, parametres_CNN, parametres_DNN, parametres_grad, learning_rate_CNN, learning_rate_DNN, beta1, beta2, C_CNN):

    parametres_CNN = update_CNN(gradients_CNN, parametres_CNN, parametres_grad, learning_rate_CNN, beta1, beta2, C_CNN)
    parametres_DNN = update_DNN(gradients_DNN, parametres_DNN, learning_rate_DNN)

    return parametres_CNN, parametres_DNN