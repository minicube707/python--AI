
import  numpy as np

from .Deep_Neuron_Network import foward_propagation_DNN, back_propagation_DNN, update_DNN
from .Convolution_Neuron_Network import foward_propagation_CNN, back_propagation_CNN, update_CNN, add_padding, reshape

def forward_propagation(X, parametres_CNN, parametres_DNN, tuple_size_activation, dimensions_CNN, C_CNN, dimensions_DNN, C_DNN, alpha):
    
     #Ajout une dimmension pour chaque image fasse 1*n*n
    N = X.size  # Nombre total d'éléments dans le vecteur
    sqrt_N = int(np.sqrt(N))
    shape = (1, sqrt_N, sqrt_N)
    X = X.reshape(shape)

    new_X = []
    if len(dimensions_CNN) > 1:
            layer = add_padding(X, dimensions_CNN["2"][2])
            tmp = reshape(layer, dimensions_CNN["1"][0], X.shape[2], dimensions_CNN["1"][1], dimensions_CNN["2"][2])
            new_X.append(tmp)

    else:
        tmp = reshape(X, dimensions_CNN["1"][0], X.shape[1], dimensions_CNN["1"][1], 0)
        new_X.append(tmp)

    X = np.concatenate(new_X)

    activations_CNN = foward_propagation_CNN(X, parametres_CNN, tuple_size_activation, dimensions_CNN, alpha)
    A = activations_CNN["A" + str(C_CNN)].reshape(1, activations_CNN["A" + str(C_CNN)].size)
    Z = activations_CNN["Z" + str(C_CNN)].reshape(1, activations_CNN["Z" + str(C_CNN)].size)
    activation_DNN = foward_propagation_DNN(A, Z, parametres_DNN, dimensions_DNN, C_DNN, alpha)

    return activations_CNN, activation_DNN

def back_propagation(activation_DNN, activations_CNN, parametres_DNN, parametres_CNN, dimensions_CNN, tuple_size_activation, dimension_DNN, C_DNN, y_train, alpha):
    
    gradients_DNN, dZ = back_propagation_DNN(activation_DNN, parametres_DNN, dimension_DNN, y_train, C_DNN, alpha)
    gradients_CNN = back_propagation_CNN(activations_CNN, parametres_CNN, dimensions_CNN, dZ, tuple_size_activation, alpha)

    return gradients_DNN, gradients_CNN

def update(gradients_CNN, gradients_DNN, parametres_CNN, parametres_DNN, parametres_grad, learning_rate_CNN, learning_rate_DNN, dimension_DNN, beta1, beta2, C_CNN):

    parametres_CNN = update_CNN(gradients_CNN, parametres_CNN, parametres_grad, learning_rate_CNN, beta1, beta2, C_CNN)
    parametres_DNN = update_DNN(gradients_DNN, parametres_DNN, learning_rate_DNN, dimension_DNN)

    return parametres_CNN, parametres_DNN