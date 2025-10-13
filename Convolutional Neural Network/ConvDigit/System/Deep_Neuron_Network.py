
import numpy as np

from .Mathematical_function import sigmoide, dx_sigmoide, relu, dx_relu, tanh, dx_tanh, softmax

def show_information_DNN(parametres, dimensions):

    C = len(parametres) // 2

    print("\n============================")
    print("    INITIALISATION DNN")
    print("============================")

    print("\nDétail du reseau de neuron")
    for c in range(1, C+1):
        print("W" + str(c), ":", parametres["W" + str(c)].shape)
        print("B" + str(c), ":", parametres["B" + str(c)].shape)

    for c in range(1, C+1):
        print(parametres["W" + str(c)].shape[0], end="")
        print("->", end="")
    print(parametres["W" + str(c)].shape[1])

    print("\n")
    print("nb_neuron, function")
    for keys, values in dimensions.items():
        print(keys, values)
    print("\n")


def initialisation_DNN(dimension, input_shape, output_shape):

    parametres ={}
    C = len(dimension)

    dimension[str(C)] = (output_shape, dimension[str(C)][1])
    nb_activation = input_shape

    #The weight and bias are initialise to bettween -1 and 1)
    for i in range(1, C+1):
        nb_neuron = dimension[str(i)][0]
        parametres["W" + str(i)] = np.random.rand(nb_activation, nb_neuron) * 2 -1
        parametres["B" + str(i)] = np.random.rand(1, nb_neuron) * 2 -1
        nb_activation = nb_neuron


        if dimension[str(i)][1] not in {"sigmoide", "relu", "tanh"}:
            show_information_DNN(parametres, dimension)
            raise NameError(f"ERROR: Activation function '{dimension[str(i)][1]}' is not defined. Please correct with 'relu', 'sigmoide' ou 'tanh'.")
 
    return parametres


def foward_propagation_DNN(X, parametres, dimension, C, alpha):

    activation = {"A0" : X, "Z0" : X}
    C = len(dimension)

    for i in range(1, C+1):
        Z = np.dot(activation["A" + str(i-1)], parametres["W" + str(i)]) + parametres["B" + str(i)]
        activation["Z" + str(i)] = Z

        if dimension[str(i)][1] == "sigmoide":
            activation["A" + str(i)] =  sigmoide(Z)
        elif dimension[str(i)][1] == "tanh":
            activation["A" + str(i)] =  tanh(Z)
        elif dimension[str(i)][1] == "relu":
            activation["A" + str(i)] =  relu(Z, alpha)

    return activation


def back_propagation_DNN(activation, parametres, dimension, y_train, C, alpha):

    m = y_train.size
    dZ = softmax(activation["A" + str(C)]) - y_train
    gradients = {}  

    for i in reversed(range(1, C+1)):
        gradients["dW" + str(i)] = 1/m * np.dot(activation["A" + str(i-1)].T, dZ)
        gradients["dB" + str(i)] = 1/m * np.mean(dZ, axis=0, keepdims=True)    
        dA = np.dot(dZ, parametres["W" + str(i)].T)
        dA = np.clip(dA, -100, 100)

        if dimension[str(i)][1] == "sigmoide":
            dZ = dA * dx_sigmoide(activation["A" + str(i-1)])
        elif dimension[str(i)][1] == "tanh":
            dZ = dA * dx_tanh(activation["A" + str(i-1)])
        elif dimension[str(i)][1] == "relu":
            dZ = dA * dx_relu(activation["Z" + str(i-1)], alpha)
    
    return gradients, dZ


def update_DNN(gradients, parametres, learning_rate, dimension):
    
    C = len(dimension)

    for c in range(1, C+1):
        parametres["W" + str(c)] = parametres["W" + str(c)] - learning_rate * gradients["dW" + str(c)]
        parametres["B" + str(c)] = parametres["B" + str(c)] - learning_rate * gradients["dB" + str(c)]
    
    return parametres


