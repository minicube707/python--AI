
import numpy as np

from .Mathematical_function import sigmoide, dx_sigmoide, relu, dx_relu

def show_information_DNN(parametres, dimensions):

    C = len(parametres) // 2

    print("\n============================")
    print("    INITIALISATION DNN")
    print("============================")

    print("\nDétail du reseau de neuron")
    for c in range(1, C+1):
        print("W" + str(c), ":", parametres["W" + str(c)].shape)
        print("b" + str(c), ":", parametres["b" + str(c)].shape)

    print("")
    print(parametres["W1"].shape[1], end="")
    print("->", end="")
    for c in range(1, C+1):
        print(parametres["W" + str(c)].shape[0], end="")
        if c < C:
            print("->", end="")

    print("\n")
    print("nb_neuron, function")
    for keys, values in dimensions.items():
        print(keys, values)
    print("\n")


def initialisation_DNN(dimension, input_shape, output_shape):

    parametres ={}
    C = len(dimension)
    
    dimension["1"] = (input_shape, dimension["1"][1])
    dimension[str(C)] = (output_shape, dimension[str(C)][1])

    #The weight and bias are initialise to bettween -1 and 1)
    for c in range(1, C):
        #The weight of the parametre is between -1 and 1
        parametres["W" + str(c)] = (np.random.rand(dimension[str(c + 1)][0], dimension[str(c)][0])*2 -1)
        parametres["b" + str(c)] = (np.random.rand(dimension[str(c + 1)][0], 1)*2 -1)

        if dimension[str(c)][1] not in {"sigmoide", "relu"}:
            show_information_DNN(parametres)
            raise NameError(f"ERROR: Activation function '{dimension[c, 1]}' is not defined. Please correct with 'relu' or 'sigmoide'.")
            
    return parametres


def foward_propagation_DNN(X, parametres, dimension, C):

    activation = {"A0" : X}

    for c in range(1, C+1):
        Z = parametres["W" + str(c)].dot(activation["A" + str(c-1)]) + parametres["b" + str(c)]

        if (dimension[str(c)][1] == "sigmoide"):
            activation["A" + str(c)] =  sigmoide(Z)
        elif (dimension[str(c)][1] == "relu"):
            activation["A" + str(c)] =  relu(Z)

    return activation


def back_propagation_DNN(activation, parametres, dimension, y_train, C):

    m = y_train.size
    dZ = activation["A" + str(C)] - y_train.reshape(y_train.shape[0], 1)
    gradients = {}  

    for c in reversed(range(1, C+1)):

        gradients["dW" + str(c)] = 1/m * np.dot(dZ, activation["A" + str(c-1)].T)
        gradients["db" + str(c)] = 1/m * np.sum(dZ, axis=1, keepdims=True)

        if (dimension[str(c)][1] == "sigmoide"):
            dZ = np.dot(parametres["W" + str(c)].T, dZ) * dx_sigmoide(activation["A" + str(c-1)])
        elif (dimension[str(c)][1] == "relu"):
            dZ = np.dot(parametres["W" + str(c)].T, dZ) * dx_relu(activation["A" + str(c-1)])
    
    return gradients, dZ


def update_DNN(gradients, parametres, learning_rate):
    
    C = len(parametres) // 2

    for c in range(1, C+1):
        parametres["W" + str(c)] = parametres["W" + str(c)] - learning_rate * gradients["dW" + str(c)]
        parametres["b" + str(c)] = parametres["b" + str(c)] - learning_rate * gradients["db" + str(c)]
    
    return parametres



