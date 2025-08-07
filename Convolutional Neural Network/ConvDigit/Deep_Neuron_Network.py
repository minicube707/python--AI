
import numpy as np

from Mathematical_function import sigmoide, softmax

def initialisation_DNN(dimension):

    parametres ={}
    C = len(dimension)

    #The weight and bias are initialise to bettween -1 and 1
    print("\nDétail du reseau de neuron")

    for c in range(1, C):
        #The weight of the parametre is between -1 and 1
        parametres["W" + str(c)] = (np.random.rand(dimension[c], dimension[c-1])*2 -1)
        parametres["b" + str(c)] = (np.random.rand(dimension[c], 1)*2 -1)
        print("W" + str(c), ":", parametres["W" + str(c)].shape)
        print("b" + str(c), ":", parametres["b" + str(c)].shape)

    for c in range(C):
        print(dimension[c], end="")
        if c < C-1:
            print("->", end="")
    print("\n")
    
    return parametres

def foward_propagation_DNN(X, parametres):
    activation = {"A0" : X}
    C = len(parametres) // 2
    for c in range(1, C+1):
        Z = parametres["W" + str(c)].dot(activation["A" + str(c-1)]) + parametres["b" + str(c)]
        activation["A" + str(c)] =  sigmoide(Z)

    return activation


def back_propagation_DNN(activation, parametres, y_train):

    m = y_train.size
    C = len(parametres) // 2
    A = softmax(activation["A" + str(C)].T)
    dZ = A.T - y_train.reshape(y_train.shape[0], 1)

    gradients = {}  

    for c in reversed(range(1, C+1)):

        gradients["dW" + str(c)] = 1/m * np.dot(dZ, activation["A" + str(c-1)].T)
        gradients["db" + str(c)] = 1/m * np.sum(dZ, axis=1, keepdims=True)

        dZ = np.dot(parametres["W" + str(c)].T, dZ) * activation["A" + str(c-1)] * (1 - activation["A" + str(c-1)])
    
    return gradients, dZ


def update_DNN(gradients, parametres, learning_rate):
    
    C = len(parametres) // 2

    for c in range(1, C+1):
        parametres["W" + str(c)] = parametres["W" + str(c)] - learning_rate * gradients["dW" + str(c)]
        parametres["b" + str(c)] = parametres["b" + str(c)] - learning_rate * gradients["db" + str(c)]
    
    return parametres



