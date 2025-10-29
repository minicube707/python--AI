
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


def initialisation(dimension):

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

def softmax(X):
    res = np.array([])
    for i in range(X.shape[0]):
        res = np.append(res, np.exp(X[i,:]) / np.sum(np.exp(X[i,:])))
         
    return res.reshape((X.shape))

def sigmoïde(X):
    return 1/(1 + np.exp(-X))


def foward_propagation(X, parametres):

    activation = {"A0" : X}
    C = len(parametres) // 2
    for c in range(1, C+1):
        Z = parametres["W" + str(c)].dot(activation["A" + str(c-1)]) + parametres["b" + str(c)]
        activation["A" + str(c)] =  sigmoïde(Z)

    return activation


def back_propagation(activation, parametres, y_train):

    m = y_train.shape[1]
    C = len(parametres) // 2

    dZ = activation["A" + str(C)] - y_train
    gradients = {}

    for c in reversed(range(1, C+1)):
        gradients["dW" + str(c)] = 1/m * np.dot(dZ, activation["A" + str(c-1)].T)
        gradients["db" + str(c)] = 1/m * np.sum(dZ, axis=1, keepdims=True)
        
        if c > 1:
            dZ = np.dot(parametres["W" + str(c)].T, dZ) * activation["A" + str(c-1)] * (1 - activation["A" + str(c-1)])

    return gradients   


def update(gradients, parametres, learning_rate):
    
    C = len(parametres) // 2

    for c in range(1, C+1):
        parametres["W" + str(c)] = parametres["W" + str(c)] - learning_rate * gradients["dW" + str(c)]
        parametres["b" + str(c)] = parametres["b" + str(c)] - learning_rate * gradients["db" + str(c)]
    
    return parametres

def dx_log_loss(y_true, y_pred):
    return -1/y_true.size * np.sum((y_true)/(y_pred) - (1 - y_true)/(1 - y_pred))

def learning_progression(X, parametres):
    activations = foward_propagation(X, parametres)
    C = len(parametres) // 2
    A = softmax(activations["A" + str(C)].T)
    return A.T

def predict(X, parametres):
    activations = foward_propagation(X, parametres)
    C = len(parametres) // 2
    Af = activations["A" + str(C)]
    return Af >= 0.5

def log_loss(A, y):
    epsilon = 1e-15 #Pour empecher les log(0) = -inf
    return  1/y.size * np.sum( -y * np.log(A + epsilon) - (1-y)*np.log(1-A + epsilon))

def verification(X, y, activation, parametres, loss, accu):

    loss = np.append(loss, log_loss(activation, y))
    y_pred = predict(X, parametres)
    current_accuracy = accuracy_score(y.flatten(), y_pred.flatten()) 
    accu = np.append(accu, current_accuracy)
    return loss, accu

#Network
def deep_neural_network(X_train, y_train, X_test, y_test, hidden_layer, learning_rate=0.05, n_iteration=1_000):

    #Initialisation
    dimensions = []
    
    dimensions = list(hidden_layer)
    dimensions.insert(0, X_train.shape[0])
    dimensions.append(y_train.shape[0])
    parametres = initialisation(dimensions) 

    train_loss = np.array([])
    train_accu = np.array([])
    train_lear = np.array([])
    test_loss = np.array([])
    test_accu = np.array([])

    C = len(parametres) // 2

    for i in tqdm(range(n_iteration)):

        activation = foward_propagation(X_train, parametres)
        gradients = back_propagation(activation, parametres, y_train)
        parametres = update(gradients, parametres, learning_rate)

        if i%10 == 0:

            #Train
            train_loss, train_accu = verification(X_train, y_train, activation["A" + str(C)], parametres, train_loss, train_accu)
            h = dx_log_loss(y_train, learning_progression(X_train, parametres))
            train_lear = np.append(train_lear, h)

            #Test
            test_activation = foward_propagation(X_test, parametres)
            test_loss, test_accu = verification(X_test, y_test, test_activation["A" + str(C)], parametres, test_loss, test_accu)


    print("L'accuracy final du train_set est de ",train_accu[-1])
    print("L'accuracy final du test_set est de ",test_accu[-1])

    plt.figure(figsize=(12,4))
    plt.subplot(1, 3, 1)
    plt.plot(train_loss, label="Cost function du train_set")
    plt.plot(test_loss, label="Cost function du test_set")
    plt.title("Fonction Cout en fonction des itérations")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(train_accu, label="Accuracy du train_set")
    plt.plot(test_accu, label="Accuracy du test_set")
    plt.title("L'acccuracy en fonction des itérations")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(train_lear, label="Variation de l'apprentisage")
    plt.title("L'acccuracy en fonction des itérations")
    plt.legend()
    plt.show()


    return parametres, dimensions, test_accu[-1]