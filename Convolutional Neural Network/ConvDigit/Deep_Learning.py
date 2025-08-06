
import  numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from Deep_Neuron_Network import initialisation_DNN, foward_propagation_DNN, back_propagation_DNN, update_DNN
from Convolution_Neuron_Network import foward_propagation_CNN, back_propagation_CNN, update_CNN, show_information, output_shape
from Initialisation_CNN import initialisation_CNN
from Evaluation_Metric import verification, dx_log_loss, learning_progression
from Display_parametre_CNN import display_kernel_and_biais


def foward_propagation(X, parametres_CNN, parametres_DNN, tuple_size_activation, dimensions_CNN, C_CNN):

    activations_CNN = foward_propagation_CNN(X, parametres_CNN, tuple_size_activation, dimensions_CNN)
    activation_DNN = foward_propagation_DNN(activations_CNN["A" + str(C_CNN)].flatten(), parametres_DNN)

    return activations_CNN, activation_DNN

def back_propagation(activation_DNN, activations_CNN, parametres_DNN, parametres_CNN, dimensions_CNN, tuple_size_activation, C_DNN, y_train):
    
    gradients_DNN = back_propagation_DNN(activation_DNN, parametres_DNN, y_train)
    gradients_CNN = back_propagation_CNN(activations_CNN, parametres_CNN, dimensions_CNN, activation_DNN["A" + str(C_DNN)], tuple_size_activation)

    return gradients_DNN, gradients_CNN

def update(gradients_CNN, gradients_DNN, parametres_CNN, parametres_DNN, parametres_grad, learning_rate_CNN, learning_rate_DNN, beta1, beta2, C_CNN):

    parametres_CNN = update_CNN(gradients_CNN, parametres_CNN, parametres_grad, learning_rate_CNN, beta1, beta2, C_CNN)
    parametres_DNN = update_DNN(gradients_DNN, parametres_DNN, learning_rate_DNN)

    return parametres_CNN, parametres_DNN

def convolution_neuron_network(X_train, y_train, X_test, y_test, nb_iteration, hidden_layer, dimensions_CNN \
        , learning_rate_CNN, learning_rate_DNN, beta1, beta2, input_shape):

    dimensions_DNN = {}

    padding_mode = "auto"
    parametres_CNN, parametres_grad, dimensions_CNN, tuple_size_activation = initialisation_CNN(input_shape, dimensions_CNN, padding_mode)

    input_size = 8
    C_CNN = len(dimensions_CNN.keys())

    for val in dimensions_CNN.values():
        o_size = output_shape(input_size, val[0], val[1], val[2])
        input_size = o_size

    dimensions_DNN = list(hidden_layer)
    dimensions_DNN.insert(0, o_size**2 * dimensions_CNN[str(C_CNN)][3])
    dimensions_DNN.append(y_train.shape[1])

    parametres_DNN = initialisation_DNN(dimensions_DNN)

    show_information(tuple_size_activation, dimensions_CNN)    

    train_loss = np.array([])
    train_accu = np.array([])
    train_lear = np.array([])
    test_loss = np.array([])
    test_accu = np.array([])
    C_CNN = len(dimensions_CNN.keys())
    C_DNN = len(parametres_DNN) // 2

    #Here 
    #the activation are in different shape, that allow the cross product for more efficy
    #the kernel are vector to do cross product
    #the gradient are vector

    for i in tqdm(range(nb_iteration)):
        for j in range(X_train.shape[0]):
            
            activations_CNN, activation_DNN = foward_propagation(X_train[j], parametres_CNN, parametres_DNN, tuple_size_activation, dimensions_CNN, C_CNN)
            gradients_DNN, gradients_CNN = back_propagation(activation_DNN, activations_CNN, parametres_DNN, parametres_CNN, dimensions_CNN, tuple_size_activation, C_DNN, y_train[j])
            parametres_CNN, parametres_DNN = update(gradients_CNN, gradients_DNN, parametres_CNN, parametres_DNN, parametres_grad, learning_rate_CNN, learning_rate_DNN, beta1, beta2, C_CNN)


        if i % 100 == 0:

            #Train
            train_loss, train_accu = verification(X_train, y_train, activation_DNN["A" + str(C_DNN)], parametres, train_loss, train_accu)
            h = dx_log_loss(y_train, learning_progression(X_train, parametres))
            train_lear = np.append(train_lear, h)

            #Test
            test_activation = foward_propagation_DNN(X_test, parametres)
            test_loss, test_accu = verification(X_test, y_test, test_activation["A" + str(C_DNN)], parametres, test_loss, test_accu)



    print(f"L'accuracy final du train_set est de {train_accu[-1]:.5f}")
    print(f"L'accuracy final du test_set est de {test_accu[-1]:.5f}")

    #Displau info of during the learing
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

    #Display kernel & biais
    display_kernel_and_biais(parametres)

    return parametres_CNN, parametres_DNN, dimensions_CNN, dimensions_DNN, test_accu[-1]
