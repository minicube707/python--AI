
import  numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from Deep_Neuron_Network import initialisation_DNN
from Convolution_Neuron_Network import show_information, output_shape
from Initialisation_CNN import initialisation_CNN
from Evaluation_Metric import learning_progression, log_loss, accuracy_score, dx_log_loss
from Display_parametre_CNN import display_kernel_and_biais
from Propagation import forward_propagation, back_propagation, update

def convolution_neuron_network(X_train, y_train, X_test, y_test, nb_iteration, hidden_layer, dimensions_CNN \
        , learning_rate_CNN, learning_rate_DNN, beta1, beta2, input_shape):

    dimensions_DNN = {}

    padding_mode = "auto"
    parametres_CNN, parametres_grad, dimensions_CNN, tuple_size_activation = initialisation_CNN(input_shape, dimensions_CNN, padding_mode)

    input_size = 28
    C_CNN = len(dimensions_CNN.keys())

    for val in dimensions_CNN.values():
        o_size = output_shape(input_size, val[0], val[1], val[2])
        input_size = o_size

    dimensions_DNN = list(hidden_layer)
    dimensions_DNN.insert(0, np.int64(np.int64(o_size**2) * np.int64(dimensions_CNN[str(C_CNN)][3])))
    dimensions_DNN.append(y_train.shape[1])

    parametres_DNN = initialisation_DNN(dimensions_DNN)
    C_DNN = len(parametres_DNN) // 2
    show_information(tuple_size_activation, dimensions_CNN)    

    train_loss = np.array([])
    train_accu = np.array([])
    train_lear = np.array([])
    test_loss = np.array([])
    test_accu = np.array([])
    

    #Here 
    #the activation are in different shape, that allow the cross product for more efficy
    #the kernel are vector to do cross product
    #the gradient are vector

    for i in tqdm(range(nb_iteration)):
        for j in range(X_train.shape[0]):
            

            activations_CNN, activation_DNN = forward_propagation(X_train[j], parametres_CNN, parametres_DNN, tuple_size_activation, dimensions_CNN, C_CNN)
            gradients_DNN, gradients_CNN = back_propagation(activation_DNN, activations_CNN, parametres_DNN, parametres_CNN, dimensions_CNN, tuple_size_activation, 
                                                            C_DNN, y_train[j])
            parametres_CNN, parametres_DNN = update(gradients_CNN, gradients_DNN, parametres_CNN, parametres_DNN, parametres_grad, learning_rate_CNN, 
                                                    learning_rate_DNN, beta1, beta2, C_CNN)


            if ((i+j) % 50 == 0):
                # --- Évaluation après chaque epoch ---
                # On évalue sur un petit sous-ensemble fixe pour limiter le bruit
                rand_idx_train = np.random.choice(X_train.shape[0], 50, replace=False)
                rand_idx_test  = np.random.choice(X_test.shape[0],  50, replace=False)

                # Train metrics
                train_loss_epoch = 0
                train_dx_l_epoch = 0
                train_accu_epoch = 0
                for idx in rand_idx_train:
                    pred = learning_progression(
                    X_train[idx], parametres_CNN, parametres_DNN,
                    tuple_size_activation, dimensions_CNN, C_CNN, C_DNN)

                    train_loss_epoch += log_loss(y_train[idx], pred)
                    train_dx_l_epoch += dx_log_loss(y_train[idx], pred)
                    train_accu_epoch += accuracy_score(y_train[idx].flatten(), (pred >= 0.5).flatten())

                train_loss = np.append(train_loss, train_loss_epoch / len(rand_idx_train))
                train_lear = np.append(train_lear, train_dx_l_epoch / len(rand_idx_train))
                train_accu = np.append(train_accu, train_accu_epoch / len(rand_idx_train))

                # Test metrics
                test_loss_epoch = 0
                test_accu_epoch = 0
                for idx in rand_idx_test:
                    pred = learning_progression(
                    X_test[idx], parametres_CNN, parametres_DNN,
                    tuple_size_activation, dimensions_CNN, C_CNN, C_DNN)
                    
                    test_loss_epoch += log_loss(y_test[idx], pred)
                    test_accu_epoch += accuracy_score(y_test[idx].flatten(), (pred >= 0.5).flatten())

                test_loss = np.append(test_loss, test_loss_epoch / len(rand_idx_test))
                test_accu = np.append(test_accu, test_accu_epoch / len(rand_idx_test))

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
    plt.title("La derive de la fonction cout")
    plt.legend()

    plt.show()

    #Display kernel & biais
    #display_kernel_and_biais(parametres_CNN)

    return parametres_CNN, parametres_DNN, dimensions_CNN, dimensions_DNN, test_accu[-1], tuple_size_activation
