
import  numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from Deep_Neuron_Network import initialisation_DNN
from Convolution_Neuron_Network import show_information, output_shape
from Initialisation_CNN import initialisation_CNN
from Evaluation_Metric import log_loss, accuracy_score, activation, dx_log_loss, confidence_score
from Display_parametre_CNN import display_kernel_and_biais
from Propagation import forward_propagation, back_propagation, update

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
    C_DNN = len(parametres_DNN) // 2
    show_information(tuple_size_activation, dimensions_CNN)    

    train_loss = np.array([])
    train_accu = np.array([])
    train_lear = np.array([])
    train_conf = np.array([])

    test_loss = np.array([])
    test_accu = np.array([])
    test_lear = np.array([])
    test_conf = np.array([])  

    #Here 
    #the activation are in different shape, that allow the cross product for more efficy
    #the kernel are vector to do cross product
    #the gradient are vector

    k = 0
    pred = np.zeros(y_train[0].shape)
    for _ in range(nb_iteration):
        for j in tqdm(range(X_train.shape[0])):
            
            
            while (accuracy_score(y_train[j].flatten(), pred.flatten()) == 0 or confidence_score(pred) < 0.15):
                activations_CNN, activation_DNN = forward_propagation(X_train[j], parametres_CNN, parametres_DNN, tuple_size_activation, dimensions_CNN, C_CNN)
                gradients_DNN, gradients_CNN = back_propagation(activation_DNN, activations_CNN, parametres_DNN, parametres_CNN, dimensions_CNN, tuple_size_activation, 
                                                                C_DNN, y_train[j])
                parametres_CNN, parametres_DNN = update(gradients_CNN, gradients_DNN, parametres_CNN, parametres_DNN, parametres_grad, learning_rate_CNN, 
                                                        learning_rate_DNN, beta1, beta2, C_CNN)
                
                pred = activation(X_train[j], parametres_CNN, parametres_DNN, tuple_size_activation, dimensions_CNN, C_CNN, C_DNN)

                k += 1
                if (k % 100 == 0):
                    # --- Évaluation après chaque epoch ---
                    # On évalue sur un petit sous-ensemble fixe pour limiter le bruit
                    rand_idx_train = np.random.choice(X_train.shape[0], 50, replace=False)
                    rand_idx_test  = np.random.choice(X_test.shape[0],  50, replace=False)

                    # Train metrics
                    train_loss_epoch = 0
                    train_dx_l_epoch = 0
                    train_accu_epoch = 0
                    train_conf_epoch = 0

                    for idx in rand_idx_train:
                        pred = activation(
                        X_train[idx], parametres_CNN, parametres_DNN,
                        tuple_size_activation, dimensions_CNN, C_CNN, C_DNN)

                        train_loss_epoch += log_loss(y_train[idx], pred)
                        train_dx_l_epoch += dx_log_loss(y_train[idx], pred)
                        train_accu_epoch += accuracy_score(y_train[idx].flatten(), pred.flatten())
                        train_conf_epoch += confidence_score(pred)

                    train_loss = np.append(train_loss, train_loss_epoch / len(rand_idx_train))
                    train_lear = np.append(train_lear, train_dx_l_epoch / len(rand_idx_train))
                    train_accu = np.append(train_accu, train_accu_epoch / len(rand_idx_train))
                    train_conf = np.append(train_conf, train_conf_epoch / len(rand_idx_train))

                    
                    # Test metrics
                    test_loss_epoch = 0
                    test_dx_l_epoch = 0
                    test_accu_epoch = 0
                    test_conf_epoch = 0

                    for idx in rand_idx_test:
                        pred = activation(
                        X_test[idx], parametres_CNN, parametres_DNN,
                        tuple_size_activation, dimensions_CNN, C_CNN, C_DNN)
                        
                        test_loss_epoch += log_loss(y_test[idx], pred)
                        test_dx_l_epoch += dx_log_loss(y_test[idx], pred)
                        test_accu_epoch += accuracy_score(y_test[idx].flatten(), pred.flatten())
                        test_conf_epoch += confidence_score(pred)

                    test_loss = np.append(test_loss, test_loss_epoch / len(rand_idx_test))
                    test_lear = np.append(test_lear, test_dx_l_epoch / len(rand_idx_test))
                    test_accu = np.append(test_accu, test_accu_epoch / len(rand_idx_test))
                    test_conf = np.append(test_conf, test_conf_epoch / len(rand_idx_test))
                

    print(accuracy_score(y_train[j].flatten(), pred.flatten()) == 0)
    print(confidence_score(pred) < 0.15)
    
    print(f"L'accuracy final du train_set est de {train_accu[-1]:.5f}")
    print(f"L'accuracy final du test_set est de {test_accu[-1]:.5f}")
    print(f"Le confidence socre final du test_set est de {test_conf[-1]:.5f}")

    # Display info during learning
    fig, axs = plt.subplots(1, 4, figsize=(16, 4), sharex=True)

    # 1. Fonction de coût
    axs[0].plot(train_loss, label="Cost function du train_set")
    axs[0].plot(test_loss, label="Cost function du test_set")
    axs[0].set_title("Fonction Cout en fonction des itérations")
    axs[0].legend()

    # 2. Dérivée de la fonction coût
    axs[1].plot(train_lear, label="Variation de l'apprentisage du train_set")
    axs[1].plot(test_lear, label="Variation de l'apprentisage du test_set")
    axs[1].set_title("La dérive de la fonction coût")
    axs[1].legend()

    # 3. Accuracy
    axs[2].plot(train_accu, label="Accuracy du train_set")
    axs[2].plot(test_accu, label="Accuracy du test_set")
    axs[2].set_title("L'accuracy en fonction des itérations")
    axs[2].legend()

    # 4. Score de confiance
    axs[3].plot(train_conf, label="Confidence score du train_set")
    axs[3].plot(test_conf, label="Confidence score du test_set")
    axs[3].set_title("Le confidence score en fonction des itérations")
    axs[3].legend()

    plt.tight_layout()
    plt.show()


    #Display kernel & biais
    #display_kernel_and_biais(parametres_CNN)

    return parametres_CNN, parametres_DNN, dimensions_CNN, dimensions_DNN, test_accu[-1], tuple_size_activation
