
import  numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import deepcopy

from Convolution_Neuron_Network import show_information
from Evaluation_Metric import activation, log_loss, accuracy_score, dx_log_loss, confidence_score
from Display_parametre_CNN import display_kernel_and_biais
from Propagation import forward_propagation, back_propagation, update


def train_one_sample(X, y, parametres_CNN, parametres_DNN, parametres_grad,
                     dimensions_CNN, tuple_size_activation, C_CNN, C_DNN,
                     learning_rate_CNN, learning_rate_DNN, beta1, beta2,
                     max_attempts=100):

    for _ in range(max_attempts):
        # Forward
        activations_CNN, activations_DNN = forward_propagation(
            X, parametres_CNN, parametres_DNN, tuple_size_activation, dimensions_CNN, C_CNN)

        # Backward
        gradients_DNN, gradients_CNN = back_propagation(
            activations_DNN, activations_CNN, parametres_DNN, parametres_CNN,
            dimensions_CNN, tuple_size_activation, C_DNN, y)

        # Update
        parametres_CNN, parametres_DNN = update(
            gradients_CNN, gradients_DNN, parametres_CNN, parametres_DNN,
            parametres_grad, learning_rate_CNN, learning_rate_DNN, beta1, beta2, C_CNN)

        # Prediction
        pred = activation(X, parametres_CNN, parametres_DNN, tuple_size_activation, dimensions_CNN, C_CNN, C_DNN)

        if accuracy_score(y.flatten(), pred.flatten()) > 0 and confidence_score(pred) >= 0.2:
            break

    return parametres_CNN, parametres_DNN


def compute_metrics(X, y, indices, parametres_CNN, parametres_DNN,
                    tuple_size_activation, dimensions_CNN, C_CNN, C_DNN):
    loss = 0
    dx_l = 0
    accu = 0
    conf = 0
    for idx in indices:
        pred = activation(X[idx], parametres_CNN, parametres_DNN,
                          tuple_size_activation, dimensions_CNN, C_CNN, C_DNN)
        loss += log_loss(y[idx], pred)
        dx_l += dx_log_loss(y[idx], pred)
        accu += accuracy_score(y[idx].flatten(), pred.flatten())
        conf += confidence_score(pred)

    n = len(indices)
    return loss / n, dx_l / n, accu / n, conf / n



def plot_metrics(train_loss, test_loss, train_lear, test_lear,
                 train_accu, test_accu, train_conf, test_conf):
    fig, axs = plt.subplots(1, 4, figsize=(16, 4), sharex=True)

    axs[0].plot(train_loss, label="Train")
    axs[0].plot(test_loss, label="Test")
    axs[0].set_title("Fonction de coût")
    axs[0].legend()

    axs[1].plot(train_lear, label="Train")
    axs[1].plot(test_lear, label="Test")
    axs[1].set_title("Dérivée coût")
    axs[1].legend()

    axs[2].plot(train_accu, label="Train")
    axs[2].plot(test_accu, label="Test")
    axs[2].set_title("Accuracy")
    axs[2].legend()

    axs[3].plot(train_conf, label="Train")
    axs[3].plot(test_conf, label="Test")
    axs[3].set_title("Confidence")
    axs[3].legend()

    plt.tight_layout()
    plt.show()



def convolution_neuron_network(
        X_train, y_train, X_test, y_test,
        nb_iteration,
        parametres_CNN, parametres_grad, parametres_DNN,
        dimensions_CNN,
        tuple_size_activation,
        learning_rate_CNN, beta1, beta2, learning_rate_DNN
    ):

    C_CNN = len(dimensions_CNN)
    C_DNN = len(parametres_DNN) // 2

    show_information(tuple_size_activation, dimensions_CNN)

    # Suivi des métriques
    train_loss, train_accu, train_lear, train_conf = [], [], [], []
    test_loss, test_accu, test_lear, test_conf = [], [], [], []

    best_accu = 0
    best_model = {"CNN": None, "DNN": None}

    k = 0
    for epoch in range(nb_iteration):
        for j in tqdm(range(X_train.shape[0]), desc=f"Époque {epoch + 1}/{nb_iteration}"):
            
            parametres_CNN, parametres_DNN = train_one_sample(
                X_train[j], y_train[j], parametres_CNN, parametres_DNN, parametres_grad,
                dimensions_CNN, tuple_size_activation, C_CNN, C_DNN,
                learning_rate_CNN, learning_rate_DNN, beta1, beta2
            )

            k += 1
            if (k % 100 == 0):
                # Évaluation partielle
                rand_idx_train = np.random.choice(X_train.shape[0], 50, replace=False)
                rand_idx_test = np.random.choice(X_test.shape[0], 50, replace=False)

                tl, tdx, ta, tc = compute_metrics(X_train, y_train, rand_idx_train,
                                                parametres_CNN, parametres_DNN,
                                                tuple_size_activation, dimensions_CNN, C_CNN, C_DNN)

                vl, vdx, va, vc = compute_metrics(X_test, y_test, rand_idx_test,
                                                parametres_CNN, parametres_DNN,
                                                tuple_size_activation, dimensions_CNN, C_CNN, C_DNN)

                train_loss.append(tl)
                train_lear.append(tdx)
                train_accu.append(ta)
                train_conf.append(tc)

                test_loss.append(vl)
                test_lear.append(vdx)
                test_accu.append(va)
                test_conf.append(vc)

                if va > best_accu:
                    best_accu = va
                    print(f"New accuracy: {train_accu[-1]}")
                    best_model["CNN"] = deepcopy(parametres_CNN)
                    best_model["DNN"] = deepcopy(parametres_DNN)

    # Résultats finaux
    print(f"\n🧠 Accuracy finale - Train : {train_accu[-1]:.5f}")
    print(f"🧪 Accuracy finale - Test  : {test_accu[-1]:.5f}")
    print(f"🔎 Confidence score - Test : {test_conf[-1]:.5f}")

    plot_metrics(train_loss, test_loss, train_lear, test_lear, train_accu, test_accu, train_conf, test_conf)

    return best_model["CNN"], best_model["DNN"], test_accu[-1], test_conf[-1]
