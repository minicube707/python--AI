
import  numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import deepcopy
import time

from .Evaluation_Metric import activation, log_loss, accuracy_score, dx_log_loss, confidence_score
from .Preprocessing import handle_key
from .Deep_Neuron_Network import foward_propagation_DNN, back_propagation_DNN, update_DNN

def train_one_sample(X, y, parametres_DNN, dimensions_DNN, C_DNN, learning_rate_DNN, alpha,
                     max_attempts, min_confidence_score):

    for _ in range(max_attempts):

        # Forward
        activations = foward_propagation_DNN(X, parametres_DNN, dimensions_DNN, C_DNN, alpha)
        gradients_DNN = back_propagation_DNN(activations, parametres_DNN, dimensions_DNN, y,  C_DNN, alpha)
        parametres_DNN = update_DNN(gradients_DNN, parametres_DNN, learning_rate_DNN, dimensions_DNN)
        pred = activation(X, parametres_DNN, dimensions_DNN, C_DNN, alpha)

        if confidence_score(y, pred) >= min_confidence_score:
            break

    return  parametres_DNN


def compute_metrics(X, y, indices, parametres_DNN, dimensions_DNN, C_DNN, alpha):

    loss = 0
    dx_l = 0
    accu = 0
    conf = 0

    for idx in indices:
        pred = activation(X[idx], parametres_DNN, dimensions_DNN, C_DNN, alpha)

        loss += log_loss(y[idx], pred)
        dx_l += dx_log_loss(y[idx], pred)
        accu += accuracy_score(y[idx], pred)
        conf += confidence_score(y[idx], pred)

    n = len(indices)
    return loss / n, dx_l / n, accu / n, conf / n


def eval_model(X_train, y_train, X_test, y_test ,parametres_DNN, dimension_DNN,
        alpha, validation_size, best_accu,
        train_loss, test_loss, train_lear, test_lear, train_accu, test_accu, train_conf, test_conf, C_DNN):

    rand_idx_train = np.random.choice(X_train.shape[0], validation_size, replace=False)
    rand_idx_test = np.random.choice(X_test.shape[0], validation_size, replace=False)

    tl, tdx, ta, tc = compute_metrics(X_train, y_train, rand_idx_train, parametres_DNN, dimension_DNN, C_DNN, alpha)
    vl, vdx, va, vc = compute_metrics(X_test, y_test, rand_idx_test, parametres_DNN, dimension_DNN, C_DNN, alpha)

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
        print(f"\nNew accuracy: {va}")
        print(f"New confidence score: {vc}")
        print(f"New loss: {vl}")
        print("")

    return train_loss, test_loss, train_lear, test_lear, train_accu, test_accu, train_conf, test_conf, best_accu


def smooth_curve(values, window=10):
    """Calcule une moyenne glissante"""
    values = np.array(values)
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window)/window, mode='valid')


def plot_metrics(train_loss, test_loss, train_lear, test_lear,
                 train_accu, test_accu, train_conf, test_conf):
    
    fig, axs = plt.subplots(1, 4, figsize=(16, 4), sharex=True)
    fig.canvas.mpl_connect('key_press_event', handle_key)  # Raccourci clavier actif

    window = 4  # Taille de la fenêtre pour le lissage

    # Données à tracer : (titre, train_data, test_data, ylim)
    metrics = [
        ("Fonction de coût", train_loss, test_loss, None),
        ("Dérivée coût", train_lear, test_lear, None),
        ("Accuracy", train_accu, test_accu, (0, 1)),
        ("Confidence", train_conf, test_conf, (0, 1))
    ]

    def plot_with_trend(ax, train, test, title, ylim=None):
        # Données brutes
        ax.plot(train, label="Train", alpha=0.5)
        ax.plot(test, label="Test", alpha=0.5)

        # Lissage
        sm_train = smooth_curve(train, window)
        sm_test = smooth_curve(test, window)

        # Centrage
        offset_train = (len(train) - len(sm_train)) // 2
        offset_test = (len(test) - len(sm_test)) // 2

        # Courbes lissées
        ax.plot(range(offset_train, offset_train + len(sm_train)), sm_train, label="Trend Train", color='fuchsia', linewidth=2)
        ax.plot(range(offset_test, offset_test + len(sm_test)), sm_test, label="Trend Test", color='lime', linewidth=2)

        ax.set_title(title)
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend()

    # Tracer les 4 métriques
    for i, (title, train_data, test_data, ylim) in enumerate(metrics):
        plot_with_trend(axs[i], train_data, test_data, title, ylim)

    plt.tight_layout()
    plt.show(block=False)

def deep_neuron_network_sequential_trainning(X_train, y_train, X_test, y_test,
        nb_iteration, parametres_DNN, dimension_DNN,
        alpha, learning_rate_DNN,
        max_attempts, min_confidence_score, validation_size, validation_frequency):

    C_DNN = len(dimension_DNN) 

    # Suivi des métriques
    train_loss, train_accu, train_lear, train_conf = [], [], [], []
    test_loss, test_accu, test_lear, test_conf = [], [], [], []

    best_accu = 0

    train_loss, test_loss, train_lear, test_lear, train_accu, test_accu, train_conf, test_conf, best_accu = eval_model(
        X_train, y_train, X_test, y_test , parametres_DNN, dimension_DNN,
        alpha, validation_size, best_accu,
        train_loss, test_loss, train_lear, test_lear, train_accu, test_accu, train_conf, test_conf, C_DNN)
    
    # Démarrer le chronomètre
    start_time = time.time()
    
    k = 0
    for _ in  tqdm(range(nb_iteration)):

        parametres_DNN = train_one_sample( X_train, y_train, parametres_DNN, dimension_DNN, C_DNN, learning_rate_DNN, 
                alpha, max_attempts, min_confidence_score)
    
        k += 1
        if (k % validation_frequency == 0):

            # Évaluation partielle
            train_loss, test_loss, train_lear, test_lear, train_accu, test_accu, train_conf, test_conf, best_accu = eval_model(
            X_train, y_train, X_test, y_test, parametres_DNN, dimension_DNN,
            alpha, validation_size, best_accu,
            train_loss, test_loss, train_lear, test_lear, train_accu, test_accu, train_conf, test_conf, C_DNN)

    # Arrêter le chronomètre
    end_time = time.time()

    # Calcul du temps en minutes
    elapsed_time_minutes = (end_time - start_time) / 60

    return parametres_DNN, train_loss, test_loss, train_lear, test_lear, train_accu, test_accu, train_conf, test_conf, elapsed_time_minutes


def deep_neuron_network(
        X_train, y_train, X_test, y_test,
        nb_iteration,parametres_DNN, dimension_DNN,
        alpha, learning_rate_DNN,
        max_attempts, min_confidence_score, validation_size, validation_frequency
    ):

    parametres_DNN, train_loss, test_loss, train_lear, test_lear, train_accu, test_accu, train_conf, test_conf, elapsed_time_minutes = deep_neuron_network_sequential_trainning(
    X_train, y_train, X_test, y_test,
    nb_iteration,parametres_DNN, dimension_DNN,
    alpha, learning_rate_DNN,
    max_attempts, min_confidence_score, validation_size, validation_frequency)


    # Résultats finaux
    print(f"\n🚂💰 Coût final - Train          : {train_loss[-1]:.5f}")
    print(f"🧪💰 Coût final - Test             : {test_loss[-1]:.5f}")
    print(f"🧠📉 Derive Coût final - Train 🚆  : {train_lear[-1]:.5f}") 
    print(f"🧠📉 Derive Coût final - Test 🧪   : {test_lear[-1]:.5f}")
    print(f"🧠 Accuracy finale - Train          : {train_accu[-1]:.5f}")
    print(f"🧪 Accuracy finale - Test           : {test_accu[-1]:.5f}")
    print(f"🔎 Confidence score - Test          : {test_conf[-1]:.5f}")

    print("\nIndicateur underfiting/overfiting")
    print(f"🧠📉 Derive Coût final - Train 🚆   : {train_lear[-1]:.5f}") 
    print(f"🧠📉 Derive Coût final - Test 🧪    : {test_lear[-1]:.5f}")
    print("Accuracy Ratio                         :", test_accu[-1] / train_accu[-1])
    print("Indicateur d’overfitting               :", test_loss[-1] - train_loss[-1])

    print(f"\nTemps d'entrenemant {elapsed_time_minutes} minutes, {elapsed_time_minutes/60} heures")
    print("")

    plot_metrics(train_loss, test_loss, train_lear, test_lear, train_accu, test_accu, train_conf, test_conf)

    return deepcopy(parametres_DNN), test_accu[-1], test_conf[-1], test_loss[-1], elapsed_time_minutes