
import  numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import deepcopy
import time

from .Evaluation_Metric import log_loss, accuracy_score, dx_log_loss, confidence_score
from .Preprocessing import handle_key

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

def compute_metrics(model, X, y, indices, batch_size=32):

    total_loss = 0.0
    total_dx = 0.0
    total_acc = 0.0
    total_conf = 0.0
    n_samples = len(indices)

    for i in range(0, n_samples, batch_size):
        batch_idx = indices[i:i + batch_size]
        X_batch = X[batch_idx]
        y_batch = y[batch_idx]

        if X_batch.ndim == 3:
            X_batch = X_batch[:, None, :, :]

        # forward batch
        pred_batch = model.forward_propagation(X_batch, training=False)

        total_loss += log_loss(pred_batch, y_batch) * len(batch_idx)
        total_dx += dx_log_loss(pred_batch, y_batch) * len(batch_idx)
        total_acc += accuracy_score(y_batch, pred_batch) * len(batch_idx)
        total_conf += confidence_score(y_batch, pred_batch) * len(batch_idx)

    total_loss /= n_samples
    total_dx /= n_samples
    total_acc /= n_samples
    total_conf /= n_samples

    return total_loss, total_dx, total_acc, total_conf

def trainnig(model, 
             X_train, y_train, X_test, y_test, batch_size,
             nb_iteration, validation_size, validation_frequency):

# Suivi des métriques
    train_loss, train_accu, train_lear, train_conf = [], [], [], []
    test_loss, test_accu, test_lear, test_conf = [], [], [], []

    rand_idx_train = np.random.choice(X_train.shape[0], validation_size, replace=False)
    rand_idx_test = np.random.choice(X_test.shape[0], validation_size, replace=False)

    tl, tdx, ta, tc = compute_metrics(model, X_train, y_train, rand_idx_train, batch_size)
    vl, vdx, va, vc = compute_metrics(model, X_test, y_test, rand_idx_test, batch_size)

    train_loss.append(tl)
    train_lear.append(tdx)
    train_accu.append(ta)
    train_conf.append(tc)

    test_loss.append(vl)
    test_lear.append(vdx)
    test_accu.append(va)
    test_conf.append(vc)
    
    best_accu = va
    print(f"\nInitial accurracy: {best_accu}")
    print(f"Initial confidence score: {vc}")
    print(f"Initial loss: {vl}")
    print("")

    # Démarrer le chronomètre
    start_time = time.time()
    global_step = 0

    for epoch in range(nb_iteration):
        for j in tqdm(range(0, X_train.shape[0], batch_size), desc=f"Époque {epoch + 1}/{nb_iteration}"):
            
            X_batch = X_train[j:j+batch_size]
            y_batch = y_train[j:j+batch_size]

            if X_batch.ndim == 3:
                X_batch = X_batch[:, None, :, :]

            model.forward_propagation(X_batch, True)
            model.backward_propagation(y_batch)
            model.update()

            global_step += 1
            if (global_step % validation_frequency == 0):
                # Évaluation partielle
                rand_idx_train = np.random.choice(X_train.shape[0], validation_size, replace=False)
                rand_idx_test = np.random.choice(X_test.shape[0], validation_size, replace=False)

                tl, tdx, ta, tc = compute_metrics(model, X_train, y_train, rand_idx_train, batch_size)
                vl, vdx, va, vc = compute_metrics(model, X_test, y_test, rand_idx_test, batch_size)

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

    # Arrêter le chronomètre
    end_time = time.time()

    # Évaluation partielle
    rand_idx_train = np.random.choice(X_train.shape[0], validation_size, replace=False)
    rand_idx_test = np.random.choice(X_test.shape[0], validation_size, replace=False)


    tl, tdx, ta, tc = compute_metrics(model, X_train, y_train, rand_idx_train, batch_size)
    vl, vdx, va, vc = compute_metrics(model, X_test, y_test, rand_idx_test, batch_size)


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

    # Calcul du temps en minutes
    elapsed_time_minutes = (end_time - start_time) / 60
        
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
