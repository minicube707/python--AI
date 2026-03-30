
import  numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

from .Evaluation_Metric import log_loss, accuracy_score, dx_log_loss, confidence_score
from .Preprocessing import handle_key

def smooth_curve(values, window=10):
    """Calcule une moyenne glissante"""
    values = np.array(values)
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window)/window, mode='valid')

def init_animation():
    fig, axs = plt.subplots(1, 4, figsize=(16, 4), sharex=True)
    fig.canvas.mpl_connect('key_press_event', handle_key)  # Raccourci clavier actif

    metrics = [
        ("Fonction de coût", None),
        ("Dérivée coût", None),
        ("Accuracy", (0, 1)),
        ("Confidence", (0, 1))
    ]

    lines = []

    for ax, (title, ylim) in zip(axs, metrics):
        line_train, = ax.plot([], [], label="Train", alpha=0.5)
        line_test, = ax.plot([], [], label="Test", alpha=0.5)
        line_trend_train, = ax.plot([], [], label="Trend Train", color='fuchsia', linewidth=2)
        line_trend_test, = ax.plot([], [],  label="Trend Test", color='lime', linewidth=2)

        ax.set_title(title)
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend()

        lines.append((line_train, line_test, line_trend_train, line_trend_test))

    plt.tight_layout()
    plt.show(block=False)

    return fig, axs, lines

def update_graph(lines, axs, data_train, data_test, window=4):

    metrics_data = [
        data_train["loss"], data_test["loss"],
        data_train["lear"], data_test["lear"],
        data_train["accu"], data_test["accu"],
        data_train["conf"], data_test["conf"]
    ]

    for i in range(4):
        train = metrics_data[i * 2]
        test = metrics_data[i * 2 + 1]

        l_train, l_test, l_trend_train, l_trend_test = lines[i]

        x = np.arange(len(train))

        # Courbes principales
        l_train.set_data(x, train)
        l_test.set_data(x, test)

        # Lissage
        sm_train = smooth_curve(train, window)
        sm_test = smooth_curve(test, window)

        if len(sm_train) > 0:
            offset = (len(train) - len(sm_train)) // 2
            l_trend_train.set_data(range(offset, offset + len(sm_train)), sm_train)

        if len(sm_test) > 0:
            offset = (len(test) - len(sm_test)) // 2
            l_trend_test.set_data(range(offset, offset + len(sm_test)), sm_test)

        axs[i].relim()
        axs[i].autoscale_view()

    plt.pause(0.05)

def compute_metrics(model, X, y, indices, batch_size, dict_performance):

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

    dict_performance["loss"].append(total_loss)
    dict_performance["lear"].append(total_dx)
    dict_performance["accu"].append(total_acc)
    dict_performance["conf"].append(total_conf)

def trainnig(model, X_train, y_train, X_test, y_test, hyperparams, dataset):

    nb_epoch = hyperparams.nb_epoch
    batch_size = hyperparams.batch_size

    validation_size = dataset.validation_size
    validation_frequency = dataset.validation_frequency
    

    # Suivi des métriques

    data_train = {
    "loss": [],
    "lear": [],
    "accu": [],
    "conf": [],
    }

    data_test = {
    "loss": [],
    "lear": [],
    "accu": [],
    "conf": [],
    }

    rand_idx_train = np.random.choice(X_train.shape[0], validation_size, replace=False)
    rand_idx_test = np.random.choice(X_test.shape[0], validation_size, replace=False)

    compute_metrics(model, X_train, y_train, rand_idx_train, batch_size, data_train)
    compute_metrics(model, X_test, y_test, rand_idx_test, batch_size, data_test)

    va = data_test["accu"][-1]
    vc = data_test["conf"][-1]
    vl = data_test["loss"][-1]
    
    best_accu = va
    print(f"\nInitial accurracy: {best_accu}")
    print(f"Initial confidence score: {vc}")
    print(f"Initial loss: {vl}")
    print("")
    
    fig, axs, lines = init_animation()
    # Démarrer le chronomètre
    start_time = time.time()
    global_step = 0

    for epoch in range(nb_epoch):
        for j in tqdm(range(0, X_train.shape[0], batch_size), desc=f"Époque {epoch + 1}/{nb_epoch}"):
            
            X_batch = X_train[j:j+batch_size]
            y_batch = y_train[j:j+batch_size]

            if X_batch.ndim == 3:
                X_batch = X_batch[:, None, :, :]

            model.forward_propagation(X_batch, True)
            model.backward_propagation(y_batch)
            model.update()

            update_graph(lines, axs, data_train, data_test)
            
            global_step += 1
            if (global_step % validation_frequency == 0):
                # Évaluation partielle
                rand_idx_train = np.random.choice(X_train.shape[0], validation_size, replace=False)
                rand_idx_test = np.random.choice(X_test.shape[0], validation_size, replace=False)

                compute_metrics(model, X_train, y_train, rand_idx_train, batch_size, data_train)
                compute_metrics(model, X_test, y_test, rand_idx_test, batch_size, data_test)

                va = data_test["accu"][-1]
                vc = data_test["conf"][-1]
                vl = data_test["loss"][-1]

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


    compute_metrics(model, X_train, y_train, rand_idx_train, batch_size, data_train)
    compute_metrics(model, X_test, y_test, rand_idx_test, batch_size, data_test)

    va = data_test["accu"][-1]
    vc = data_test["conf"][-1]
    vl = data_test["loss"][-1]

    if va > best_accu:
        best_accu = va
        print(f"\nNew accuracy: {va}")
        print(f"New confidence score: {vc}")
        print(f"New loss: {vl}")
        print("")

    # Calcul du temps en minutes
    elapsed_time_minutes = (end_time - start_time) / 60
        
    # Résultats finaux
    print(f"\n🚂💰 Coût final - Train          : {data_train["loss"][-1]:.5f}")
    print(f"🧪💰 Coût final - Test             : {data_test["loss"][-1]:.5f}")

    print(f"🧠 Accuracy finale - Train          : {data_train["accu"][-1]:.5f}")
    print(f"🧪 Accuracy finale - Test           : {data_test["accu"][-1]:.5f}")

    print(f"🔎 Confidence score - Train         : {data_train["conf"][-1]:.5f}")
    print(f"🔎 Confidence score - Test          : {data_test["conf"][-1]:.5f}")

    print("\nIndicateur underfiting/overfiting")
    print(f"🧠📉 Derive Coût final - Train 🚆   : {data_train["lear"][-1]:.5f}") 
    print(f"🧠📉 Derive Coût final - Test 🧪    : {data_test["lear"][-1]:.5f}")
    print("Accuracy Ratio                         :", data_test["accu"][-1] / data_train["accu"][-1])
    print("Indicateur d’overfitting               :", data_test["loss"][-1] - data_train["loss"][-1])

    print(f"\nTemps d'entrenemant {elapsed_time_minutes} minutes, {elapsed_time_minutes/60} heures")
    print("")

    update_graph(lines, axs, data_train, data_test)

    return data_test, elapsed_time_minutes