
import numpy as np
import matplotlib.pyplot as plt
import time
import os

from tqdm import tqdm
from PIL import Image

from .Evaluation_Metric import accuracy_score, confidence_score
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

    plt.pause(0.01)

def compute_metrics_full_data(model, X, y, indices, batch_size, dict_performance):

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
        batch_len = len(y_batch)

        total_loss += model.loss_metric.forward(pred_batch, y_batch) * batch_len
        total_dx += np.mean(model.loss_metric.backward()) * batch_len

        total_acc += accuracy_score(y_batch, pred_batch) * batch_len
        total_conf += confidence_score(y_batch, pred_batch) * batch_len

    total_loss /= n_samples
    total_dx /= n_samples
    total_acc /= n_samples
    total_conf /= n_samples

    dict_performance["loss"].append(total_loss)
    dict_performance["lear"].append(total_dx)
    dict_performance["accu"].append(total_acc)
    dict_performance["conf"].append(total_conf)

def training_full_data(model, X_train, y_train, X_test, y_test, hyperparams, dataset):

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

    compute_metrics_full_data(model, X_train, y_train, rand_idx_train, batch_size, data_train)
    compute_metrics_full_data(model, X_test, y_test, rand_idx_test, batch_size, data_test)

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

            global_step += 1
            if (global_step % validation_frequency == 0):
                # Évaluation partielle
                rand_idx_train = np.random.choice(X_train.shape[0], validation_size, replace=False)
                rand_idx_test = np.random.choice(X_test.shape[0], validation_size, replace=False)

                compute_metrics_full_data(model, X_train, y_train, rand_idx_train, batch_size, data_train)
                compute_metrics_full_data(model, X_test, y_test, rand_idx_test, batch_size, data_test)

                update_graph(lines, axs, data_train, data_test)

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

    compute_metrics_full_data(model, X_train, y_train, rand_idx_train, batch_size, data_train)
    compute_metrics_full_data(model, X_test, y_test, rand_idx_test, batch_size, data_test)

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
    print(f"\n🚂💰 Coût final - Train          : {data_train['loss'][-1]:.5f}")
    print(f"🧪💰 Coût final - Test             : {data_test['loss'][-1]:.5f}")

    print(f"🧠 Accuracy finale - Train          : {data_train['accu'][-1]:.5f}")
    print(f"🧪 Accuracy finale - Test           : {data_test['accu'][-1]:.5f}")

    print(f"🔎 Confidence score - Train         : {data_train['conf'][-1]:.5f}")
    print(f"🔎 Confidence score - Test          : {data_test['conf'][-1]:.5f}")

    print("\nIndicateur underfiting/overfiting")
    print(f"🧠📉 Derive Coût final - Train 🚆   : {data_train['lear'][-1]:.5f}") 
    print(f"🧠📉 Derive Coût final - Test 🧪    : {data_test['lear'][-1]:.5f}")
    print("Accuracy Ratio                         :", data_test['accu'][-1] / data_train['accu'][-1])
    print("Indicateur d’overfitting               :", data_test['loss'][-1] - data_train['loss'][-1])

    print(f"\nTemps d'entrenemant {elapsed_time_minutes} minutes, {elapsed_time_minutes/60} heures")
    print("")

    update_graph(lines, axs, data_train, data_test)

    return data_test, elapsed_time_minutes


def sample_files(file_paths, labels, sample_size):

    if (sample_size > len(file_paths)):
        sample_size = len(file_paths)

    indices = np.random.choice(len(file_paths), size=sample_size, replace=False)
    
    sampled_files = [file_paths[i] for i in indices]
    sampled_labels = [labels[i] for i in indices]
    
    return sampled_files, sampled_labels

def batch_generator(file_paths, labels, batch_size, img_size, shuffle, picture_in_RGB):
    n = len(file_paths)
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)
        
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_idx = indices[start:end]
        X_batch = []
        y_batch = []
        for i in batch_idx:

            # Lecture image
            if (picture_in_RGB):
                img = Image.open(file_paths[i]).convert('RGB')  # 'L' pour grayscale, 'RGB' si couleur
            else:
                img = Image.open(file_paths[i]).convert('L')

            img = img.resize(img_size)
            img_array = np.array(img) / 255.0  # normalisation

            # ajouter canal si grayscale
            if img_array.ndim == 2:
                img_array = img_array[None, :, :]

            X_batch.append(img_array)
            y_batch.append(labels[i])

        yield np.array(X_batch), np.array(y_batch)

def compute_metrics_batch_data(model, file_paths, labels, batch_size, dict_performance, img_size, picture_in_RGB):

    total_loss = 0.0
    total_dx = 0.0
    total_acc = 0.0
    total_conf = 0.0
    n_samples = len(file_paths)

    gen = batch_generator(file_paths, labels, batch_size, img_size, False, picture_in_RGB)

    for X_batch, y_batch in gen:
        
        X_batch = X_batch.transpose(0, 3, 1, 2)
        pred_batch = model.forward_propagation(X_batch, training=False)
        batch_len = len(y_batch)

        total_loss += model.loss_metric.forward(pred_batch, y_batch) * batch_len
        total_dx += np.mean(model.loss_metric.backward()) * batch_len
 
        total_acc += accuracy_score(y_batch, pred_batch) * batch_len
        total_conf += confidence_score(y_batch, pred_batch) * batch_len

    total_loss /= n_samples
    total_dx /= n_samples
    total_acc /= n_samples
    total_conf /= n_samples

    dict_performance["loss"].append(total_loss)
    dict_performance["lear"].append(total_dx)
    dict_performance["accu"].append(total_acc)
    dict_performance["conf"].append(total_conf)

def training_batch_data(model, hyperparams, dataset, train_files, train_labels, test_files, test_labels, picture_in_RGB):

    nb_epoch = hyperparams.nb_epoch
    batch_size = hyperparams.batch_size
    img_size = (hyperparams.input_shape[1], hyperparams.input_shape[2])

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

    train_sample_files, train_sample_labels = sample_files(train_files, train_labels, validation_size)
    test_sample_files, test_sample_labels = sample_files(test_files, test_labels, validation_size)

    compute_metrics_batch_data(model, train_sample_files, train_sample_labels, batch_size, data_train, img_size, picture_in_RGB)
    compute_metrics_batch_data(model, test_sample_files, test_sample_labels, batch_size, data_test, img_size, picture_in_RGB)

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

        train_gen = batch_generator(train_files, train_labels, batch_size, img_size, True, picture_in_RGB)
        steps_per_epoch = np.ceil(len(train_files) / batch_size)

        for X_batch, y_batch in tqdm(train_gen, total=steps_per_epoch, desc=f"Epoch {epoch+1}/{nb_epoch}"):
            
            X_batch = X_batch.transpose(0, 3, 1, 2)

            model.forward_propagation(X_batch, training=True)
            model.backward_propagation(y_batch)
            model.update()

            global_step += 1
            if (global_step % validation_frequency == 0):
                
                # Évaluation partielle
                train_sample_files, train_sample_labels = sample_files(train_files, train_labels, validation_size)
                test_sample_files, test_sample_labels = sample_files(test_files, test_labels, validation_size)

                compute_metrics_batch_data(model, train_sample_files, train_sample_labels, batch_size, data_train, img_size, picture_in_RGB)
                compute_metrics_batch_data(model, test_sample_files, test_sample_labels, batch_size, data_test, img_size, picture_in_RGB)
                
                update_graph(lines, axs, data_train, data_test)
                
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
    train_sample_files, train_sample_labels = sample_files(train_files, train_labels, validation_size)
    test_sample_files, test_sample_labels = sample_files(test_files, test_labels, validation_size)

    compute_metrics_batch_data(model, train_sample_files, train_sample_labels, batch_size, data_train, img_size, picture_in_RGB)
    compute_metrics_batch_data(model, test_sample_files, test_sample_labels, batch_size, data_test, img_size, picture_in_RGB)

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
    print(f"\n🚂💰 Coût final - Train          : {data_train['loss'][-1]:.5f}")
    print(f"🧪💰 Coût final - Test             : {data_test['loss'][-1]:.5f}")

    print(f"🧠 Accuracy finale - Train          : {data_train['accu'][-1]:.5f}")
    print(f"🧪 Accuracy finale - Test           : {data_test['accu'][-1]:.5f}")

    print(f"🔎 Confidence score - Train         : {data_train['conf'][-1]:.5f}")
    print(f"🔎 Confidence score - Test          : {data_test['conf'][-1]:.5f}")

    print("\nIndicateur underfiting/overfiting")
    print(f"🧠📉 Derive Coût final - Train 🚆   : {data_train['lear'][-1]:.5f}") 
    print(f"🧠📉 Derive Coût final - Test 🧪    : {data_test['lear'][-1]:.5f}")
    print("Accuracy Ratio                         :", data_test['accu'][-1] / data_train['accu'][-1])
    print("Indicateur d’overfitting               :", data_test['loss'][-1] - data_train['loss'][-1])

    print(f"\nTemps d'entrenemant {elapsed_time_minutes} minutes, {elapsed_time_minutes/60} heures")
    print("")

    update_graph(lines, axs, data_train, data_test)

    return data_test, elapsed_time_minutes