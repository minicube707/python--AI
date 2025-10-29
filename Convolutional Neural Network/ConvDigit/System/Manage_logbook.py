import pandas as pd

import os

def fill_information(name, date, training_time,
                    nb_epoch,  max_attempts, min_confidence_score, beta1, beta2, alpha,
                    cost_loss, accuracy, confidence_score, 
                    lr_CNN, kernel_size, nb_kernel, activation_function_CNN,
                    lr_DNN, nb_neurons, activation_function_DNN,
                    training_size, test_size, 
                    model_fine_tunning, nb_fine_tunning, validation_size,
                    validation_frequency, ratio_test):

    new_log = {}
    
    new_log = {
    "name": name,
    "date": date, 
    "training_time_(min)": training_time,

    "nb_epoch": nb_epoch,
    "max_attempts": max_attempts,
    "min_confidence_score": min_confidence_score,
    "beta1": beta1,
    "beta2": beta2,
    "alpha": alpha,

    "cost_loss": cost_loss,
    "accuracy": accuracy,
    "confidence_score": confidence_score,

    "learning_rate_CNN": lr_CNN,
    "kernel_size": kernel_size,
    "kernel_number": nb_kernel,
    "activation_function_CNN": activation_function_CNN,
    
    "learning_rate_DNN": lr_DNN,
    "neurons_number": nb_neurons,
    "activation_function_DNN": activation_function_DNN,

    "Size_training_set": training_size,
    "Size_test_set": test_size,
    "Ratio_train/test":ratio_test,
    "Size_validation_set": validation_size,
    "Validation_frequency":validation_frequency,

    "Based_model": model_fine_tunning,
    "Number_fine_tunning": nb_fine_tunning, 
    }

    return (new_log)


def add_model(new_log, path, csv_file):

    # Chemin complet du fichier CSV
    filename = os.path.join(path, csv_file)
    
    # Créer le dossier si besoin
    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"[INFO] Dossier créé : {os.path.abspath(folder)}")

    try:
        df = pd.read_csv(filename, sep=';')
        df = pd.concat([df, pd.DataFrame([new_log])], ignore_index=True)

    except FileNotFoundError:

        df = pd.DataFrame([new_log])
    
    df.to_csv(filename, sep=';', index=False)
    print("Update LogBook")


def show_all_info_model(model_dict):
    for key, value in model_dict.items():
        print(f"{key}: {value}")

def show_info_main(path, csv_file):

    # Charger le fichier CSV
    df = pd.read_csv(path + "/" + csv_file, sep=';')

    # Afficher toutes les valeurs  
    print(df[["name", "date", "training_time_(min)", "accuracy", "confidence_score", "Number_fine_tunning"]].rename(lambda x: x + 1))
