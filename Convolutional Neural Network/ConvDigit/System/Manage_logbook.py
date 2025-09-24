import pandas as pd
import os

module_dir = os.path.dirname(__file__)
os.chdir(module_dir)

def fill_information(name, date, 
                    nb_epoch, max_attempts, min_confidence_score,
                    training_time,
                    accuracy, confidence_score,
                    kernel_size, nb_kernel, 
                    lr_CNN, lr_DNN, beta1, beta2, 
                    training_size, test_size,
                    model_fine_tunning, nb_fine_tunning):

    new_log = {}
    
    new_log = {
    "name": name,
    "date": date, 
    "nb_epoch": nb_epoch,
    "max_attempts": max_attempts,
    "min_confidence_score": min_confidence_score,
    "training_time_(min)": training_time,
    "accuracy": accuracy,
    "confidence_score": confidence_score,
    "kernel_size": kernel_size,
    "number_kernel": nb_kernel,
    "learning_rate_CNN": lr_CNN,
    "learning_rate_DNN": lr_DNN,
    "beta1": beta1,
    "beta2": beta2,
    "Size_training_set": training_size,
    "Size_test_set": test_size,
    "Based_model": model_fine_tunning,
    "Number_fine_tunning": nb_fine_tunning, 
    }

    return (new_log)


def add_model(new_log, path, csv_file):

    filename= path + "/" + csv_file

    try:
        df = pd.read_csv(filename, sep=';')
        df = pd.concat([df, pd.DataFrame([new_log])], ignore_index=True)

    except FileNotFoundError:

        df = pd.DataFrame([new_log])
    
    df.to_csv(filename, sep=';', index=False)
    print("Update LogBook")


def show_info_all(path, csv_file):

    # Charger le fichier CSV
    df = pd.read_csv(path + "/" + csv_file, sep=';')

    # Afficher toutes les colonnes sans troncature
    pd.set_option('display.max_columns', None)

    # Afficher toutes les valeurs
    print(df)

def show_info_main(path, csv_file):

    # Charger le fichier CSV
    df = pd.read_csv(path + "/" + csv_file, sep=';')

    # Afficher toutes les valeurs  
    print(df[["name", "date", "training_time_(min)", "accuracy", "confidence_score", "Number_fine_tunning"]].rename(lambda x: x + 1))
