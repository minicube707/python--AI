import pandas as pd
import os

module_dir = os.path.dirname(__file__)
os.chdir(module_dir)

def fill_information(name_model, date,
                        nb_epoch, test_accu, 
                        learning_rate,
                        training_size, test_size, 
                        baseline_mode, nb_fine_tunning):

    new_log = {}
    
    new_log = {
    "name": name_model,
    "date": date, 

    "nb_epoch": nb_epoch,

    "accuracy": test_accu,
    
    "learning_rate_DNN": learning_rate,

    "Size_training_set": training_size,
    "Size_test_set": test_size,

    "Based_model": baseline_mode,
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
    print(df[["name", "date", "accuracy", "Number_fine_tunning"]].rename(lambda x: x + 1))
