import os
import numpy as np
import pandas as pd

from .Manage_logbook import show_info_main

from .Evaluation_Metric import CrossEntropyLoss, BinaryCrossEntropy
from .Mathematical_function import Softmax, Sigmoide
from .Layers import Flatten, GlobalAveragePooling

from .FullModel import FullModel
from .Optimizer import Adam

from .Dataclasses import Hyperparams, Dataset

from .Constante import FOLDER_NAME_LOGBOOK, FOLDER_NAME_MODEL

def load_model(path, model_name, new_hyperparams):

    params = load_model_parameters(path, model_name + ".npz")
    df = load_model_hyperparameters(path, model_name + ".json")
    data = df.to_dict(orient="records")

    log = data[0]

    hyperparams = Hyperparams(**log.get("hyperparameters", {}))
    
    if (new_hyperparams != None):
        hyperparams.nb_epoch = new_hyperparams.nb_epoch
        hyperparams.batch_size = new_hyperparams.batch_size
        hyperparams.lr = new_hyperparams.lr
        hyperparams.beta1 = new_hyperparams.beta1
        hyperparams.beta2 = new_hyperparams.beta2
        hyperparams.alpha = new_hyperparams.alpha
        hyperparams.optimizer = new_hyperparams.optimizer
        hyperparams.contamination = new_hyperparams.contamination
        hyperparams.support = new_hyperparams.support
        
    structure = (log.get("structure", [{}]))
    performance = log.get("performance", {})
    dataset = Dataset(**log.get("dataset", {}))
    metadata = log.get("metadata", {})


    if hyperparams.loss_metric == "CrossEntropyLoss":
        loss_metric = CrossEntropyLoss
    elif hyperparams.loss_metric == "BinaryCrossEntropy":
        loss_metric = BinaryCrossEntropy
    else:
        raise Exception("Unknow loss metric: ", hyperparams.loss_metric)
    
    
    if hyperparams.output_layer == "Softmax":
        output_layer = Softmax  
    elif hyperparams.output_layer == "Sigmoide":
        output_layer = Sigmoide 
    else:
        raise Exception("Unknow output layer: ", hyperparams.output_layer)
        
        
    if hyperparams.optimizer == "Adam":
        optimizer = Adam()
    else:
        raise Exception("Unknow optimizer: ", hyperparams.optimizer)
    
    if hyperparams.transition_layer == "Flatten":
        transition_layer = Flatten()  
    elif hyperparams.transition_layer == "GlobalAveragePooling":
        transition_layer = GlobalAveragePooling() 
    else:
        raise Exception("Unknow transition layer: ", hyperparams.transition_layer)
    
    hyperparams.check_support()
    model = FullModel(hyperparams, structure, loss_metric, output_layer, optimizer, transition_layer)
    model.load(params)

    return model, hyperparams, structure, performance, dataset, metadata

def load_model_parameters(path, model_name):

    model_dir = os.path.join(path, FOLDER_NAME_MODEL)
    model_path = os.path.join(model_dir, model_name)
    
    if not os.path.exists(model_path):
        chemin_absolu = os.path.abspath(model_path)

        print("")
        print(f"[ERREUR] Fichier '{model_name}' non trouvé.")
        print(f"Chemin testé (absolu) : {chemin_absolu}\n")

        # Liste les fichiers disponibles pour aider au debug
        if os.path.exists(model_dir):
            print(f"📂 Fichiers disponibles dans le dossier {FOLDER_NAME_MODEL} :")
            for f in os.listdir(model_dir):
                print(" -", f)
        else:
            print(f"❌ Le dossier '{FOLDER_NAME_MODEL}' n'existe pas.")

        exit(1)

    npz_file = np.load(model_path, allow_pickle=True)
    params = {key: npz_file[key] for key in npz_file.keys()}

    return params

def load_model_hyperparameters(path, model_name):

    model_dir = os.path.join(path, FOLDER_NAME_LOGBOOK)
    model_path = os.path.join(model_dir, model_name)

    if not os.path.exists(model_path):
        chemin_absolu = os.path.abspath(model_path)

        print("")
        print(f"[ERREUR] Fichier '{model_name}' non trouvé.")
        print(f"Chemin testé (absolu) : {chemin_absolu}\n")

        # Liste les fichiers disponibles pour aider au debug
        if os.path.exists(model_dir):
            print(f"📂 Fichiers disponibles dans le dossier {FOLDER_NAME_LOGBOOK} :")
            for f in os.listdir(model_dir):
                print(" -", f)
        else:
            print(f"❌ Le dossier '{FOLDER_NAME_LOGBOOK}' n'existe pas.")

        exit(1)

    try:
        df = pd.read_json(model_path)
    except ValueError:  # fichier vide ou mal formé
        print(f"[ERREUR] Le fichier '{model_name}' est vide ou mal formé.")
        df = pd.DataFrame()
    
    return df


def save_model_parameters(path, model_name, model):

    model_path = os.path.join(path, FOLDER_NAME_MODEL)
    
    # Créer le dossier s'il n'existe pas
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        print(f"[INFO] Dossier '{FOLDER_NAME_MODEL}' créé à : {os.path.abspath(model_path)}")

    # Sauvegarder le modèle
    model_path = os.path.join(model_path, model_name)
    model.save(model_path)
    print(f"SUCCÈS: Modèle sauvegardé")


def file_management(date, test_accu, test_conf):
    str_accu = f"{test_accu:.5f}".replace(".", ",")
    str_conf = f"{test_conf:.5f}".replace(".", ",")
    name_model = f"({str_accu})({str_conf})({date})"

    return name_model


def transform_name(filename):
    
    if filename.endswith(".npz"):
        filename = filename[:-4]

    if "Dataset" in filename:
        new_name = filename.replace("Dataset", "Package")
    
    elif "Package" in filename:
        new_name = filename
    
    else:
        new_name = "Package_" + filename

    return new_name

def select_model(path, json_dir):

    json_path = os.path.join(path, json_dir)

    # Vérification de l'existence du fichier
    if not os.path.exists(json_path):
        print(f"[ERREUR] Fichier '{json_dir}' non trouvé.")
        print(f"Chemin testé (absolu) : {os.path.abspath(json_path)}\n")
        exit(1)

    print("\nModèles disponibles:")
    df = show_info_main(json_path)
        
    # Étape 4 : Demander à l'utilisateur de choisir un modèle
    index = 0
    while index < 1 or index > len(df):
        try:
            index = int(input(f"\nQuel modèle souhaitez-vous charger ? (1 à {len(df)})(0 exit)\n ") )
        except ValueError:
            continue
        if index == 0:
            exit(1)
        if index < 1 or index > len(df):
            print(f"Veuillez entrer un nombre entre 1 et {len(df)}")

    # Étape 5 : Récupérer la ligne choisie (index - 1 car affichage commence à 1)
    selected_row = df.iloc[index - 1]

    # Convertir toute la ligne en dictionnaire
    model_info_dict = selected_row.to_dict()
    
    # Extraire le nom du modèle à partir du dictionnaire
    selected_model_name = model_info_dict["Name"]

    print(f"\nModèle sélectionné : {selected_model_name}")

    # Étape 6 : Chercher le fichier dans le dossier Model/
    model_dir = os.path.join(path, FOLDER_NAME_MODEL, selected_model_name + ".npz")

    if not os.path.exists(model_dir):
        chemin_absolu = os.path.abspath(model_dir)
        print(f"[ERREUR] Dossier '{model_dir}' non trouvé.")
        print(f"Chemin testé (absolu) : {chemin_absolu}\n")
        exit(1)

    print(f"\n✅ Modèle sélectionné : {selected_model_name}")
    print(f"📂 Chemin : {model_dir}")

    return selected_model_name, model_dir
