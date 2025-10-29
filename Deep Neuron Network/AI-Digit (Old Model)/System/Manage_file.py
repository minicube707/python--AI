import os
import re
import pickle
import pandas as pd

from .Manage_logbook import show_info_main

def load_model(path, model_name):
    model_dir = os.path.join(path, "Model")
    model_path = os.path.join(model_dir, model_name)
    
    if not os.path.exists(model_path):
        chemin_absolu = os.path.abspath(model_path)
        print(f"[ERREUR] Fichier '{model_name}' non trouvé.")
        print(f"Chemin testé (absolu) : {chemin_absolu}\n")

        # Liste les fichiers disponibles pour aider au debug
        if os.path.exists(model_dir):
            print("📂 Fichiers disponibles dans le dossier Model :")
            for f in os.listdir(model_dir):
                print(" -", f)
        else:
            print("❌ Le dossier 'Model' n'existe pas.")

        exit(1)

    with open(model_path, 'rb') as file:
        return pickle.load(file)

def transform_name(filename: str) -> str:
    # Remplacer "Dataset" par "Package"
    new_name = filename.replace("Dataset", "Package")
    # Retirer ".npz" à la fin
    if new_name.endswith(".npz"):
        new_name = new_name[:-4]
    return new_name

def save_model(path, model_name, data):

    model_path = os.path.join(path, "Model")
    
    # Créer le dossier s'il n'existe pas
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        print(f"[INFO] Dossier 'Model' créé à : {os.path.abspath(model_path)}")

    # Sauvegarder le modèle
    model_path = os.path.join(model_path, model_name)
    with open(model_path, 'wb') as file:
        pickle.dump(data, file)
    print(f"SUCCÈS: Modèle sauvegardé")


def file_management(dimensions, test_accu):
    str_dim = str(tuple(dimensions)).replace(" ", "")
    str_accu = f"{test_accu:.5f}".replace(".", ",")
    name_model = f"DM{str_dim}({str_accu}).pickle"

    return name_model

def select_model(path, csv_file):

    # Étape 2 : Lire le fichier CSV dans le dossier logbook
    logbook_path = os.path.join(path, csv_file)
    
    if not os.path.exists(logbook_path):
        chemin_absolu = os.path.abspath(logbook_path)
        print(f"[ERREUR] Fichier '{csv_file}' non trouvé.")
        print(f"Chemin testé (absolu) : {chemin_absolu}\n")
        exit(1)

    df = pd.read_csv(logbook_path, sep=';')

    # Étape 3 : Afficher les lignes disponibles dans le logbook
    print("\nModèles disponibles dans le logbook:")
    show_info_main(path, csv_file)

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
    selected_model_name = model_info_dict['name']

    print(f"\nModèle sélectionné : {selected_model_name}")

    # Étape 6 : Chercher le fichier dans le dossier Model/
    model_dir = os.path.join(path, "Model")

    if not os.path.exists(model_dir):
        chemin_absolu = os.path.abspath(model_dir)
        print(f"[ERREUR] Dossier '{model_dir}' non trouvé.")
        print(f"Chemin testé (absolu) : {chemin_absolu}\n")
        exit(1)

    print(f"\n✅ Modèle sélectionné : {selected_model_name}")
    print(f"📂 Chemin : {model_dir}")

    return selected_model_name, model_info_dict


def get_hidden_layers(name):

    match = re.search(r"\(([^)]*)\)", name)
    numbers = re.findall(r"\d+", match.group(1))  # Extraire uniquement les nombres
    result = tuple(map(int, numbers))  # Convertir en tuple d'entiers

    print("\nHidden layers:\n",result)  
    return result 