import os
import pickle
import pandas as pd
from datetime import datetime

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



def save_model(path, model_name, data):

    model_path = os.path.join(path, "Model")
    
    if not os.path.exists(model_path):
        chemin_absolu = os.path.abspath(model_path)
        print(f"[ERREUR] Dossier 'Model' non trouvé.")
        print(f"Chemin testé (absolu) : {chemin_absolu}\n")
        exit(1)

    model_path = os.path.join(model_path, model_name)
    with open(model_path, 'wb') as file:
        pickle.dump(data, file)


def file_management(test_accu, test_conf):
    str_accu = f"{test_accu:.5f}".replace(".", ",")
    str_conf = f"{test_conf:.5f}".replace(".", ",")

     # Obtenir la date du jour au format JJ-MM-AAAA
    date_str = datetime.now().strftime("%d-%m-%Y")

    name_model = f"DM({str_accu})({str_conf})({date_str}).pickle"

    return name_model


def select_model(path, csv_file):

    # Étape 2 : Lire le fichier CSV dans le dossier logbook
    logbook_path = os.path.join(path, "LogBook", csv_file)
    
    if not os.path.exists(logbook_path):
        chemin_absolu = os.path.abspath(logbook_path)
        print(f"[ERREUR] Fichier '{csv_file}' non trouvé.")
        print(f"Chemin testé (absolu) : {chemin_absolu}\n")
        exit(1)

    df = pd.read_csv(logbook_path, sep=';')

    # Étape 3 : Afficher les lignes disponibles dans le logbook
    print("\nModèles disponibles dans le logbook:")
    show_info_main("LogBook", csv_file)

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
