import pandas as pd
import os
import json
from dataclasses import asdict

def create_new_log(new_log, path, json_file):

    # Chemin complet du fichier JSON
    filename = os.path.join(path, json_file)
    
    # Créer le dossier si besoin
    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"[INFO] Dossier créé : {os.path.abspath(folder)}")

    df = pd.DataFrame()
    df = pd.concat([df, pd.DataFrame([new_log])], ignore_index=True)
    df.to_json(filename, orient='records', indent=4)
    print("Update LogBook")


def save_model_configuration(mode, 
                   hyperparams, performance, dataset, structure,
                   elapsed_time_minutes,
                   metadata, metadata_old,
                   module_dir):

    if mode in {1}:
        metadata["Based_model"] = "X"
        metadata["total_epoch"] = hyperparams.nb_epoch
        metadata["training_time_(min)"] = elapsed_time_minutes
        metadata["Number_fine_tuning"] = 0

    else:
        metadata["Based_model"] = metadata_old["name"]
        metadata["total_epoch"] = metadata_old["total_epoch"] + hyperparams.nb_epoch
        metadata["training_time_(min)"] = metadata_old["training_time_(min)"] + elapsed_time_minutes
        metadata["Number_fine_tuning"] = metadata_old["Number_fine_tuning"] + 1

    new_log = {
        "hyperparameters": hyperparams,
        "structure" : structure,
        "performance": performance,
        "dataset": dataset,
        "metadata": metadata,
    }
    
    create_new_log(new_log, os.path.join(module_dir, "LogBook"), metadata["name"] + ".json")


def show_all_info_model(hyperparams, structure, performance, dataset, metadata):
    
    print("\nHyperparams:")
    for key, value in asdict(hyperparams).items():
        print(f"{key}: {value}")

    print("\nStructure CNN:")
    for key, value in structure[0].items():
        print(f"{key}: {value}")

    print("\nStructure DNN:")
    for key, value in structure[1].items():
        print(f"{key}: {value}")

    print("\nPerformance:")
    for key, value in performance.items():
        print(f"{key}: {value}")

    print("\nDataset:")
    for key, value in asdict(dataset).items():
        print(f"{key}: {value}")

    print("\nMetadata:")
    for key, value in metadata.items():
        print(f"{key}: {value}")


def show_info_main(json_path):
    json_files = [f for f in os.listdir(json_path) if f.endswith(".json")]
    if not json_files:
        print(f"[INFO] Aucun fichier JSON trouvé dans '{json_path}'.")
        return

    # Colonnes principales à afficher (aplaties avec json_normalize)
    columns_to_show = [
        "metadata_name",
        "metadata_date",
        "metadata_training_time_(min)",
        "performance_accuracy",
        "performance_confidence_score",
        "metadata_Number_fine_tuning",
        "metadata_Based_model"
    ]
    
    all_dfs = []

    for json_file in json_files:
        logbook_path = os.path.join(json_path, json_file)

        try:
            with open(logbook_path, "r") as f:
                data = json.load(f)
        except (ValueError, json.JSONDecodeError):
            print(f"[ERREUR] Le fichier '{json_file}' est vide ou mal formé.")
            continue

        # Normalisation du JSON imbriqué
        df = pd.json_normalize(data, sep="_")

        # Ne garder que les colonnes existantes
        existing_columns = [col for col in columns_to_show if col in df.columns]
        df_to_show = df[existing_columns].copy()
        all_dfs.append(df_to_show)

    if not all_dfs:
        print("[INFO] Aucun log valide trouvé.")
        return

    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df.index += 1  # index à partir de 1
    print(final_df.to_string(index=True))
    return final_df