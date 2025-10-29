
import os
import sys
import  numpy as np

module_dir = os.path.dirname(__file__)
os.chdir(module_dir)

def manage_data():
    print(module_dir)
    dataset_path = os.path.join(module_dir, "../../../Dataset") 
    
    # Vérifier si le dossier Dataset existe
    if not os.path.exists(dataset_path):
        print(f"❌ Erreur : le dossier '{dataset_path}' n'existe pas.")
        sys.exit(1)

    # Ne garder que les fichiers
    folders = os.listdir(dataset_path)
    files = [f for f in folders if os.path.isfile(os.path.join(dataset_path, f))]

    if not files:
        print(f"⚠️ Aucun fichier trouvé dans '{dataset_path}'.")
        exit(1)

    # Afficher les fichiers avec un numéro
    print("Sélectionnez un fichier en entrant son numéro :")
    for idx, file in enumerate(files, start=1):
        print(f"{idx}. {file}")

    # Demander à l'utilisateur de choisir
    while True:
        choice = input("Entrez le numéro du fichier : ")
        if not choice.isdigit():
            print("❌ Veuillez entrer un numéro valide.")
            continue

        choice = int(choice)
        if 1 <= choice <= len(files):
            selected_file = files[choice - 1]
            print(f"\n✅ Vous avez sélectionné : {selected_file}")
            with np.load(os.path.join(dataset_path, selected_file)) as f:
                return f["data"], f["target"], selected_file

        elif choice == 0:
            exit(0)

        else:
            print(f"❌ Numéro invalide. Veuillez choisir entre 1 et {len(files)}.")
