import os
import numpy as np
from sklearn.datasets import load_iris

# Dossier où enregistrer le dataset
dataset_path = "Dataset"
os.makedirs(dataset_path, exist_ok=True)

# Nom du fichier de sortie
selected_file = "CIRFA10.npz"
file_path = os.path.join(dataset_path, selected_file)

# Chargement du dataset Iris depuis sklearn
iris = load_iris()

# Sauvegarde dans un fichier .npz
np.savez(file_path, data=iris.data, target=iris.target)

print(f"✅ Dataset enregistré dans : {file_path}")

# --- Exemple de rechargement (comme ton code) ---
with np.load(os.path.join(dataset_path, selected_file)) as f:
    data, target, selected = f["data"], f["target"], selected_file

print("✅ Chargement réussi !")
print("Forme de data :", data.shape)
print("Forme de target :", target.shape)
