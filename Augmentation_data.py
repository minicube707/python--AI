"""
augment_mnist.py
----------------
Script pour augmenter un dataset MNIST en NumPy :
- Décalage de 4 pixels (haut, bas, gauche, droite)
- Ajout de bruit
- Création d'une 11e classe "autre"
"""

import numpy as np
import os
import matplotlib.pyplot as plt

module_dir = os.path.dirname(__file__)
os.chdir(module_dir)

# ==============================
#  Fonctions utilitaires
# ==============================

def load_mnist_npz(dataset_path, selected_file):
    """Charge un fichier MNIST .npz"""
    print("PATH ", os.path.join(dataset_path, selected_file))
    with np.load(os.path.join(dataset_path, selected_file)) as f:
        X, y = f["data"], f["target"]
    X = X.astype(np.float32) / 255.0  # normalisation
    print(f"[INFO] Dataset chargé : {X.shape[0]} images, shape {X.shape[1:]}")
    return X, y


def shift_images(X, shift_x=0, shift_y=0):
    """Décale les images de quelques pixels"""
    shifted = np.roll(X, shift_x, axis=2)
    shifted = np.roll(shifted, shift_y, axis=1)
    
    # Mise à zéro des bandes créées
    if shift_x > 0:
        shifted[:, :, :shift_x] = 0
    elif shift_x < 0:
        shifted[:, :, shift_x:] = 0
    if shift_y > 0:
        shifted[:, :shift_y, :] = 0
    elif shift_y < 0:
        shifted[:, shift_y:, :] = 0
    return shifted


def add_noise(X, noise_level=0.2):
    """Ajoute du bruit gaussien"""
    noise = np.random.normal(0, noise_level, X.shape)
    X_noisy = X + noise
    return np.clip(X_noisy, 0.0, 1.0)


def create_other_class(X, n_other_ratio=0.1):
    """Crée une classe 10 ("autre") à partir de bruit aléatoire"""
    n_other = int(len(X) * n_other_ratio)
    # mélange de bruit pur + bruit sur des images réelles
    mix = X[np.random.randint(0, len(X), (n_other,))]
    mask = np.random.rand(*mix.shape) > 0.5
    X_other = mask * mix + (1 - mask) * np.random.rand(*mix.shape)
    y_other = np.full(n_other, 10)  # classe 10 = "autre"
    print(f"[INFO] Création de {n_other} images pour la classe 'autre'")
    return X_other, y_other


def handle_key(event):
    if event.key == ' ':
        plt.close(event.canvas.figure)  # Ferme la fenêtre associée

def visualize_samples(X, y, n=20):
    """Affiche n images de chaque classe"""
    
    classes = np.unique(y)
    for cls in classes:
        # Crée une figure pour chaque classe
        fig = plt.figure(figsize=(16, 8))
        fig.suptitle(f"Classe {cls}", fontsize=16)
        fig.canvas.mpl_connect('key_press_event', handle_key)  # Active la détection de la touche

        # Récupère les indices des images correspondant à cette classe
        indices = np.where(y == cls)[0][:n]
        
        # Affiche les n premières images
        for i, idx in enumerate(indices):
            plt.subplot(2, n // 2, i + 1)
            plt.imshow(X[idx], cmap='gray')
            plt.title(f"{int(y[idx])}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()


# ==============================
#  Pipeline principal
# ==============================

def augment_mnist(dataset_path, selected_file, save_path=None):
    X, y = load_mnist_npz(dataset_path, selected_file)

    # --- Décalages de 4 pixels ---
    print("[INFO] Application des décalages ...")
    X_up = shift_images(X, shift_y=-4)
    X_down = shift_images(X, shift_y=4)
    X_left = shift_images(X, shift_x=-4)
    X_right = shift_images(X, shift_x=4)

    X_shifted = np.concatenate([X, X_up, X_down, X_left, X_right], axis=0)
    y_shifted = np.concatenate([y] * 5, axis=0)

    # --- Ajout de bruit ---
    print("[INFO] Ajout du bruit ...")
    X_noisy = add_noise(X_shifted, noise_level=0.2)

    # --- Création de la classe "autre" ---
    #X_other, y_other = create_other_class(X_noisy, n_other_ratio=0.1)

    # --- Fusion finale ---
    #X_final = np.concatenate([X_noisy, X_other], axis=0)
    #y_final = np.concatenate([y_shifted, y_other], axis=0)
    X_final = X_noisy
    y_final = y_shifted

    print(f"[INFO] Dataset final : {X_final.shape}, classes uniques : {np.unique(y_final)}")

    # --- Visualisation rapide ---
    visualize_samples(X_final, y_final)

    # --- Sauvegarde optionnelle ---
    if save_path is not None:
        np.savez_compressed(save_path, data=X_final, target=y_final)
        print(f"[INFO] Dataset sauvegardé dans {save_path}")

    return X_final, y_final


# ==============================
#  Exemple d'utilisation
# ==============================

if __name__ == "__main__":
    dataset_path = r"Dataset"
    selected_file = "Dataset M-NIST Digit1.npz"
    save_path = r"Dataset\Dataset M-NIST Digit2.npz"

    full_path = os.path.join(dataset_path, selected_file)
    print("Trying to open:", full_path)
    print("Exists?", os.path.exists(full_path))

    X_final, y_final = augment_mnist(dataset_path, selected_file, save_path)
    print(np.unique(y_final))
