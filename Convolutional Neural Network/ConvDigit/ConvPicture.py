
import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # Issue on linux PC 42
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageOps

from System.Manage_file import select_model, load_model
from System.Preprocessing import  handle_key

module_dir = os.path.dirname(__file__)
os.chdir(module_dir)

def research(model, hyperparams):
    
    file_paths = input("Enter the path to load your picture:\n").strip().strip('"')
    img_shape = hyperparams.input_shape
    img_size = (img_shape[1], img_shape[2])
    
    # Lecture image
    if (img_shape[0] == 3):
        img = Image.open(file_paths).convert('RGB')  # 'L' pour grayscale, 'RGB' si couleur
        
    else:
        img = Image.open(file_paths).convert('L')
        
    # Resize sans déformation (padding)
    img = ImageOps.pad(img, img_size, method=Image.Resampling.LANCZOS)            
    img_array = np.array(img) / np.max(img)  # normalisation

    # ajouter canal si grayscale
    if  img_array.ndim == 2:
        img_array = img_array[None, None, :, :]  # (1, 1, H, W)

    elif img_array.ndim == 3:
        img_array = np.transpose(img_array, (2, 0, 1))  # (C, H, W)
        img_array = img_array[None, :, :, :]  # (1, C, H, W)

    else:
        raise ValueError(f"Unexpected image shape: {img_array.shape}")

    # Prédiction
    y_pred = model.forward_propagation(img_array, False).flatten()

    if hyperparams.loss_metric == "CrossEntropyLoss":
        pred = np.argmax(y_pred)
        porcent = np.max(y_pred)
    
    else:
        pred = (y_pred >= 0.5).astype(int).item()
        porcent = np.max(y_pred)
    
    # Création de la figure avec 2 sous-graphiques (image + histogramme)
    fig, axs = plt.subplots(2, 1, figsize=(5, 7), gridspec_kw={'height_ratios': [3, 1]})
    fig.canvas.mpl_connect('key_press_event', handle_key)  # Connecte l'événement clavier

    # Affichage de l'image
    if img_array.ndim == 2:
        axs[0].imshow(img_array[0])
    else:
        axs[0].imshow(img_array[0].transpose(1, 2, 0))
        
    axs[0].set_title(f"Predict:{pred} ({np.round(porcent, 2)}%)")
    axs[0].axis("off")

    # Affichage de l'histogramme des probabilités
    axs[1].bar(range(len(y_pred)), y_pred, color="blue")
    axs[1].set_xticks(range(len(y_pred)))
    axs[1].set_xlabel("Classes")
    axs[1].set_ylabel("Probability")
    axs[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.show()


def lister_dossiers():
    # Récupère le chemin du répertoire courant
    repertoire_courant = os.getcwd()
    
    # Liste uniquement les dossiers qui contient des models
    dossiers = [
    d for d in os.listdir(repertoire_courant)
    if os.path.isdir(os.path.join(repertoire_courant, d)) and "Package" in d
    ]
            
    if not dossiers:
        print("Aucun dossier trouvé dans le répertoire courant.")
        return None
    
    # Affiche les dossiers avec un numéro
    print("Dossiers disponibles :")
    for i, dossier in enumerate(dossiers, start=1):
        print(f"{i}. {dossier}")
    
    # Demande à l'utilisateur de choisir un dossier
    while True:
        try:
            choix = int(input("\nEntrez le numéro du dossier à choisir : "))
            if 1 <= choix <= len(dossiers):
                dossier_choisi = dossiers[choix - 1]
                print(f"\nVous avez choisi : {dossier_choisi}")
                return dossier_choisi
            
            elif choix == 0:
                exit(1)

            else:
                print("Numéro invalide, réessayez.")

        except ValueError:
            print("Veuillez entrer un nombre valide.")


#Main algorithm
def main ():

    module_dir = lister_dossiers() 
    model_name = select_model(module_dir, "LogBook")
    model, hyperparams, _, _, _, _ = load_model(module_dir, model_name, None)    
    research(model, hyperparams)
                
               
main()