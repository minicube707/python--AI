
import os
import numpy as np
import cupy as cp

from PIL import Image, ImageOps

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


def picture_preprocessing(img, img_shape):
    
    img_size = (img_shape[1], img_shape[2])
    
    # RGB / grayscale
    if img_shape[0] == 3:
        img = img.convert("RGB")
    else:
        img = img.convert("L")
        
    # Resize sans déformation (padding)
    img = ImageOps.pad(img, img_size, method=Image.Resampling.LANCZOS)            
    img_array = np.array(img) / np.max(img)  # normalisation

    # add channel if grayscale
    if  img_array.ndim == 2:
        img_array = img_array[None, None, :, :]  # (1, 1, H, W)

    elif img_array.ndim == 3:
        img_array = np.transpose(img_array, (2, 0, 1))  # (C, H, W)
        img_array = img_array[None, :, :, :]  # (1, C, H, W)
    
    else:
        raise ValueError(f"Unexpected image shape: {img_array.shape}")
      
    return img_array


def picture_prediction(model, hyperparams, img_array):
    
    if (model.support == "GPU"):
        img_array = cp.array(img_array)

    # Prediction
    prediction_scores = model.forward_propagation(img_array, False).flatten()
    
    if (model.support == "GPU"):
        prediction_scores = cp.asnumpy(prediction_scores)
        img_array = cp.asnumpy(img_array)
    
    # Classification
    if hyperparams.loss_metric == "CrossEntropyLoss":
        predicted_class = np.argmax(prediction_scores)
        confidence_score = np.max(prediction_scores)
    
    else:
        predicted_class = (prediction_scores >= 0.5).astype(int).item()
        confidence_score = np.max(prediction_scores)
    
    return prediction_scores, predicted_class, confidence_score