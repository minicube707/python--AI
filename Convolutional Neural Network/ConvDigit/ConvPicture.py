
import os
import numpy as np
import matplotlib

import matplotlib.pyplot as plt
from PIL import Image

from System.Manage_file import select_model, load_model
from System.User_Input import handle_key

from System.Constante import FOLDER_NAME_LOGBOOK

from utilsConv import lister_dossiers, picture_preprocessing, picture_prediction

matplotlib.use("TkAgg")  # Issue on linux PC 42

module_dir = os.path.dirname(__file__)
os.chdir(module_dir)


def research(model, hyperparams, dataset):
    
    file_paths = input("\nEnter the path to load your picture:\n").strip().strip('"')
    
    if file_paths == "0":
        exit(0)
        
    img_shape = hyperparams.input_shape
    
    class_to_idx = dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    # =========================
    # Preprocessing
    # =========================
    img = Image.open(file_paths)
    img_array = picture_preprocessing(img, img_shape)
   
    # =========================
    # Prediction
    # =========================
    prediction_scores, predicted_class, confidence_score = picture_prediction(model, hyperparams, img_array)
    
    # =========================
    # Display
    # =========================  
    display_pred = idx_to_class[predicted_class]
     
    # Création de la figure avec 2 sous-graphiques (image + histogramme)
    fig, axs = plt.subplots(2, 1, figsize=(5, 7), gridspec_kw={'height_ratios': [3, 1]})
    fig.canvas.mpl_connect('key_press_event', handle_key)  # Connecte l'événement clavier

    # Affichage de l'image
    if img_array.ndim == 2:
        axs[0].imshow(img_array[0])
    else:
        axs[0].imshow(img_array[0].transpose(1, 2, 0))
        
    axs[0].set_title(f"Predict:{display_pred} ({np.round(confidence_score, 2)}%)")
    axs[0].axis("off")

    # Affichage de l'histogramme des probabilités
    axs[1].bar(range(len(prediction_scores)), prediction_scores, color="blue")
    axs[1].set_xticks(range(len(prediction_scores)))
    axs[1].set_xlabel("Classes")
    axs[1].set_ylabel("Probability")
    axs[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.show()



#Main algorithm
def main ():

    module_dir = lister_dossiers() 
    model_name, _ = select_model(module_dir, FOLDER_NAME_LOGBOOK)
    model, hyperparams, _, _, dataset, _ = load_model(module_dir, model_name, None)    
    
    while(1):
        research(model, hyperparams, dataset)
                
               
main()