
import os
import numpy as np
import cupy as cp
import matplotlib
import time
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

from System.Manage_file import select_model, load_model
from System.User_Input import handle_key

from System.Constante import FOLDER_NAME_LOGBOOK

matplotlib.use("TkAgg")  # Issue on linux PC 42

module_dir = os.path.dirname(__file__)
os.chdir(module_dir)


def research(model,hyperparams,dataset,interval=0.5):

    img_shape = hyperparams.input_shape
    img_size = (img_shape[1], img_shape[2])

    class_to_idx = dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise Exception("Impossible d'ouvrir la webcam")

    last_prediction_time = 0

    current_text = "Waiting prediction..."

    print("Q pour quitter")

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        now = time.time()

        # =========================
        # New prediction
        # =========================
        if now - last_prediction_time >= interval:

            last_prediction_time = now

            # OpenCV BGR -> RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            img = Image.fromarray(frame_rgb)

            # RGB / grayscale
            if img_shape[0] == 3:
                img = img.convert("RGB")
            else:
                img = img.convert("L")

            # Resize
            img = ImageOps.pad(img, img_size, method=Image.Resampling.LANCZOS)

            # =========================
            # Preprocessing
            # =========================
            img_array = np.array(img).astype(np.float32)

            if np.max(img_array) > 0:
                img_array /= np.max(img_array)

            # grayscale
            if img_array.ndim == 2:
                img_array = img_array[None, None, :, :]

            # RGB
            elif img_array.ndim == 3:
                img_array = np.transpose(img_array, (2, 0, 1))
                img_array = img_array[None, :, :, :]

            # GPU
            if model.support == "GPU":
                img_array = cp.array(img_array)

            # =========================
            # Prediction
            # =========================
            y_pred = model.forward_propagation(img_array, False).flatten()

            if model.support == "GPU":
                y_pred = cp.asnumpy(y_pred)

            # Classification
            if hyperparams.loss_metric == "CrossEntropyLoss":
                pred = np.argmax(y_pred)
                confidence = float(np.max(y_pred))

            else:
                pred = int((y_pred >= 0.5).astype(int).item())
                confidence = float(np.max(y_pred))

            display_pred = idx_to_class[pred]

            current_text = (f"{display_pred} ({confidence:.2f})")

        # =========================
        # Display
        # =========================
        cv2.putText(frame,current_text,(20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Live Prediction", frame)

        # Quitter
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return 1

    cap.release()
    cv2.destroyAllWindows()


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
    model_name, _ = select_model(module_dir, FOLDER_NAME_LOGBOOK)
    model, hyperparams, _, _, dataset, _ = load_model(module_dir, model_name, None)    
    
    while(1):
        if (research(model, hyperparams, dataset)):
            break
                
               
main()