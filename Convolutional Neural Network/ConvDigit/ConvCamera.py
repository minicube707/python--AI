
import os
import matplotlib
import numpy as np
import time
import cv2
import matplotlib.pyplot as plt
from PIL import Image

from System.Manage_file import select_model, load_model
from System.User_Input import handle_key

from System.Constante import FOLDER_NAME_LOGBOOK

from utilsConv import lister_dossiers, picture_preprocessing, picture_prediction

matplotlib.use("TkAgg")  # Issue on linux PC 42

module_dir = os.path.dirname(__file__)
os.chdir(module_dir)


def new_prediction(model, hyperparams, frame, img_shape, idx_to_class):
    
    #OpenCV BGR -> RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # =========================
    # Preprocessing
    # =========================
    img = Image.fromarray(frame_rgb)
    img_array = picture_preprocessing(img, img_shape)
    
    # =========================
    # Prediction
    # =========================
    prediction_scores, predicted_class, confidence_score = picture_prediction(model, hyperparams, img_array)

    pred_class = idx_to_class[predicted_class]
    
    return prediction_scores, pred_class, confidence_score, img_array


def get_center_square_coords(width: int, height: int, size: int):

    #Image center
    cx, cy = width // 2, height // 2

    #Square coordinates
    x1 = cx - size // 2
    y1 = cy - size // 2
    x2 = cx + size // 2
    y2 = cy + size // 2

    return x1, y1, x2, y2


def init(n_classes):

    # Figure
    fig, axs = plt.subplots(2, 1, figsize=(5, 7), gridspec_kw={'height_ratios': [3, 1]})

    # Event clavier
    fig.canvas.mpl_connect('key_press_event', handle_key)

    # Histogramme initial
    bars = axs[1].bar(range(n_classes), [0] * n_classes, color="blue")

    axs[1].set_xticks(range(n_classes))
    axs[1].set_xlabel("Classes")
    axs[1].set_ylabel("Probability")
    axs[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.show(block=False)

    return fig, axs, bars


def update(axs, bars,
           prediction_scores,
           predicted_class,
           confidence_score,
           img_array):

    # -------- IMAGE --------

    axs[0].clear()

    if img_array.ndim == 2:
        axs[0].imshow(img_array[0], cmap="gray")
    else:
        axs[0].imshow(img_array[0].transpose(1, 2, 0))

    axs[0].set_title(f"Predict: {predicted_class} ({np.round(confidence_score, 2)}%)")
    axs[0].axis("off")

    # -------- HISTOGRAMME --------

    for bar, score in zip(bars, prediction_scores):
        bar.set_height(score)
        bar.set_color("blue")

    plt.pause(0.01)

  
def research(model, hyperparams, dataset, interval=1):

    img_shape = hyperparams.input_shape

    class_to_idx = dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise Exception("Unable to open webcam")

    last_prediction_time = 0
    current_text = "Waiting prediction..."
    squarre_size = 300
    
    print("Q to leave")

    fig, axs, bars = init(1)
    
    while True:

        ret, frame = cap.read()
        h, w, _ = frame.shape

        if not ret:
            break

        # =========================
        # New prediction
        # =========================
        now = time.time()
        if now - last_prediction_time >= interval:
            last_prediction_time = now
            
            # Extract the centered square region and run the model prediction on it
            x1, y1, x2, y2 = get_center_square_coords(w, h, squarre_size)
            squarre_frame = frame[y1:y2, x1:x2]
            prediction_scores, predicted_class, confidence_score, img_array = new_prediction(model, hyperparams, squarre_frame, img_shape, idx_to_class)
            
        # =========================
        # Display
        # =========================  
        #cv2.putText(frame, current_text,(20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("Live Prediction", frame)
    
        # Quitter
        if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
            return 1
     
        update(axs, bars, prediction_scores, predicted_class, confidence_score, img_array)
 
    cap.release()
    cv2.destroyAllWindows()


#Main algorithm
def main ():

    module_dir = lister_dossiers() 
    model_name, _ = select_model(module_dir, FOLDER_NAME_LOGBOOK)
    model, hyperparams, _, _, dataset, _ = load_model(module_dir, model_name, None)    
    
    while(1):
        if (research(model, hyperparams, dataset)):
            break
                
               
main()