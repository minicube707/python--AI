
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
    _, predicted_class, confidence_score = picture_prediction(model, hyperparams, img_array)

    display_pred = idx_to_class[predicted_class]
    current_text = (f"{display_pred} ({confidence_score:.2f})")
    
    return current_text


def get_center_square_coords(width: int, height: int, size: int):

    #Image center
    cx, cy = width // 2, height // 2

    #Square coordinates
    x1 = cx - size // 2
    y1 = cy - size // 2
    x2 = cx + size // 2
    y2 = cy + size // 2

    return x1, y1, x2, y2


class ComparisonLayerDisplay:
    def __init__(self, shape, Z_enabled=True):
        """
        shape : (D, H, W)
        """

        self.D = shape[0]
        max_par_fig = self.D
        self.max_par_fig = max_par_fig
        self.Z_enabled = Z_enabled


        cols = min(4, min(self.D, max_par_fig))
        rows = int(np.ceil(min(self.D, max_par_fig) / cols))

        self.cols = cols
        self.rows = rows

        fig_cols = cols * 2 if Z_enabled else cols

        self.fig, self.axes = plt.subplots(rows, fig_cols, figsize=(2.2 * fig_cols, 2.2 * rows))
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02, hspace=0.02)

        if rows == 1:
            self.axes = np.expand_dims(self.axes, 0)

        if fig_cols == 1:
            self.axes = np.expand_dims(self.axes, 1)

        self.images_A = []
        self.images_Z = []

        n = min(self.D, max_par_fig)

        for i in range(n):

            row = i // cols
            col = i % cols

            # ---------- A ----------
            ax_a = self.axes[row, col * 2] if Z_enabled else self.axes[row, col]

            img_a = ax_a.imshow(np.zeros(shape[1:]), animated=True,)

            ax_a.set_xticks([])
            ax_a.set_yticks([])
            ax_a.set_frame_on(False)

            self.images_A.append(img_a)

            # ---------- Z ----------
            if Z_enabled:

                ax_z = self.axes[row, col * 2 + 1]

                img_z = ax_z.imshow(np.zeros(shape[1:]), animated=True)

                ax_z.set_xticks([])
                ax_z.set_yticks([])
                ax_z.set_frame_on(False)

                self.images_Z.append(img_z)

        plt.show(block=False)

    def update(self, A, Z=None):

        if A.ndim == 4:
            A = A[0]

        if Z is not None and Z.ndim == 4:
            Z = Z[0]

        n = min(A.shape[0], self.max_par_fig)

        for i in range(n):

            self.images_A[i].set_data(A[i])
            self.images_A[i].set_clim(vmin=np.min(A[i]), vmax=np.max(A[i]))

            if self.Z_enabled and Z is not None:

                self.images_Z[i].set_data(Z[i])
                self.images_Z[i].set_clim(vmin=np.min(Z[i]), vmax=np.max(Z[i]))

        self.fig.canvas.draw_idle()
        plt.pause(0.001)

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
    time_to_change = 10
    last_change_time=0
    level = 0
    print("Q to leave")

    viewer = ComparisonLayerDisplay(shape=(24, 64, 64), Z_enabled=True)
    
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
            current_text = new_prediction(model, hyperparams, frame, img_shape, idx_to_class)
            
        # =========================
        # Display
        # =========================  
        cv2.putText(frame, current_text,(20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("Live Prediction", frame)
    
        # Quitter
        if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
            return 1
            
        if now - last_change_time >= time_to_change:
            last_change_time = now
            level += 1
            
        if level >= model.cnn_model.C_CNN:
            level = 0
            
        img = Image.fromarray(frame)
        img_array = picture_preprocessing(img, img_shape)
        A, Z =  model.cnn_model.get_activations(img_array, level)
        viewer.update(A, Z)
            
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