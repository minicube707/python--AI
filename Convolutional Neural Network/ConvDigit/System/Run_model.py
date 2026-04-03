
import os
import numpy as np

from datetime import datetime
from PIL import Image
from tqdm import tqdm

#System
from .Set_mode import set_mode

from .Manage_data import manage_data
from .Manage_file import file_management, select_model, load_model, save_model_parameters, transform_name
from .Manage_logbook import save_model_configuration, show_all_info_model

from .Preprocessing import preprocessing, get_data_shape

from .FullModel import FullModel
from .Training import training_full_data, training_batch_data

from .Display_parametre_CNN import display_kernel_and_biais, display_first_picture, display_dataset

def load_file_paths(base_dir):
    file_paths = []
    labels = []

    for label in tqdm(['0','1'], desc="Classes"):
        folder = os.path.join(base_dir, label)
        for filename in os.listdir(folder):
            if filename.endswith('.jpg'):
                file_paths.append(os.path.join(folder, filename))
                labels.append(int(label))

    return file_paths, labels

def change_dim_picture(input_shape):

    print("Current shape of a picture: ", input_shape)

    while(1):
        str_load_in_color = input("Do you want to load the images in color? (yes/no): ").strip().lower()
        
        if str_load_in_color == "yes" or str_load_in_color == "y":
            print("Images will be loaded in color (RGB).")
            picture_in_RGB = True
            break
        
        elif str_load_in_color == "non" or str_load_in_color == "n":
            print("Images will be loaded in grayscale (black and white).")
            picture_in_RGB = False
            break

        else:
            print("Error: Please enter yes or no")

    while(1):
        str_answer = input("Which shape do you want to train: ").strip()
        
        try:
            int_answer = int(str_answer)

        except:
            print("Please enter a number")
            continue

    if (not picture_in_RGB):
        input_shape = (1, int_answer, int_answer)

    else:
        input_shape = (input_shape[0], int_answer, int_answer)
	
    return picture_in_RGB, input_shape
    
def run_training_pipeline(module_dir, hyperparams, structure, loss_metric, output_layer, optimizer, dataset):

    while True:
        answer = input("Is your dataset a single .npz file? (Yes/No)\n")
        if answer == "yes" or answer == "y" or answer == "Y" or  answer == "YES":

            dataset_full_size = True

            X, y, data_name = manage_data()
            input_shape, output_shape =  get_data_shape(X, y)

            dir_name = transform_name(data_name)
            module_dir = os.path.join(module_dir, dir_name)

            X_train, y_train, X_test, y_test, transformer = preprocessing(X, y, hyperparams, dataset)
            break

        elif answer == "no" or answer == "n" or answer == "NO" or answer == "N":

            dataset_full_size = False

            data_name = "Breast_Cancer"
            dir_name = transform_name(data_name)
            module_dir = os.path.join(module_dir, dir_name)

            path_train_file = input("Enter the path for the train file:\n")
            train_files, train_labels = load_file_paths(path_train_file)

            path_test_file = input("Enter the path for the test file:\n")
            test_files, test_labels = load_file_paths(path_test_file)

            img = Image.open(train_files[0])
            img_array = np.array(img)

            img_array = img_array.transpose(2, 0, 1)
            input_shape, output_shape =  img_array.shape, 2

            picture_in_RGB, input_shape = change_dim_picture(input_shape)
            break

        else:
            print("Please answer by yes or no")

    mode = set_mode()

    if mode in {4} and dataset_full_size:
        model_name = select_model(module_dir, "LogBook")
        model, hyperparams, structure, performance, dataset, metadata_old = load_model(module_dir, model_name)

        print("")
        show_all_info_model(hyperparams, structure, performance, dataset, metadata_old)
        display_kernel_and_biais(X, y, model.cnn_model)
        exit(0)

    hyperparams.check_support()

    if mode in {1}:

        # ============================
        #     INITIALISATION CNN
        # ============================
        hyperparams.add_shape(input_shape, output_shape)
        model = FullModel(hyperparams, structure, loss_metric, output_layer, optimizer)
        metadata_old = None

    else:

        # ============================
        #       SELECT A MODEL
        # ============================

        # Chargement du modele existant
        model_name = select_model(module_dir, "LogBook")
        model, _, _, _, _, metadata_old = load_model(module_dir, model_name)
        model.set_alpha(hyperparams.alpha)

    if mode in {1, 2}:
        # ============================
        #       TRAINNING
        # ============================

        # Entraînement d'un nouveau modèle

        if dataset_full_size:
            data_test, elapsed_time_minutes = training_full_data(model, X_train, y_train, X_test, y_test, hyperparams, dataset)

        else:
            data_test, elapsed_time_minutes = training_batch_data(model, hyperparams, dataset, train_files, train_labels, test_files, test_labels, picture_in_RGB)
        
        # ============================
        #          SAVE
        # ============================

        # Sauvegarde du meilleur modèle entraîné ou chargé
        date = datetime.today()
        date = date.strftime('%Y-%m-%d_%H-%M-%S')

        name_model = file_management(date, data_test["accu"][-1], data_test["conf"][-1])
        print(name_model)

        metadata = {}
        metadata["name"] = name_model
        metadata["date"] = date
        
        performance = {}
        performance["cost_loss"] = data_test["loss"][-1]
        performance["accuracy"] = data_test["accu"][-1]
        performance["confidence_score"] = data_test["conf"][-1]

        save_model_parameters(module_dir, name_model, model)

        save_model_configuration(mode, 
                    hyperparams, performance, dataset, structure,
                    elapsed_time_minutes,
                    metadata, metadata_old,
                    module_dir)
        
        #______________________________________________________________#
        while(1):
            str_answer = input("How many test do you want to do ?\n")
            try:
                nb_test = int(str_answer)
            except:
                print("Please enter a number")
                continue
        
            if (nb_test == 0):
                print("Exit")
                exit(0)
            
            else:
                break

        if dataset_full_size:
            y_final = transformer.inverse_transform(y_test)
            display_first_picture(model, X_test, y_final)
            display_dataset(model, X_test, y_final, nb_test)