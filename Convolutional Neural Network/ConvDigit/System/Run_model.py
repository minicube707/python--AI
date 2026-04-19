# Standard library
import os
from datetime import datetime

# Third-party
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from PIL import Image, ImageOps

# Local imports
from .Constante import FOLDER_NAME_LOGBOOK

# System
from .Set_mode import set_mode
from .User_Input import ask_yes_no

# Data / file management
from .Manage_data import manage_data
from .Manage_file import (
    file_management,
    select_model,
    load_model,
    save_model_parameters,
    transform_name,
)
from .Manage_logbook import (
    save_model_configuration,
    show_all_info_model,
)

# Preprocessing
from .Preprocessing import preprocessing, get_data_shape

# Model & training
from .FullModel import FullModel
from .Training import (
    training_full_data,
    training_batch_data,
    batch_generator,
)

# Visualization
from .Display_parametre_CNN import (
    display_kernel_and_biais,
    display_first_picture,
    display_dataset,
)

def load_file_paths(base_dir, class_to_idx):
    file_paths = []
    labels = []

    if class_to_idx is None:
        class_names = sorted(os.listdir(base_dir))  # stable
        class_to_idx = {name: idx for idx, name in enumerate(class_names)} #Dict key: name_folder: value: number

    for label in tqdm(class_to_idx.keys(), desc="Classes"):
        folder = os.path.join(base_dir, label)

        if not os.path.isdir(folder):
            continue  # ignore fichiers

        for filename in os.listdir(folder):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_paths.append(os.path.join(folder, filename))
                labels.append(class_to_idx[label])

    print("Number of files uploaded:", len(file_paths))
    return file_paths, labels, class_to_idx

def change_dim_picture(train_picture, input_shape):

    #Shape already init
    if (len(input_shape) == 3):
        if input_shape[0] == 1:
            return False, input_shape
        else:
            return True, input_shape
        
    img = Image.open(train_picture)
    img_array = np.array(img)
            
    if np.ndim(img_array) == 3:
        img_array = img_array.transpose((2, 0, 1))
                
    print("\nCurrent shape of a picture: ", img_array.shape)

    while(1):
        
        if np.ndim(img_array) == 2:
            str_load_in_color = "n"
        else:
            str_load_in_color = input("\nDo you want to load the images in color? (yes/no): ").strip().lower()
        
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
        answer = input("\nWhich shape do you want to train (-1 if unchanged): ").strip()
        
        if (answer == "-1"):
            int_answer = img_array.shape[0]
            break
        
        if not answer.isdigit():
            print("Please enter a number")
            continue
        
        else:
            int_answer = int(answer)
            break
                  
    if (not picture_in_RGB):
        input_shape = (1, int_answer, int_answer)

    else:
        input_shape = (3, int_answer, int_answer)
	
    return picture_in_RGB, input_shape

def get_package_folder(current_path, create_new_folder):
    
    while True:
        
        if create_new_folder:
            answer = input("\nWould you use an existance Package ?\n").strip().lower()
        else:
            answer = "y"
            
        if answer == "yes" or answer == "y":
            
            folders = sorted(os.listdir(current_path))
            folder = [f for f in folders if os.path.isdir(os.path.join(current_path, f)) and "Package" in f]
            
            # Afficher les fichiers avec un numéro
            print("\nSélectionnez un fichier en entrant son numéro :")
            for idx, file in enumerate(folder, start=1):
                print(f"{idx}. {file}")
            
            # Demander à l'utilisateur de choisir
            while True:
                choice = input("Entrez le numéro du fichier : ")
                if not choice.isdigit():
                    print("❌ Veuillez entrer un numéro valide.")
                    continue

                choice = int(choice)
                if 1 <= choice <= len(folder):
                    selected_folder = folder[choice - 1]
                    print(f"\n✅ Vous avez sélectionné : {selected_folder}")
                    return selected_folder

                elif choice == 0:
                    exit(0)

                else:
                    print(f"❌ Numéro invalide. Veuillez choisir entre 1 et {len(folder)}.")
        
        else:
            new_folder =  input("Enter name of the new package: \n")
            return new_folder


def get_dataset_config(module_dir, hyperparams, dataset):
    
    answer = ask_yes_no("\nIs your dataset a single .npz file?")

    if answer:
        return handle_single_npz(module_dir, hyperparams, dataset)
    else:
        return handle_dataset_folder(module_dir, hyperparams, dataset)


def handle_single_npz(module_dir, hyperparams, dataset):
    

    data = manage_data()
    
    split_mode = data["split_mode"]
    data_name = data["selected_file"]
    
    dir_name = transform_name(data_name)
    module_dir = os.path.join(module_dir, dir_name)     
    
    if not split_mode:
        
        X = split_mode = data["X"]
        y = split_mode = data["y"]
                            
        dataset_size = int(len(y) * (1 - hyperparams.contamination))
        dataset.completion_value(dataset_size, None, None, hyperparams.batch_size, True)
    
        input_shape, output_shape =  get_data_shape(X, y)
        X_train, y_train, X_test, y_test, class_to_idx = preprocessing(X, y, hyperparams, dataset)
        
    else:
        
        X_train = split_mode = data["x_train"]
        y_train = split_mode = data["y_train"]
        X_test = split_mode = data["x_test"]
        y_test = split_mode = data["y_test"]
                    
        dataset_size = int((len(y_train) + len(y_test)))
        dataset.completion_value(dataset_size, len(y_train), len(y_test), hyperparams.batch_size, False)
    
        input_shape, output_shape =  get_data_shape(X_train, y_train.flatten())
        class_to_idx = {int(label): idx for idx, label in enumerate(np.unique(y_train.flatten()))}
            
    if X_train.ndim == 4 and X_test.ndim == 4:
        X_train = X_train.transpose((0, 3, 1, 2))
        X_test = X_test.transpose((0, 3, 1, 2))
                
    dataset.class_to_idx = class_to_idx
    dataset.print_info()
    
    return {
        "full_size": True,
        
        "module_dir": module_dir,
        "input_shape": input_shape,
        "output_shape": output_shape,
                
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        
        "class_to_idx": class_to_idx,
    }
    

def print_info_dataset(train_files, train_labels, test_files, test_labels, class_to_idx):
    
    print("")
    print("NB Train file: ", len(train_files))
    print("NB Train label: ", np.unique(train_labels, return_counts=True))
    print("NB Test file: ", len(test_files))
    print("NB Test label: ", np.unique(test_labels, return_counts=True))

           
def handle_dataset_folder(module_dir, hyperparams, dataset):
    
    split_mode = ask_yes_no("Dataset already split train/test?")

    if split_mode:
        #Train Set
        path_train_file = input("\nEnter the path for the train file:\n").strip()
        train_files, train_labels, class_to_idx = load_file_paths(path_train_file, None)

        #Test Set
        path_test_file = input("\nEnter the path for the test file:\n").strip()
        test_files, test_labels, _ = load_file_paths(path_test_file, class_to_idx)
    
    else:
        path_dataset = input("\nEnter the path for the dataset:\n").strip()
        files, labels, class_to_idx = load_file_paths(path_dataset, None)
                    
        train_files, test_files, train_labels, test_labels = train_test_split(
            files,
            labels,
            test_size=dataset.ratio_test,
            stratify=labels,
            random_state=42
        )
        
    print_info_dataset(train_files, train_labels, test_files, test_labels, class_to_idx)
    
    picture_in_RGB, input_shape = change_dim_picture(train_files[0], hyperparams.input_shape)
    
    dataset_size = len(train_files) + len(test_files)
    dataset.class_to_idx = class_to_idx
    dataset.completion_value(dataset_size, len(train_files), len(test_files), hyperparams.batch_size, False)
    dataset.print_info()
    
    return {
        "full_size": False,
        
        "module_dir": module_dir,
        "input_shape": input_shape,
        "output_shape": len(class_to_idx.keys()),
        
        "train_files": train_files,
        "train_labels": train_labels,
        "test_files": test_files,
        "test_labels": test_labels,
        
        "class_to_idx": class_to_idx,
        
        "picture_in_RGB": picture_in_RGB
    }

         
def run_training_pipeline(module_dir, hyperparams, structure, loss_metric, output_layer, optimizer, transition_layer, dataset):

    mode = set_mode()

    if mode in {5}:
        
        data_name = get_package_folder(module_dir, False)
        module_dir = os.path.join(module_dir, data_name)
        
        model_name, path_model = select_model(module_dir, FOLDER_NAME_LOGBOOK)   
        path = os.path.dirname(os.path.dirname(path_model))
        path_data = os.path.join(path, FOLDER_NAME_LOGBOOK, model_name + ".json")

        print("\nThis file would be delete:")
        print(path_model)
        print(path_data)
        
        answer = ask_yes_no("\nDelete this model?")
        
        if not answer:
            print("Cancel, the model is not deleted")
            return
        
        os.remove(path_model)
        os.remove(path_data)
        print(f"\nModel deleted: {model_name}")
        return
    
    if mode in {4}:
        # === Load Model ===
        data_name = get_package_folder(module_dir, False)
        module_dir = os.path.join(module_dir, data_name)

        model_name, _ = select_model(module_dir, FOLDER_NAME_LOGBOOK)
        model, hyperparams, structure, performance, dataset, metadata_old = load_model(
            module_dir, model_name, None
        )

        print("")
        show_all_info_model(hyperparams, structure, performance, dataset, metadata_old)

        # === Chose user dataset ===
        use_dataset = ask_yes_no("\nWould you use a dataset ?")

        if use_dataset:
            dataset_config = get_dataset_config(module_dir, hyperparams, dataset)
            dataset_full_size = dataset_config["full_size"]

        else:
            # === Loading a single image ===
            file_path = input("\nEnter the path to load your picture:\n").strip().strip('"')

            if file_path == "0":
                exit(0)

            img_shape = hyperparams.input_shape
            img_size = (img_shape[1], img_shape[2])

            # Read picture
            mode = 'RGB' if img_shape[0] == 3 else 'L'
            img = Image.open(file_path).convert(mode)

            # Resize with padding
            img = ImageOps.pad(img, img_size, method=Image.Resampling.LANCZOS)

            # Normalization
            img_array = np.array(img) / np.max(img)

            # Formatting (batch + channels)
            if img_array.ndim == 2:
                img_array = img_array[None, None, :, :]  # (1, 1, H, W)

            elif img_array.ndim == 3:
                img_array = np.transpose(img_array, (2, 0, 1))  # (C, H, W)
                img_array = img_array[None, :, :, :]  # (1, C, H, W)

            else:
                raise ValueError(f"Unexpected image shape: {img_array.shape}")

            display_kernel_and_biais(img_array, None, model.cnn_model)
            exit(0)

        # === Test data preparation ===
        if dataset_full_size:
            X_test = dataset_config["X_test"]
            y_final = dataset_config["y_test"]

        else:
            test_files = dataset_config["test_files"]
            test_labels = dataset_config["test_labels"]
            picture_in_RGB = dataset_config["picture_in_RGB"]

            img_size = (hyperparams.input_shape[1], hyperparams.input_shape[2])
            batch_size = hyperparams.batch_size

            test_gen = batch_generator(
                test_files,
                test_labels,
                batch_size,
                img_size,
                True,
                picture_in_RGB
            )

            X_test, y_final = next(test_gen)

        # === Final Display ===
        display_kernel_and_biais(X_test, y_final, model.cnn_model)
        exit(0)

    if mode in {1}:

        # ============================
        #     INITIALISATION CNN
        # ============================
        dataset_config = get_dataset_config(module_dir, hyperparams, dataset)
        
        input_shape = dataset_config["input_shape"]
        output_shape = dataset_config["output_shape"]
        class_to_idx = dataset_config["class_to_idx"]
        dataset_full_size = dataset_config["full_size"]
                
        hyperparams.add_shape(input_shape, output_shape)
        hyperparams.check_support()
    
        model = FullModel(hyperparams, structure, loss_metric, output_layer, optimizer, transition_layer)
        metadata_old = None

        data_name = get_package_folder(module_dir, True)
        module_dir = os.path.join(module_dir, data_name)
        
    else:

        # ============================
        #       SELECT A MODEL
        # ============================

        # Chargement du modele existant
        data_name = get_package_folder(module_dir, False)
        module_dir = os.path.join(module_dir, data_name)
        model_name, _ = select_model(module_dir, FOLDER_NAME_LOGBOOK)
        
        if mode in {3}:
            hyperparams = None
            
        model, hyperparams, _, _, _, metadata_old = load_model(module_dir, model_name, hyperparams)
        
        dataset_config = get_dataset_config(module_dir, hyperparams, dataset)
        
        input_shape = dataset_config["input_shape"]
        output_shape = dataset_config["output_shape"]
        class_to_idx = dataset_config["class_to_idx"]
        dataset_full_size = dataset_config["full_size"]
        
    if mode in {1, 2}:
        # ============================
        #       TRAINNING
        # ============================

        # Entraînement d'un nouveau modèle

        #For One-Hot Encoder
        if model.loss_metric.class_ == "CrossEntropyLoss":
            num_classes = len(class_to_idx)
        else:
            num_classes = None
                
        if dataset_full_size:
            
            X_train = dataset_config["X_train"]
            y_train = dataset_config["y_train"]
            X_test = dataset_config["X_test"]
            y_test = dataset_config["y_test"]
            
            data_train, data_test, elapsed_time_minutes = training_full_data(
                model, 
                hyperparams, dataset,
                X_train, y_train, 
                X_test, y_test,
                num_classes)

        else:
            
            train_files = dataset_config["train_files"]
            train_labels = dataset_config["train_labels"]
            test_files = dataset_config["test_files"]
            test_labels = dataset_config["test_labels"]
            picture_in_RGB = dataset_config["picture_in_RGB"]
                    
            data_train, data_test, elapsed_time_minutes = training_batch_data(
                model, 
                hyperparams, dataset,
                train_files, train_labels, 
                test_files, test_labels, 
                picture_in_RGB, num_classes)
        
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
        performance["accuracy_ratio"] = data_test['accu'][-1] / data_train['accu'][-1]
        performance["overfitting_indicator"] = data_test['loss'][-1] - data_train['loss'][-1]
        
        save_model_parameters(module_dir, name_model, model)

        save_model_configuration(mode, 
                    hyperparams, performance, dataset, structure,
                    elapsed_time_minutes,
                    metadata, metadata_old,
                    module_dir)
        
    #______________________________________________________________#
    while(1):
        
        str_answer = input("\nHow many test do you want to do ?\n")

        if not str_answer.isdigit():
            print("Please enter a number")
            continue
        
        else:
            break
    
    nb_test = int(str_answer)
        
    if (nb_test == 0):
        print("Exit")
        exit(0)
        
    if dataset_full_size:
        X_test = dataset_config["X_test"]
        y_test = dataset_config["y_test"]
                                
        display_first_picture(model, X_test, y_test, class_to_idx)
        display_dataset(model, X_test, y_test, nb_test, class_to_idx)
        
    else:
        test_files = dataset_config["test_files"]
        test_labels = dataset_config["test_labels"]
        picture_in_RGB = dataset_config["picture_in_RGB"]
            
        img_size = (hyperparams.input_shape[1], hyperparams.input_shape[2])
        batch_size = hyperparams.batch_size
        test_gen = batch_generator(test_files, test_labels, batch_size, img_size, True, picture_in_RGB)
        
        X_test, y_final = next(test_gen)                
        display_first_picture(model, X_test, y_final, class_to_idx)
        display_dataset(model, X_test, y_final, nb_test, class_to_idx)