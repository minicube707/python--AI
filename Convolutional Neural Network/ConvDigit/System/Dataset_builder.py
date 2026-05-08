
# Standard library
import os

# Third-party
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image

# System
from .User_Input import ask_yes_no

# Data / file management
from .Manage_data import manage_data, load_file_paths
from .Manage_file import transform_name

# Preprocessing
from .Preprocessing import preprocessing, get_data_shape


def configure_image_shape(train_picture, input_shape):

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

     
    if np.ndim(img_array) == 3 and ask_yes_no("\nDo you want to load the images in color?"):
        print("Images will be loaded in color (RGB).")
        picture_in_RGB = True
    
    else:
        print("Images will be loaded in grayscale (black and white).")
        picture_in_RGB = False


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


def select_type_dataset(module_dir, hyperparams, dataset):
    
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
        path_train_file = input("\nEnter the path for the train file:\n").strip().strip('"')
        train_files, train_labels, class_to_idx = load_file_paths(path_train_file, None)

        #Test Set
        path_test_file = input("\nEnter the path for the test file:\n").strip().strip('"')
        test_files, test_labels, _ = load_file_paths(path_test_file, class_to_idx)
    
    else:
        path_dataset = input("\nEnter the path for the dataset:\n").strip().strip('"')
        files, labels, class_to_idx = load_file_paths(path_dataset, None)
                    
        train_files, test_files, train_labels, test_labels = train_test_split(
            files,
            labels,
            test_size=dataset.ratio_test,
            stratify=labels,
            random_state=42
        )
        
    print_info_dataset(train_files, train_labels, test_files, test_labels, class_to_idx)
    
    picture_in_RGB, input_shape = configure_image_shape(train_files[0], hyperparams.input_shape)
    
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


def get_name_package_folder(current_path, create_new_folder):
                      
    if not create_new_folder or ask_yes_no("\nWould you use an existance Package ?"):

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