# Standard library
import os
from datetime import datetime

# Third-party
import numpy as np
from PIL import Image, ImageOps

# Local imports
from .Constante import FOLDER_NAME_LOGBOOK

# System
from .Set_mode import set_mode
from .User_Input import ask_yes_no

# Data / file management
from .Manage_file import (
    file_management,
    select_model,
    load_model,
    save_model_parameters,
)
from .Manage_logbook import (
    save_model_configuration,
    show_all_info_model,
)
from .Dataset_builder import select_type_dataset, get_name_package_folder

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


def delete_mode(module_dir):

    data_name = get_name_package_folder(module_dir, False)
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


def load_model_for_test(module_dir):
    data_name = get_name_package_folder(module_dir, False)
    module_dir = os.path.join(module_dir, data_name)

    model_name, _ = select_model(module_dir, FOLDER_NAME_LOGBOOK)

    return load_model(module_dir, model_name, None), module_dir


def display_model_info(hyperparams, structure, performance, dataset, metadata):
    print("")
    show_all_info_model(hyperparams, structure, performance, dataset, metadata)


def load_single_image_for_test(file_path, input_shape):
    img_size = (input_shape[1], input_shape[2])
    mode = 'RGB' if input_shape[0] == 3 else 'L'

    img = Image.open(file_path).convert(mode)
    img = ImageOps.pad(img, img_size, method=Image.Resampling.LANCZOS)

    img_array = np.array(img) / np.max(img)

    if img_array.ndim == 2:
        img_array = img_array[None, None, :, :]
    elif img_array.ndim == 3:
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = img_array[None, :, :, :]
    else:
        raise ValueError(f"Unexpected image shape: {img_array.shape}")

    return img_array


def handle_test_input(module_dir, hyperparams, dataset):
    use_dataset = ask_yes_no("\nWould you use a dataset ?")

    if use_dataset:
        return "dataset", select_type_dataset(module_dir, hyperparams, dataset)

    file_path = input("\nEnter the path to load your picture:\n").strip().strip('"')

    if file_path == "0":
        exit(0)

    return "image", load_single_image_for_test(file_path, hyperparams.input_shape)


def prepare_test_data(dataset_config, hyperparams):
    if dataset_config["full_size"]:
        return dataset_config["X_test"], dataset_config["y_test"]

    test_files = dataset_config["test_files"]
    test_labels = dataset_config["test_labels"]
    picture_in_RGB = dataset_config["picture_in_RGB"]

    img_size = (hyperparams.input_shape[1], hyperparams.input_shape[2])

    test_gen = batch_generator(
        test_files,
        test_labels,
        hyperparams.batch_size,
        img_size,
        True,
        picture_in_RGB
    )

    return next(test_gen)


def exam_mode(module_dir):

    # Load model
    (model, hyperparams, structure, performance, dataset, metadata), module_dir = load_model_for_test(module_dir)

    display_model_info(hyperparams, structure, performance, dataset, metadata)

    # User choice
    input_type, data = handle_test_input(module_dir, hyperparams, dataset)

    if input_type == "image":
        display_kernel_and_biais(data, None, model.cnn_model, dataset)
        exit(0)

    # Dataset case
    X_test, y_final = prepare_test_data(data, hyperparams)

    display_kernel_and_biais(X_test, y_final, model.cnn_model, dataset)
    exit(0)


def save_model(module_dir, model, hyperparams, dataset, structure, elapsed_time_minutes, metadata_old, mode, data_test, data_train):
    
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


def new_model_mode(module_dir, hyperparams, structure, loss_metric, output_layer, optimizer, transition_layer, dataset):
      
    dataset_config = select_type_dataset(module_dir, hyperparams, dataset)
        
    input_shape = dataset_config["input_shape"]
    output_shape = dataset_config["output_shape"]
    class_to_idx = dataset_config["class_to_idx"]
    dataset_full_size = dataset_config["full_size"]
            
    hyperparams.add_shape(input_shape, output_shape)
    hyperparams.check_support()

    model = FullModel(hyperparams, structure, loss_metric, output_layer, optimizer, transition_layer)
    metadata_old = None

    data_name = get_name_package_folder(module_dir, True)
    module_dir = os.path.join(module_dir, data_name)
    
    return module_dir, model, metadata_old, class_to_idx, dataset_full_size, dataset_config


def fine_tuning_model(module_dir, hyperparams, mode, dataset):
     
    # Chargement du modele existant
    data_name = get_name_package_folder(module_dir, False)
    module_dir = os.path.join(module_dir, data_name)
    model_name, _ = select_model(module_dir, FOLDER_NAME_LOGBOOK)
    
    if mode in {3}:
        hyperparams = None
    
    model, hyperparams, _, _, _, metadata_old = load_model(module_dir, model_name, hyperparams)

    dataset_config = select_type_dataset(module_dir, hyperparams, dataset)
    
    class_to_idx = dataset_config["class_to_idx"]
    dataset_full_size = dataset_config["full_size"]
    
    return module_dir, model, hyperparams, metadata_old, class_to_idx, dataset_full_size, dataset_config


def training(model, hyperparams, dataset, class_to_idx, dataset_config, dataset_full_size):
    
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
        
    return data_train, data_test, elapsed_time_minutes


def final_verification(model, hyperparams, class_to_idx, dataset_config, dataset_full_size):
    
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


def run_training_pipeline(module_dir, hyperparams, structure, loss_metric, output_layer, optimizer, transition_layer, dataset):

    mode = set_mode()

    if mode in {5}:
        delete_mode(module_dir)
        
    if mode in {4}:
        exam_mode(module_dir)
        
    if mode in {1}:
        module_dir, model, metadata_old, class_to_idx, dataset_full_size, dataset_config = new_model_mode(module_dir, hyperparams, structure, loss_metric, output_layer, optimizer, transition_layer, dataset)
    
    else:
        module_dir, model, hyperparams, metadata_old, class_to_idx, dataset_full_size, dataset_config = fine_tuning_model(module_dir, hyperparams,mode, dataset)
        
    if mode in {1, 2}:
        data_train, data_test, elapsed_time_minutes = training(model, hyperparams, dataset, class_to_idx, dataset_config, dataset_full_size)
        
        save_model(module_dir, model, hyperparams, dataset, structure, elapsed_time_minutes, metadata_old, mode, data_test, data_train)
        
        
    final_verification(model, hyperparams, class_to_idx, dataset_config, dataset_full_size)
