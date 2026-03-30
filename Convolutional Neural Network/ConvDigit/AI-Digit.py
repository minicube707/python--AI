
import os
from datetime import datetime

#System
from System.Set_mode import set_mode
from System.Manage_data import manage_data
from System.Manage_file import file_management, select_model, load_model, save_model_parameters, transform_name
from System.Manage_logbook import save_model_configuration, show_all_info_model

#IA
from System.Preprocessing import preprocessing, get_data_shape
from System.Display_parametre_CNN import display_kernel_and_biais, display_first_picture, display_dataset

from System.Evaluation_Metric import CrossEntropyLoss
from System.Mathematical_function import Softmax
from System.FullModel import FullModel
from System.Optimizer import Adam
from System.Trainning import trainnig

from System.Dataclasses import Hyperparams, Dataset

module_dir = os.path.dirname(__file__)
os.chdir(module_dir)


#Load data
X, y, data_name = manage_data()
dir_name = transform_name(data_name)
module_dir = os.path.join(module_dir, dir_name)

# Mode d'exécution (1: train + save, 2: load + save, 3: load)
mode = set_mode()

if mode in {4}:
    model_name = select_model(module_dir, "LogBook")
    model, hyperparams, structure, performance, dataset, metadata_old = load_model(module_dir, model_name)

    print("")
    show_all_info_model(hyperparams, structure, performance, dataset, metadata_old)
    display_kernel_and_biais(X, y, model.cnn_model)
    exit(0)

elif (X is None and y is None):
    print("Error: Data not load")
    exit(0)


input_shape, output_shape =  get_data_shape(X, y)

# ============================
#         PARAMÈTRES
# ============================

hyperparams = {
    "nb_epoch": 20,
    "batch_size": 64,

    "lr": 0.001,
    "beta1": 0.9,
    "beta2": 0.999,
    "alpha" : 0.001,

    "contamination" : 0.1,

    "padding_mode": "auto"
}

dataset = {
    "validation_size" : 1_000,
    "validation_frequency" : 100,

    "dataset_size": -1,
    "ratio_test" : 0.2,
}

# Structure CNN : (kernel size, stride, padding, nb kernels, type_layer, activations function)
structure_CNN = {
    "1": (5, 1, 0, 32, "conv", "leaky relu", 0.0),
    "2": (2, 2, 0, 1, "pool", "max", 0.0),
    "3": (3, 1, 0, 64, "conv", "leaky relu", 0.1),
    "4": (2, 2, 0, 1, "pool", "max", 0.0),
    "5": (3, 1, 0, 64, "conv", "leaky relu", 0.1)
}

# Structure DNN : (number of neurone, activations function) 
structure_DNN = {
    "1": (124, "leaky relu", 0.2),
    "2": (64, "leaky relu", 0.2),
    "3": (64, "leaky relu", 0.2),
    "4": (0,  "leaky relu", 0.2)
}

hyperparams = Hyperparams(**hyperparams)

loss_metric = CrossEntropyLoss()
output_layer = Softmax()    
optimizer = Adam(hyperparams)

hyperparams.add_shape(input_shape, output_shape)
hyperparams.add_training_parameters(loss_metric, output_layer, optimizer)

dataset = Dataset(**dataset)
structure = (structure_CNN, structure_DNN)

if mode in {1}:

    # ============================
    #     INITIALISATION CNN
    # ============================
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


# ============================
#     PRÉTRAITEMENT DONNÉES
# ============================

dataset.completion_value(y)
dataset.print_info()
X_train, y_train, X_test, y_test, transformer = preprocessing(X, y, hyperparams, dataset)

# pour ton modèle CNN NumPy (channels first)
X_train = X_train.transpose(0, 3, 1, 2)  # (50000, 3, 32, 32)
X_test = X_test.transpose(0, 3, 1, 2)

if mode in {1, 2}:
    # ============================
    #       TRAINNING
    # ============================

    # Entraînement d'un nouveau modèle
    data_test, elapsed_time_minutes = trainnig(model, X_train, y_train, X_test, y_test, hyperparams, dataset)
    
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

y_final = transformer.inverse_transform(y_test)

nb_test = 10
display_first_picture(model, X_test, y_final)
display_dataset(model, X_test, y_final, nb_test)


