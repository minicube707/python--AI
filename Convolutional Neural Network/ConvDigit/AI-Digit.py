
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from datetime import datetime
from pathlib import Path

#System
from System.Set_mode import set_mode
from System.Manage_data import manage_data
from System.Manage_file import file_management, select_model, load_model, save_model, transform_name
from System.Manage_logbook import fill_information, add_model

#IA
from System.Preprocessing import preprocessing, handle_key, show_information_setting
from System.Display_parametre_CNN import display_kernel_and_biais

from System.Evaluation_Metric import CrossEntropyLoss
from System.Mathematical_function import Softmax
from System.FullModel import FullModel
from System.Optimizer import Adam
from System.Trainning import trainnig

module_dir = os.path.dirname(__file__)
os.chdir(module_dir)


#Data_Digit
X, y, data_name = manage_data()
dir_name = transform_name(data_name)
module_dir = os.path.join(module_dir, dir_name)


# ============================
#         PARAMÈTRES
# ============================

nb_iteration = 1
validation_size = 1_000
ratio_test = 0.2
validation_frequency = 100
dataset_size = 60_000
batch_size = 32
contamination = 0.1

# Paramètres d'apprentissage
lr = 0.001
beta1 = 0.9
beta2 = 0.999
alpha = 0.001


# ============================
#     PRÉTRAITEMENT DONNÉES
# ============================

# Forme d'entrée (nb_data, hauteur, largeur)
if X.ndim == 3:
    input_shape = (1, X.shape[1],X.shape[2])
    print("X has 3 dimensions")
    print("Input shappe: ", input_shape)

# Forme d'entrée (nb_data, hauteur, largeur, cannaux)
elif X.ndim == 4:
    _, _, _, channel = X.shape
    input_shape = (channel, X.shape[1], X.shape[2])
    print("X has 4 dimensions")
    print("Input shappe: ", input_shape)

else:
    raise ValueError(f"Unsupported input dimension: {X.ndim}")
    
X_train, y_train, X_test, y_test, transformer = preprocessing(X, y, dataset_size, ratio_test, contamination)

if (validation_size > len(y_test)):
    validation_size = len(y_test)

if (validation_frequency == -1):
    validation_frequency = X_train.shape[0]

show_information_setting(nb_iteration, lr, beta1, beta2, alpha, validation_size, ratio_test, dataset_size)


# Mode d'exécution (1: train + save, 2: load + save, 3: load)
mode = set_mode()

# if mode in {4}:
#     model, model_info = select_model(module_dir, "LogBook/model_logbook.csv")
#     parametres_CNN, dimensions_CNN, parametres_DNN, dimensions_DNN = load_model(module_dir, model)

#     print("")
#     show_all_info_model(model_info)
#     display_kernel_and_biais(X, y, model)
#     exit(0)

# elif (X is None and y is None):
#     print("Error: Data not load")
#     exit(0)

# pour ton modèle CNN NumPy (channels first)
X_train = X_train.transpose(0, 3, 1, 2)  # (50000, 3, 32, 32)
X_test  = X_test.transpose(0, 3, 1, 2)

if mode in {1}:


    # ============================
    #     INITIALISATION CNN
    # ============================

    # Structure CNN : (kernel_size, stride, padding, nb_kernels, type_layer, activation)
    dimensions_CNN = {
        "1": (5, 1, 0, 32, "conv", "leaky relu", 0.0),
        "2": (2, 2, 0, 1, "pool", "max", 0.0),
        "3": (3, 1, 0, 64, "conv", "leaky relu", 0.1),
        "4": (2, 2, 0, 1, "pool", "max",0.0),
        "5": (3, 1, 0, 64, "conv", "leaky relu", 0.1)
    }
    
    # Structure DNN : (number of neurone, activations) 
    dimensions_DNN = {
        "1": (124, "leaky relu", 0.2),
        "2": (64, "leaky relu", 0.2),
        "3": (64, "leaky relu", 0.2),
        "4": (0,  "leaky relu", 0.2)
    }

    

    #Initialisation
    padding_mode = "auto"               # Mode de padding : 'auto' = calcul automatique
    loss_metric = CrossEntropyLoss()
    output_layer = Softmax()    
    optimizer = Adam(lr, beta1, beta2)
    model = FullModel(X_train, y_train, 
                      dimensions_CNN, dimensions_DNN, 
                      optimizer, loss_metric, output_layer, 
                      input_shape, alpha, padding_mode)


 
else:
    # ============================
    #       SELECT A MODEL
    # ============================

    # Chargement du modele existant
    model, model_info = select_model(module_dir, "LogBook/model_logbook.csv")
    parametres_CNN, dimensions_CNN, parametres_DNN, dimensions_DNN = load_model(module_dir, model)

if mode in {1, 2}:
    # ============================
    #       TRAINNING
    # ============================

    # Entraînement d'un nouveau modèle
    trainnig(model, X_train, y_train, X_test, y_test, batch_size, nb_iteration, validation_size, validation_frequency)
    
    # ============================
    #          SAVE
    # ============================

    # Sauvegarde du meilleur modèle entraîné ou chargé
    # name_model = file_management(test_accu, test_conf)
    # print(name_model)
    # save_model(module_dir, name_model, (parametres_CNN, dimensions_CNN, parametres_DNN, dimensions_DNN))

    # date = datetime.today()
    # date = date.strftime('%d/%m/%Y')
 
    # if mode in {1}:
    #     nb_epoch = nb_iteration
    #     training_time = elapsed_time_minutes
    #     baseline_mode = "X"
    #     nb_fine_tunning = 0

    #     #CNN
    #     str_size_CNN = ','.join(str(v[0]) for v in dimensions_CNN.values() if v[4] == 'kernel')
    #     str_nb_kernel_CNN = ','.join(str(v[3]) for v in dimensions_CNN.values() if v[4] == 'kernel')
    #     str_function_CNN = ','.join(str(v[5]) for v in dimensions_CNN.values() if v[4] == 'kernel')

    #     #DNN
    #     str_size_DNN = ','.join(str(v[0]) for v in dimensions_DNN.values())
    #     str_function_DNN = ','.join(str(v[1]) for v in dimensions_DNN.values())

    # else:
    #     nb_epoch = float(model_info["nb_epoch"]) + nb_iteration
    #     training_time = float(model_info["training_time_(min)"]) + elapsed_time_minutes
    #     baseline_mode = model_info["name"]
    #     nb_fine_tunning = float(model_info["Number_fine_tunning"]) + 1

    #     #CNN
    #     str_size_CNN = model_info["kernel_size"]
    #     str_nb_kernel_CNN = model_info["kernel_number"]
    #     str_function_CNN = model_info["activation_function_CNN"]

    #     #DNN
    #     str_size_DNN = model_info["neurons_number"]
    #     str_function_DNN = model_info["activation_function_DNN"]

    # new_log =  fill_information(name_model, date, training_time,
    #                 nb_epoch,  max_attempts, min_confidence_score, beta1, beta2, alpha,
    #                 test_loss, test_accu, test_conf, 
    #                 learning_rate_CNN, str_size_CNN, str_nb_kernel_CNN, str_function_CNN,
    #                 learning_rate_DNN, str_size_DNN, str_function_DNN,
    #                 len(y_train), len(y_test), 
    #                 baseline_mode, nb_fine_tunning, validation_size,
    #                 validation_frequency, ratio_test, dataset_size)
    
    # add_model(new_log, os.path.join(module_dir, "LogBook"), "model_logbook.csv")

#______________________________________________________________#

y_final = transformer.inverse_transform(y_test)

#Affichage des 15 premières images
fig = plt.figure(figsize=(16,8))
fig.canvas.mpl_connect('key_press_event', handle_key)  # Active la détection de la touche
for i in range(1,16):

    # Prédiction des probabilités avec softmax
    
    if X_test[i].ndim == 3:
        y_pred = model.forward_propagation(X_test[i][None, ...], False)
    elif X_test[i].ndim == 2:
        y_pred = model.forward_propagation(X_test[i][None, None, ...], False)

    pred = np.argmax(y_pred)
    porcent = np.max(y_pred)

    plt.subplot(4,5, i)
    plt.imshow(X_test[i], cmap="gray")

    plt.title(f"Value:{y_final[i]} Predict:{pred}  ({np.round(porcent, 2)}%)")
    plt.tight_layout()
    plt.axis("off")
plt.show()  

nb_test = 10
print("")
for i in range(nb_test):
   
    index = input(f"Please enter a number between 1 and {X_test.shape[0]}: ")
   
   # Check if input is empty or invalid
    if not index.strip():  
        print("❌ Please enter a valid number.")
        continue
    
    try:
        index = int(index)
    except ValueError:
        print("❌ Invalid input. Please enter an integer.")
        continue

    # Exit condition
    if index < 0:
        print("Exiting")
        break
    
    # Prédiction des probabilités avec softmax
    if X_test[index].ndim == 3:
        y_pred = model.forward_propagation(X_test[index][None, ...], False)[0]
    elif X_test[index].ndim == 2:
        y_pred = model.forward_propagation(X_test[index][None, None, ...], False)[0]

    pred = np.argmax(y_pred)
    porcent = np.max(y_pred)

    # Création de la figure avec 2 sous-graphiques (image + histogramme)
    fig, axs = plt.subplots(2, 1, figsize=(5, 7), gridspec_kw={'height_ratios': [3, 1]})
    fig.canvas.mpl_connect('key_press_event', handle_key)  # Connecte l'événement clavier
    
    axs[0].imshow(X_test[index], cmap="gray")
    axs[0].set_title(f"Value:{y_final[index]} Predict:{pred} ({np.round(porcent, 2)}%)")
    axs[0].axis("off")

    # Affichage de l'histogramme des probabilités
    axs[1].bar(range(len(y_pred)), y_pred, color="blue")
    axs[1].set_xticks(range(len(y_pred)))
    axs[1].set_xlabel("Classes")
    axs[1].set_ylabel("Probability")
    axs[1].set_ylim(0, 1)

    # Ajout des lignes horizontales tous les 0.1
    axs[1].set_yticks([i / 10 for i in range(11)])  # De 0.0 à 1.0 par pas de 0.1
    axs[1].grid(axis='y', linestyle='--', linewidth=0.5, color='red')  # Ligne fine et discrète

    plt.tight_layout()
    plt.show()
