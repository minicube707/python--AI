
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

from System.Set_mode import set_mode
from System.Preprocessing import preprocessing
from System.Deep_Neuron_Network import deep_neural_network, foward_propagation, softmax
from System.Manage_file import file_management, select_model, load_model, save_model, transform_name, get_hidden_layers
from System.Manage_data import manage_data
from System.Manage_logbook import fill_information, add_model

module_dir = os.path.dirname(__file__)
os.chdir(module_dir)


#Data_Digit
X, y, data_name = manage_data()
dir_name = transform_name(data_name)
module_dir = os.path.join(module_dir, dir_name)

# Forme d'entrée (canaux, hauteur, largeur)
if X.ndim == 3:
    _, side, _ = X.shape
    input_shape = X.shape
    X = X.reshape(X.shape[0], -1)

elif X.ndim == 2:
    n_samples, n_features = X.shape
    side = int(np.sqrt(n_features))
    input_shape = (1, side, side)

else:
    raise ValueError(f"Unsupported input dimension: {X.ndim}")


# ============================
#     PRÉTRAITEMENT DONNÉES
# ============================
X_train, y_train, X_test, y_test, transformer = preprocessing(X, y, input_shape)


# ============================
#         PARAMÈTRES
# ============================
learning_rate = 0.01
n_iteration = 1


# Mode d'exécution (1: train + save, 2: load + save, 3: load)
mode = set_mode()

if mode in {1}:

    # ============================
    #     INITIALISATION DNN
    # ============================

    hidden_layer = (32, 32, 32)

else:
    # ============================
    #       SELECT A MODEL
    # ============================

    # Chargement du modele existant
    model, model_info = select_model(module_dir, "LogBook/model_logbook.csv")
    parametres_DNN, dimensions_DNN = load_model(module_dir, model) 
    hidden_layer = get_hidden_layers(model)

print("\nDetail DNN")
print(hidden_layer)

if mode in {1, 2}:
    # ============================
    #       TRAINNING
    # ============================

    # Entraînement d'un nouveau modèle 
    parametres, dimensions, test_accu = deep_neural_network(X_train, y_train, X_test, y_test, hidden_layer, learning_rate, n_iteration)


    # ============================
    #          SAVE
    # ============================

    name_model = file_management(hidden_layer, test_accu)
    print(name_model)
    save_model(module_dir, name_model, parametres)

    date = datetime.today()
    date = date.strftime('%d/%m/%Y')
 
    if mode in {1}:
        nb_epoch = n_iteration
        baseline_mode = "X"
        nb_fine_tunning = 0

    else:
        nb_epoch = float(model_info["nb_epoch"]) + n_iteration
        baseline_mode = model_info["name"]
        nb_fine_tunning = float(model_info["Number_fine_tunning"]) + 1

    new_log =  fill_information(name_model, date,
                        nb_epoch, test_accu, 
                        learning_rate,
                        len(y_train), len(y_test), 
                        baseline_mode, nb_fine_tunning)
    
    add_model(new_log, os.path.join(module_dir, "LogBook"), "model_logbook.csv")


#______________________________________________________________#
X_final = X_test.T
y_final = y_test.T
y_final = transformer.inverse_transform(y_final)

C = len(parametres) // 2
activation = foward_propagation(X_final.T, parametres)


#Affichage des 15 premières images
plt.figure(figsize=(16,8))
for i in range(1,16):
    # Prédiction des probabilités avec softmax
    probabilities = softmax(activation["A" + str(C)].T)[i,:]
    pred = np.argmax(probabilities)
    porcent = np.max(probabilities)

    plt.subplot(4,5, i)
    plt.imshow(X_final.reshape((X_final.shape[0], side, side))[i], cmap="gray")
    plt.title(f"Value:{y_final[i]} Predict:{pred}  ({np.round(porcent, 2)}%)")
    plt.tight_layout()
    plt.axis("off")
plt.show()  

nb_test = 10
print("")
for i in range(nb_test):
    index = int(input(f"Please enter a number between 1 and {X_final.shape[0]}: "))
    if index == -1:
        break
    
    # Prédiction des probabilités avec softmax
    probabilities = softmax(activation["A" + str(C)].T)[index,:]
    pred = np.argmax(probabilities)
    porcent = np.max(probabilities)

    # Création de la figure avec 2 sous-graphiques (image + histogramme)
    fig, axs = plt.subplots(2, 1, figsize=(5, 7), gridspec_kw={'height_ratios': [3, 1]})

    # Affichage de l'image
    axs[0].imshow(X_final.reshape((X_final.shape[0], side, side))[index], cmap="gray")
    axs[0].set_title(f"Value:{y_final[index]} Predict:{pred} ({np.round(porcent, 2)}%)")
    axs[0].axis("off")

    # Affichage de l'histogramme des probabilités
    axs[1].bar(range(len(probabilities)), probabilities, color="blue")
    axs[1].set_xticks(range(len(probabilities)))
    axs[1].set_xlabel("Classes")
    axs[1].set_ylabel("Probability")
    axs[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.show()
