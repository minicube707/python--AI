
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

from Preprocessing import preprocessing
from Mathematical_function import softmax
from File_Management import file_management, select_model
from Deep_Learning import convolution_neuron_network
from Deep_Learning import foward_propagation
from Initialisation_CNN import initialisation_CNN
from Convolution_Neuron_Network import show_information


module_dir = os.path.dirname(__file__)
os.chdir(module_dir)

load = False

#Data_Digit
with np.load("data/load_digits.npz") as f:
        X, y = f["data"], f["target"]

#Initialisation CNN
learning_rate_CNN = 0.005
beta1 = 0.9
beta2 = 0.99

#Initialisation DNN
hidden_layer = (16,16)
learning_rate_DNN = 0.05

dimensions_CNN = {}
#Kernel size, stride, padding, nb_kernel, type layer, function
dimensions_CNN = {  "1" :(2, 1, 0, 2, "kernel", "relu"),
                    "2" :(2, 2, 0, 1, "pooling", "max"),
                    "3" :(2, 1, 0, 10, "kernel", "sigmoide")}

nb_iteration = 10

#Number of channel by picture
input_shape = (1, 8, 8)

padding_mode = "auto"
_, _, dimensions_CNN, _ = initialisation_CNN(input_shape, dimensions_CNN, padding_mode)
  
X_train, y_train, X_test, y_test, transformer = preprocessing(X, y, dimensions_CNN)

if load:    
    #Load the model
    model = select_model()
    with open("Model/" + str(model), 'rb') as file:
        parametres = pickle.load(file)

else:   
     parametres_CNN, parametres_DNN, dimensions_CNN, dimensions_DNN, test_accu = convolution_neuron_network(X_train, y_train, X_test, y_test, nb_iteration, hidden_layer, dimensions_CNN \
        , learning_rate_CNN, learning_rate_DNN, beta1, beta2, input_shape)

if not load:

    #Save the best model
    name_model = file_management(test_accu)
    print(name_model)
    with open("Model/" + name_model, 'wb') as file:
        pickle.dump(parametres, file)


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
    plt.imshow(X_final.reshape((X_final.shape[0], 8, 8))[i], cmap="gray")
    plt.title(f"Value:{y_final[i]} Predict:{pred}  ({np.round(porcent, 2)}%)")
    plt.tight_layout()
    plt.axis("off")
plt.show()  

nb_test = 10
print("")
for i in range(nb_test):
    index = int(input(f"Please enter a number between 1 and {X_final.shape[0]}: "))
    if index == index < 0:
        break
    
    # Prédiction des probabilités avec softmax
    probabilities = softmax(activation["A" + str(C)].T)[index,:]
    pred = np.argmax(probabilities)
    porcent = np.max(probabilities)

    # Création de la figure avec 2 sous-graphiques (image + histogramme)
    fig, axs = plt.subplots(2, 1, figsize=(5, 7), gridspec_kw={'height_ratios': [3, 1]})

    # Affichage de l'image
    axs[0].imshow(X_final.reshape((X_final.shape[0], 8, 8))[index], cmap="gray")
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
