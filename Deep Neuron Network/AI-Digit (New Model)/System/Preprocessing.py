
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import IsolationForest

from .Sklearn_tools import train_test_split, Label_binarizer

def show_information_setting(nb_iteration, max_attempts, min_confidence_score, alpha, 
                             learning_rate_DNN, validation_size, ratio_test):

    print("\n============================")
    print("         SETTING")
    print("============================")

    print("\nInfo Training")
    print("Nombre d'iteration: ", nb_iteration);
    print("Max attempts: ", max_attempts)
    print("Min confidence score: ", min_confidence_score)
    print("Validation_size: ", validation_size)
    print("Ratio trainset/testset: ", ratio_test)

    print("\nInfo DNN")
    print("Learning rate: ", learning_rate_DNN)
    print("Alpha: ", alpha)


def handle_key(event):
    if event.key == ' ':
        plt.close(event.canvas.figure)  # Ferme la fenêtre associée


def set_mode():

    while(1):
        print("\n0: Exit")
        print("1: CNN Data")
        print("2: DNN Data")
        str_answer = input("What type of data it is ?\n")
        try:
            int_answer = int(str_answer)
        except:
            print("Please answer with only 1 or 2")
            continue
        if (int_answer == 0):
            print("Exit")
            exit(0)

        if (int_answer == 1):
            print("You use CNN Data")
            return(1)
        
        elif (int_answer == 2):
            print("You use DNN Data")
            return(2)
        
        else:
            print("Please answer with only 1 or 2")


def preprocessing(X, y, input_shape, test_size=0.1):
    
    print("Data shape")
    print("X:",X.shape)
    print("Y:",y.shape)
    
    data_mode = set_mode()

    # --- MODE 1 : Affichage d’images ---
    if (data_mode == 1):
        classes = np.unique(y)
        for cls in classes:
            fig = plt.figure(figsize=(16, 8))
            fig.suptitle(f"Classe {cls}", fontsize=16)
            fig.canvas.mpl_connect('key_press_event', handle_key)  # Active la détection de la touche
            
            # Récupère les indices des images correspondant à la classe cls
            indices = np.where(y == cls)[0][:15]  # 15 premières images
            for i, idx in enumerate(indices):
                plt.subplot(3, 5, i + 1)  # 3 lignes, 5 colonnes
                plt.imshow(X[idx].reshape(input_shape[1], input_shape[2]), cmap="gray")
                plt.title(f"{y[idx]}")
                plt.axis("off")

            plt.tight_layout()
            plt.show() 

    # --- MODE 2 : Visualisation graphique ---
    if (data_mode == 2):

        df = pd.DataFrame(X, columns=[
            "sepal length (cm)", "sepal width (cm)",
            "petal length (cm)", "petal width (cm)"
        ])
        df["species"] = y  # ajoute les étiquettes d’espèces

        # Style esthétique
        sns.set(style="whitegrid", context="notebook", palette="muted")

        # --- 1. Pairplot ---
        sns.pairplot(df, hue="species", diag_kind="kde", height=2.2)
        fig = plt.figure(figsize=(8, 6))
        fig.canvas.mpl_connect('key_press_event', handle_key)  # Active la détection de la touche
        fig.suptitle("Visualisation du dataset Iris 🌸", y=1.02, fontsize=16)
        plt.show()

    #______________________________________________________________#
    #Remove the bad data
    model=IsolationForest(contamination=0.2)
    model.fit(X)
    outlier = model.predict(X) == 1
    X = X[outlier]
    y = y[outlier]


    #______________________________________________________________#
    #Split the dataset for the training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    y_train = y_train.reshape((y_train.shape[0], 1))
    y_test = y_test.reshape((y_test.shape[0], 1))


    #______________________________________________________________#
    #Encode the labels for the trainning
    transformer=LabelBinarizer()
    transformer.fit(y_train)
    y_train = transformer.transform(y_train.reshape((-1, 1)))
    y_test = transformer.transform(y_test.reshape((-1, 1)))

    print("\nTrain")
    print("La dimension de X_train",X_train.shape)
    print("La dimension de y_train",y_train.shape)
    print(np.unique(y_train, return_counts=True))

    print("\nTest")
    print("La dimension de X_test",X_test.shape)
    print("La dimension de y_test",y_test.shape)
    print(np.unique(y_test, return_counts=True))
  
    New_X_train = X_train / X_train.max()
    New_X_test = X_test / X_train.max()

    

    #Pour les X se sont les variables en premier (ici les pixels) puis le nombres d'échantillons 
    #Pour les y se sont les labels d'abord puis le nombre d'échantillons
    print("\nNew_X_train.shape:", New_X_train.shape)
    print("New_X_test.shape:", New_X_test.shape)
    print("y_test.shape:", y_test.shape)
    print("y_train.shape:", y_train.shape)

    #Affichage des 15 premières images du dataset
    n = min(16, len(y_train))
    fig = plt.figure(figsize=(n,8))
    fig.canvas.mpl_connect('key_press_event', handle_key)  # Active la détection de touches 
    fig.suptitle("Train Dataset")
    for i in range(1,n):
        plt.subplot(4,5, i)
        plt.imshow(New_X_train.reshape((New_X_train.shape[0], input_shape[1], input_shape[2]))[i], cmap="gray")
        plt.title(str(np.argmax(y_train[i])))
        plt.axis("off")
    plt.tight_layout()    
    plt.show() 

    #Affichage des 15 premières images
    n = min(16, len(y_test))
    fig = plt.figure(figsize=(n,8))
    fig.canvas.mpl_connect('key_press_event', handle_key)  # Active la détection de touches 
    fig.suptitle("Test Dataset")
    for i in range(1,n):
        plt.subplot(4,5, i)
        plt.imshow(New_X_test.reshape((New_X_test.shape[0], input_shape[1], input_shape[2]))[i], cmap="gray")
        plt.title(str(np.argmax(y_test[i])))
        plt.axis("off")
    plt.tight_layout()    
    plt.show() 

    return New_X_train, y_train, New_X_test, y_test, transformer