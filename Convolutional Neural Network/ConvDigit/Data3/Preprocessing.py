
import numpy as np

from Sklearn_tools import train_test_split, Label_binarizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import IsolationForest

import matplotlib.pyplot as plt

from Convolution_Neuron_Network import add_padding, reshape

def preprocessing(X, y, input_shape, test_size=0.1):
    X = X.reshape(-1, 8, 8)

    #Affichage des 15 premières images
    plt.figure(figsize=(16,8))
    plt.suptitle("Dataset")
    for i in range(1,16):
        plt.subplot(4,5, i)
        plt.imshow(X.reshape((X.shape[0], input_shape[1], input_shape[2]))[i], cmap="gray")
        plt.title(y[i])
        plt.axis("off")
    plt.tight_layout()    
    plt.show()  


    #______________________________________________________________#
    #Remove the bad data
    X_reshape = X.reshape(X.shape[0], -1)
    model=IsolationForest(contamination=0.2)
    model.fit(X_reshape)
    outlier = model.predict(X_reshape) == 1
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
    plt.figure(figsize=(16,8))
    plt.suptitle("Train Dataset")
    for i in range(1,16):
        plt.subplot(4,5, i)
        plt.imshow(New_X_train.reshape((New_X_train.shape[0], input_shape[1], input_shape[2]))[i], cmap="gray")
        plt.title(y_train[i])
        plt.axis("off")
    plt.tight_layout()    
    plt.show() 

    #Affichage des 15 premières images
    plt.figure(figsize=(16,8))
    plt.suptitle("Test Dataset")
    for i in range(1,16):
        plt.subplot(4,5, i)
        plt.imshow(New_X_test.reshape((New_X_test.shape[0], input_shape[1], input_shape[2]))[i], cmap="gray")
        plt.title(y_test[i])
        plt.axis("off")
    plt.tight_layout()    
    plt.show() 

    return New_X_train, y_train, New_X_test, y_test, transformer