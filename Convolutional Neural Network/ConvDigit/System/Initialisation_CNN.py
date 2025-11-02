
import numpy as np

from .Convolution_Neuron_Network import calcul_output_shape, error_initialisation
from .Deep_Neuron_Network import initialisation_DNN

"""
initialisation_extraction:
=========DESCRIPTION=========
Extrait all the information inside the dict

=========INPUT=========
dict    dimensions :    all the information on how is built the CNN
int     i :             the stage of the CNN

=========OUTPUT=========
int     k_size :        the size in row of kernel
int     stride :        how many pixel the kernel move  
int     padding :       how many pixel we add to the border of the activation
int     nb_kernel :     how many kernel
string  type_layer :    the type of layer 
string  fonction :      the type of function
"""
def initialisation_extraction(dimensions, i):
    #Kernel size, stride, padding, nb_kernel, type layer, function

    k_size = dimensions[str(i)][0]
    stride = dimensions[str(i)][1]
    padding = dimensions[str(i)][2]
    nb_kernel = dimensions[str(i)][3]
    type_layer = dimensions[str(i)][4]
    fonction = dimensions[str(i)][5]

    return k_size, stride, padding, nb_kernel, type_layer, fonction


"""
initialisation_pooling:
=========DESCRIPTION=========
Set the value for pooling operation

=========INPUT=========
dict    parametres :    dictionary to fill with the pooling operation
int     k_size :        the size in row of the kernel
string  type_layer :    the type of layer 
string  fonction :      the type of function
int     i :             the stage of the CNN

=========OUTPUT=========
dict    parametres :    containt all the information for the pooling operation
"""
def initialisation_pooling(parametres, k_size, type_layer, fonction, i):

    parametres["K" + str(i)] = k_size**2
    parametres["b" + str(i)] = None
    parametres["l" + str(i)] = type_layer
    parametres["f" + str(i)] = fonction
    
    return parametres


"""
initialisation_kernel:
=========DESCRIPTION=========
Set the value for kernel operation, the update operation

=========INPUT=========
dict    parametres :        dictionary to fill with the kernel information
dict    parametres_grad :   dictionary to fill with the update information
int     k_size :            the size in row of the kernel
int     o_size :            the size in row of the output
int     nb_kernel :         the number of kernel
string  type_layer :        the type of layer 
string  fonction :          the type of function
int     i :                 the stage of the CNN

=========OUTPUT=========
dict    parametres :        containt all the information for the kernel operation
dict    parametres_grad :   containt all the information for the update operation
"""
def initialisation_kernel(parametres, parametres_grad, k_size, type_layer, fonction, i, nb_kernel, nb_layer, o_size):
    shape = (nb_kernel, nb_layer, k_size**2, 1)

    if fonction == "relu":
        std = np.sqrt(2 / (nb_layer * k_size**2))
        K = np.random.randn(*shape).astype(np.float32) * std

    elif fonction == "tanh" or  fonction == "sigmoide":
        limit = np.sqrt(6 / (nb_layer + nb_kernel))
        K = (np.random.rand(*shape).astype(np.float32) * 2 - 1) * limit

    else:
        # Default to small random values
        K = np.random.randn(*shape).astype(np.float32) * 0.01

    b_shape = (nb_kernel, np.int64(o_size)**2, 1)
    b = np.zeros(b_shape).astype(np.float32)  # Bias souvent initialisé à 0

    parametres["K" + str(i)] = K
    parametres["b" + str(i)] = b
    parametres["l" + str(i)] = type_layer
    parametres["f" + str(i)] = fonction

    parametres_grad["m" + str(i)] = np.zeros(shape).astype(np.float32)
    parametres_grad["v" + str(i)] = np.zeros(shape).astype(np.float32)

    return parametres, parametres_grad


"""
initialisation_calcul:
=========DESCRIPTION=========
Preproce the information to built the CNN

=========INPUT=========
int     x_shape1 :      the shape of the input
dict    dimensions :    all the information on how is built the CNN
string  padding_mode :  string to know if the auto-padding is active

=========OUTPUT=========
dict    dimension :     all the information on how is built the CNN
list    list_size_activation :     list of all activation shape with number of activation and padding
"""
def initialisation_calcul(x_shape, dimensions, padding_mode):

    list_size_activaton = []
    list_size_activaton.append((x_shape[0], x_shape[1]))
    nb_activation  = x_shape[0]
    input_size =  x_shape[1]
    previ_input_size = input_size
    
    for i in range(1, len(dimensions)+1):

        k_size, stride, padding, nb_channel, type_layer, fonction = initialisation_extraction(dimensions, i)

        #If the input doesn't match perfectly with the kernel and padding and is in mode auto-correction, the system correct the mistake and add the right padding
        if input_size % stride != 0 and padding_mode == "auto":
            padding = stride - input_size % stride
            list_size_activaton[-1] = (list_size_activaton[-1][0], input_size + padding)

        if (dimensions[str(i)][4] == "kernel"):
            #Add the modificaton to the dict
            dimensions[str(i)] = k_size, stride, padding, nb_channel, type_layer, fonction
        
        else:
            nb_channel = nb_activation
            nb_channel = list_size_activaton[-1][0]
            dimensions[str(i)] = k_size, stride, padding, nb_channel, type_layer, fonction

        o_size = calcul_output_shape(input_size, k_size, stride, padding)
        previ_input_size = input_size + padding
        input_size = o_size
        nb_activation = dimensions[str(i)][3]
        
        list_size_activaton.append((nb_channel, input_size))
        error_initialisation(dimensions, (input_size, nb_activation), previ_input_size, type_layer, fonction, stride)

    return dimensions, list_size_activaton

"""
initialisation_affectation:
=========DESCRIPTION=========
Set all the value to built the CNN

=========INPUT=========
dict    dimensions :    all the information on how is built the CNN
int     x_shape :       the shape of the input

=========OUTPUT=========
dict    parametres :        containt all the information for the kernel operation
dict    parametres_grad :   containt all the information for the update operation
"""
def initialisation_affectation(dimensions, x_shape, list_size_activation):

    parametres = {}
    parametres_grad = {}

    nb_layer = x_shape[0]
    o_size = x_shape[1]

    for i in range(1, len(dimensions)+1):
        k_size, _, _, nb_kernel, type_layer, fonction = initialisation_extraction(dimensions, i)
        o_size = calcul_output_shape(o_size, dimensions[str(i)][0], dimensions[str(i)][1], dimensions[str(i)][2])

        if type_layer == "kernel":
            parametres, parametres_grad = initialisation_kernel(parametres, parametres_grad, k_size, type_layer, fonction, i, nb_kernel, nb_layer, o_size)

        elif type_layer == "pooling":
            parametres = initialisation_pooling(parametres, k_size, type_layer, fonction, i)

        nb_layer = nb_kernel

    return parametres, parametres_grad


"""
initialisation_CNN:
=========DESCRIPTION=========
Set all the value to built the CNN

=========INPUT=========
int     x_shape :       the shape of the input
dict    dimensions :    all the information on how is built the CNN
string  padding_mode :  string to know if the auto-padding is active

=========OUTPUT=========
dict    parametres :        containt all the information for the kernel operation
dict    parametres_grad :   containt all the information for the update operation
dict    dimension :         all the information on how is built the CNN
"""
def initialisation_CNN(x_shape, dimensions, padding_mode):

    dimensions, list_size_activation = initialisation_calcul(x_shape, dimensions, padding_mode)
    parametres, parametres_grad = initialisation_affectation(dimensions, x_shape, list_size_activation)

    return parametres, parametres_grad, dimensions


"""
initialisation_AI:
=========DESCRIPTION=========
Set all the value to built the AI

=========INPUT=========
tuple   input_shape :   tuple contain the dimension of the input
dict    dimensions_CNN :    all the information on how is built the CNN
string  padding_mode :  string to know if the auto-padding is active
tuple   hidden_layer :  tuple contain the information of the hidden layer
tuple   output_shape :  tuple contain the dimension of the output

=========OUTPUT=========
dict    parametres_CNN :    containt all the parametre of the CNN
dict    parametres_grad :   containt all the information for the update operation
dict    parametres_DNN :    containt all the parametre of the DNN
tuple   list_size_activation:   tuple of all activation shape with number of activation and padding
"""
def initialisation_AI(input_shape, dimensions_CNN, padding_mode, dimensions_DNN, output_shape): 
        
    parametres_CNN, parametres_grad, dimensions_CNN = initialisation_CNN (
    input_shape, dimensions_CNN, padding_mode
    )

    input_size = input_shape[1]
    for val in dimensions_CNN.values():
        input_size = calcul_output_shape(input_size, val[0], val[1], val[2])

    last_CNN_layer = dimensions_CNN[str(len(dimensions_CNN))]
    flattened_size = np.int32((np.int32(input_size)**2 * last_CNN_layer[3]))

    parametres_DNN = initialisation_DNN (dimensions_DNN, flattened_size, output_shape[1])

    return parametres_CNN, parametres_grad, parametres_DNN, dimensions_CNN