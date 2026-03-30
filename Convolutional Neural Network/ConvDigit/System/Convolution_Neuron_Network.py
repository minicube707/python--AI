
import numpy as np
from .Mathematical_function import Linear, ReLU, LeakyReLU, Sigmoide, Tanh
from .Layer import MaxPooling, Convolution, BatchNorm, Dropout, Block

def calcul_output_shape(input_size, k_size, stride, padding):
    return np.int8((input_size - k_size + padding) / stride +1)

class CNN():

    def __init__(self, structure, input_shape, padding_mode, alpha, optimizer):

        self.structure = structure
        self.layers = []
        self.C_CNN = len(structure)
        self.logits = None
       
        self.initialisation (input_shape, padding_mode, alpha)

        self.optimizer = optimizer

    def initialisation(self, x_shape, padding_mode, alpha):
        
        self.initialisation_calcul(x_shape, padding_mode)
        self.initialisation_affectation(x_shape, alpha)
    

    def initialisation_extraction(self, structure, i):

        #Kernel size, stride, padding, nb_kernel, type layer, function, dropout

        k_size = structure[str(i)][0]
        stride = structure[str(i)][1]
        padding = structure[str(i)][2]
        nb_kernel = structure[str(i)][3]
        type_layer = structure[str(i)][4]
        fonction = structure[str(i)][5]
        dropout = structure[str(i)][6]

        return k_size, stride, padding, nb_kernel, type_layer, fonction, dropout
    
    def initialisation_calcul(self, x_shape, padding_mode):
        
        structure = self.structure
        nb_channel = x_shape[0]
        input_size = x_shape[1]

        previ_input_size = input_size
        previ_channel = nb_channel

        
        for i in range(1, len(structure)+1):

            k_size, stride, padding, nb_kernel, type_layer, fonction, dropout = self.initialisation_extraction(structure, i)

            #Add padding
            if input_size % stride != 0 and padding_mode == "auto":
                padding = int(stride - input_size % stride)
                structure[str(i)] = (k_size, stride, padding, nb_kernel, type_layer, fonction, dropout)
                
            if type_layer == "conv":
                nb_channel = nb_kernel
                previ_channel = nb_channel

            #Conserve the nb of channel
            elif type_layer == "pool":
                structure[str(i)] = (k_size, stride, padding, previ_channel, type_layer, fonction, dropout)

            o_size = calcul_output_shape(input_size, k_size, stride, padding)
            input_size = o_size
            previ_input_size = input_size

            self.structure = structure

            self.error_initialisation(x_shape, input_size, previ_input_size, type_layer, fonction, stride, dropout)


    def initialisation_affectation(self, x_shape, alpha):

        nb_layer = x_shape[0]
        o_size = x_shape[1]
        C = self.C_CNN
        structure = self.structure

        for i in range(1, C + 1):
            k_size, stride, padding, nb_kernel, type_layer, activation_function, dropout_per = self.initialisation_extraction(structure, i)
            o_size = calcul_output_shape(o_size, structure[str(i)][0], structure[str(i)][1], structure[str(i)][2])

            if type_layer == "conv":
                
                if (i < C):
                    o_size = o_size + padding

                corr =  Convolution(nb_kernel, nb_layer, k_size, stride, o_size, padding)

                #Batchnorm
                batchnorm = BatchNorm(nb_kernel)

                #Activation
                if activation_function == "sigmoide":
                    activation = Sigmoide()
                
                elif activation_function == "tanh":
                    activation = Tanh()
                
                elif activation_function == "relu":
                    activation = ReLU()

                elif activation_function == "leaky relu":
                    activation = LeakyReLU(alpha)

                #Droout
                dropout = Dropout(dropout_per)

            elif type_layer == "pool":
                corr = MaxPooling(k_size, stride, padding)

                batchnorm = Linear()
                activation = Linear()
                dropout = Linear()

            self.layers.append(Block(corr, batchnorm, activation, dropout))
            nb_layer = nb_kernel
    

    def error_initialisation(self, x_shape, input_size, previ_input_size, type_layer, fonction, stride, dropout):

        if input_size < 1:
            self.show_information(x_shape)
            raise ValueError(f"ERROR: The current dimension is {input_size}. Dimension can't be negatif")
            
        if previ_input_size % input_size != 0 and stride != 1:
            self.show_information(x_shape)
            raise ValueError(f"ERROR: Issue with the dimension for the pooling. {previ_input_size} not divide {input_size}")
        
        if type_layer not in ["conv", "pool"]:
            self.show_information(x_shape)
            raise NameError(f"ERROR: Layer parametre '{type_layer}' is not defined. Please correct with 'pool' or 'conv'.")
        
        if fonction not in ["relu", "sigmoide", "max", "tanh", "leaky relu"]:
            self.show_information(x_shape)
            raise NameError(f"ERROR: Layer parametre '{fonction}' is not defined. Please correct with 'relu', 'leaky relu', 'sigmoide', 'max' ou 'tanh'.")

        if ( not (0 <= dropout <= 1)):
            self.show_information(x_shape)
            raise NameError(f"ERROR: dropout percent should be betwenn 0 and 1.")
    

    def forward_propagation(self, X, training):

        for block in self.layers:
            X = block.forward(X, training)

        self.logits = X

    def backward_propagation(self, dZ):
        
        for block in reversed(self.layers):
            dZ = block.backward(dZ)

    def update(self):
        params = self.get_parameters_update()
        self.optimizer.update(params)


    def get_parameters_update(self):
        params = []
        for block in self.layers:

            if isinstance(block.dense, Convolution):
                params += block.dense.get_params_update()
                params += block.batchnorm.get_params_update()

        return params
    
    def set_parameters(self, parameters):

        for i, block in enumerate(self.layers):
            
            if isinstance(block.dense, Convolution):
                K = parameters[f"CNN_K{i}"]
                B = parameters[f"CNN_B{i}"]
                block.dense.set_params(K, B)

                g = parameters[f"CNN_g{i}"]
                b = parameters[f"CNN_b{i}"]
                rm = parameters[f"CNN_rm{i}"]
                rv = parameters[f"CNN_rv{i}"]
                block.batchnorm.set_params(g, b, rm, rv)

    def set_alpha(self, alpha):

        for block in (self.layers):
            if isinstance(block.activation, LeakyReLU):
                block.activation.alpha = alpha


    def get_activatoins(self, X, i):
        
        c = 0
        for block in self.layers:

            Z1 = block.dense.forward(X)
            Z1_cpy = Z1.copy()
            Z2 = block.batchnorm.forward(Z1, False)
            Z3 = block.activation.forward(Z2)
            Z3_cpy = Z3.copy()
            Z4 = block.dropout.forward(Z3, False)

            if c >= i:
                break
        
            c += 1
            X = Z4
        
        if isinstance(block.dense, MaxPooling):
            return Z1_cpy, None
        
        return Z1_cpy, Z3_cpy
    
    def show_information(self, input_size):
        
        structure = self.structure

        print("\n============================")
        print("    INITIALISATION CNN")
        print("============================")

        print("\nDétail de la convolution :")
        print("Nb activation")
        print(f"{input_size[0]}", end="")
        print("->", end="")
        for i in range(1, len(structure)+1):

            if i < len(structure):
                print(f"{structure[str(i)][3]}", end="")
                print("->", end="")

        print(f"{structure[str(i)][3]}")  

        print("\nPadding")
        outpu_shape = input_size[2]
        for i in range(len(structure)):
            
            if i < len(structure):
                print(f"{outpu_shape}", end="")
                print(f"({structure[str(i+1)][2]})", end="")
                print("->", end="")

            outpu_shape = calcul_output_shape(outpu_shape, structure[str(i+1)][0], structure[str(i+1)][1], structure[str(i+1)][2])

        print(f"{outpu_shape}")  

        print("\nkernel size, stride, padding, nb_kernel, type layer, function, dropout")
        for keys, values in structure.items():
            print(keys, values)

    def save(self):

        save = {}
        for i, block in enumerate(self.layers):

            if isinstance(block.dense, Convolution):

                K, B = block.dense.get_params_save()
                save[f"CNN_K{i}"] = K
                save[f"CNN_B{i}"] = B

                g, b, rm, rv = block.batchnorm.get_params_save()
                save[f"CNN_g{i}"] = g
                save[f"CNN_b{i}"] = b
                save[f"CNN_rm{i}"] = rm
                save[f"CNN_rv{i}"] = rv

        return save