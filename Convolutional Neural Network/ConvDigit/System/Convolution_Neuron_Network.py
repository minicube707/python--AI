
import numpy as np

from .Mathematical_function import Linear, ReLU, LeakyReLU, Sigmoide, Tanh
from .Layers import MaxPooling, Convolution, BatchNorm, Dropout, Block

def calcul_output_shape(input_size, k_size, stride, padding):
    return int((input_size - k_size + padding) / stride + 1)

class CNN():

    def __init__(self, structure, input_shape, padding_mode, alpha, optimizer, support):

        self.structure = structure
        self.layers = []
        self.C_CNN = len(structure)
        self.logits = None
        self.support = support

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

        prev_channels = x_shape[0]
        input_size = x_shape[1]

        for i in range(1, len(structure) + 1):

            # --- Get parameters ---
            k_size, stride, padding, nb_kernel, layer_type, activation, dropout = self.initialisation_extraction(structure, i)
                        
            # --- Add padding ---
            if padding_mode in {"auto", "same"} and input_size % stride != 0:
                padding = stride - (input_size % stride)
            
            if padding_mode == "same" and layer_type == "conv" and stride == 1:
                padding = input_size - calcul_output_shape(input_size, k_size, stride, 0) 
                
            # --- Calcul output size ---
            output_size = calcul_output_shape(input_size, k_size, stride, padding)
                
            # --- Manage channels ---
            if layer_type == "conv":
                current_channels = nb_kernel

            elif layer_type == "pool":
                nb_kernel = prev_channels

            else:
                raise ValueError(f"Unknown layer type: {layer_type}")

            # --- Update structure ---
            structure[str(i)] = (
                k_size, stride, padding, nb_kernel,
                layer_type, activation, dropout
            )

            # --- Validation ---
            self.error_initialisation(
                x_shape,
                output_size,
                input_size + padding,
                layer_type,
                activation,
                stride,
                dropout
            )

            # --- Update next iteration ---
            input_size = output_size
            prev_channels = current_channels

        self.structure = structure

    def initialisation_affectation(self, x_shape, alpha):

        nb_layer = x_shape[0]
        o_size = x_shape[1]
        C = self.C_CNN
        structure = self.structure
        support = self.support

        for i in range(1, C + 1):
            k_size, stride, padding, nb_kernel, type_layer, activation_function, dropout_per = self.initialisation_extraction(structure, i)
            o_size = calcul_output_shape(o_size, structure[str(i)][0], structure[str(i)][1], structure[str(i)][2])

            # Construction de la layer
            if type_layer == "conv":

                corr = Convolution.add_layer(nb_kernel, nb_layer, k_size, stride, o_size, padding, support)
                batchnorm = BatchNorm.add_layer(nb_kernel, support)
                
                if (activation_function == "sigmoide"):
                    activation = Sigmoide.add_layer(support)
                elif (activation_function == "tanh"):
                    activation = Tanh.add_layer(support)
                elif (activation_function == "relu"):
                    activation = ReLU.add_layer(support)
                elif (activation_function == "leaky relu"):
                    activation = LeakyReLU.add_layer(alpha, support)
                elif (activation_function == "linear"):
                    activation = Linear()
                else:
                    raise Exception("Undefine activatoin function")
                
                dropout = Dropout.add_layer(dropout_per, support)

            elif type_layer == "pool":
                corr = MaxPooling.add_layer(k_size, stride, padding, support)
                batchnorm = Linear()
                activation = Linear()
                dropout = Linear()

            self.layers.append(Block(corr, batchnorm, activation, dropout))
            nb_layer = nb_kernel
    

    def error_initialisation(self, x_shape, output_size, input_size, type_layer, fonction, stride, dropout):

        if output_size < 1:
            self.show_information(x_shape)
            raise ValueError(f"ERROR: The current dimension is {output_size}. Dimension can't be negatif")
            
        if input_size % output_size != 0 and stride != 1:
            self.show_information(x_shape)
            raise ValueError(f"ERROR: Issue with the dimension for the pooling. {input_size} not divide {output_size}")
        
        if type_layer not in ["conv", "pool"]:
            self.show_information(x_shape)
            raise NameError(f"ERROR: Layer parametre '{type_layer}' is not defined. Please correct with 'pool' or 'conv'.")
        
        if fonction not in ["relu", "sigmoide", "max", "tanh", "leaky relu", "linear"]:
            self.show_information(x_shape)
            raise NameError(f"ERROR: Layer parametre '{fonction}' is not defined. Please correct with 'linear', 'relu', 'leaky relu', 'sigmoide', 'max' ou 'tanh'.")

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


    def get_parameters_update(self) :
        params = []
        for block in self.layers:

            if block.dense.class_ == "Convolution":
                params += block.dense.get_params_update()
                params += block.batchnorm.get_params_update()

        return params
    
    def set_parameters(self, parameters):

        for i, block in enumerate(self.layers):
            
            if block.dense.class_ == "Convolution":
                K = parameters[f"CNN_K{i}"]
                B = parameters[f"CNN_B{i}"]
                block.dense.set_params(K, B)

                g = parameters[f"CNN_g{i}"]
                b = parameters[f"CNN_b{i}"]
                rm = parameters[f"CNN_rm{i}"]
                rv = parameters[f"CNN_rv{i}"]
                block.batchnorm.set_params(g, b, rm, rv)


    def get_activations(self, X, i):
        
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
        
        if block.dense.class_ == "MaxPooling":
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

        print("\nNumber of parameter:", f"{self.get_nb_parameter():,}".replace(",", " "))
        

    def save(self):

        save = {}
        for i, block in enumerate(self.layers):

            if block.dense.class_ == "Convolution":

                K, B = block.dense.get_params_save()
                save[f"CNN_K{i}"] = K
                save[f"CNN_B{i}"] = B

                g, b, rm, rv = block.batchnorm.get_params_save()
                save[f"CNN_g{i}"] = g
                save[f"CNN_b{i}"] = b
                save[f"CNN_rm{i}"] = rm
                save[f"CNN_rv{i}"] = rv

        return save
    
    def get_nb_parameter(self):
        
        nb_parameter = 0
        for block in self.layers:
            
            if block.dense.class_ == "Convolution":
                K, B = block.dense.get_params_save()
                nb_parameter += K.size
                nb_parameter += B.size

        return nb_parameter