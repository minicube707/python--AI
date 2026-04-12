
from .Mathematical_function import ReLU, LeakyReLU, Sigmoide, Tanh, Linear
from .Layers import Dense, BatchNorm, Dropout, Block


class DNN():
    
    def __init__(self, x_shape, y_shape, structure, alpha, optimizer, support):

        self.structure = structure
        self.layers = []
        self.C_DNN = len(structure)
        self.logits = None
        self.support = support

        self.initialisation(x_shape, y_shape, alpha)
        
        self.optimizer = optimizer

    def initialisation(self, x_shape, y_shape, alpha):

        structure = self.structure
        C_DNN = self.C_DNN
        support = self.support

        structure[str(C_DNN)] = (y_shape, structure[str(C_DNN)][1], structure[str(C_DNN)][2])
        nb_activation = x_shape

        for i in range(1, C_DNN + 1):
            nb_neuron, activation_function, dropout_per = structure[str(i)]

            dense = Dense.add_layer(nb_activation, nb_neuron, support)
            batchnorm = BatchNorm.add_layer(nb_neuron, support)

            # instanciation activation
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

            self.layers.append(Block(dense, batchnorm, activation, dropout))
            nb_activation = nb_neuron
    
    def get_parameters_update(self):
        params = []
        for block in self.layers:
            params += block.dense.get_params_update()
            params += block.batchnorm.get_params_update()
        return params


    def set_parameters(self, parameters):

        for i, block in enumerate(self.layers):
            
            W = parameters[f"DNN_W{i}"]
            B = parameters[f"DNN_B{i}"]
            block.dense.set_params(W, B)

            g = parameters[f"DNN_g{i}"]
            b = parameters[f"DNN_b{i}"]
            rm = parameters[f"DNN_rm{i}"]
            rv = parameters[f"DNN_rv{i}"]
            block.batchnorm.set_params(g, b, rm, rv)
          
    def forward_propagation(self, X, training):

        for block in self.layers:
            X = block.forward(X, training)

        self.logits = X

    def backward_propagation(self, dZ):
        
        for block in reversed(self.layers):
            dZ = block.backward(dZ)
        return dZ
    
    def update(self):
        params = self.get_parameters_update()
        self.optimizer.update(params)

    def show_information(self):

        structure = self.structure
        C_DNN = self.C_DNN

        print("")
        print("============================")
        print("    INITIALISATION DNN")
        print("============================")

        print("\nDétail de la convolution :")
        print("Nb activation")
        for c in range(1, C_DNN + 1):
            print(structure[str(c)][0], end="")
            if c < C_DNN:
                print("->", end="")
        print("")


        print("")
        for c, block in enumerate(self.layers):
            print("W" + str(c + 1), ":", block.dense.W.shape)
            print("B" + str(c + 1), ":", block.dense.b.shape)
         

        print("")
        print("nb neuron, function, dropout")
        for keys, values in structure.items():
            print(keys, values)

        print("\nNumber of parameter:", f"{self.get_nb_parameter():,}".replace(",", " "))
        
    def save(self):

        save = {}
        for i, block in enumerate(self.layers):

            W, B = block.dense.get_params_save()
            save[f"DNN_W{i}"] = W
            save[f"DNN_B{i}"] = B

            g, b, rm, rv = block.batchnorm.get_params_save()
            save[f"DNN_g{i}"] = g
            save[f"DNN_b{i}"] = b
            save[f"DNN_rm{i}"] = rm
            save[f"DNN_rv{i}"] = rv

        return save
    
    def get_nb_parameter(self):
        
        nb_parameter = 0
        for block in self.layers:
        
            W, B = block.dense.get_params_save()
            nb_parameter += W.size
            nb_parameter += B.size

        return nb_parameter