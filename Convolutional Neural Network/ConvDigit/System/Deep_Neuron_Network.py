
from .Mathematical_function import ReLU, LeakyReLU, Sigmoide, Tanh
from .Layer import BatchNorm, Dropout, Block, Dense

class DNN():
    
    def __init__(self, y, x_shape, dimensions, alpha, optimizer):

        self.dimensions = dimensions
        self.layers = []
        self.C_DNN = len(dimensions)
        self.logits = None

        DNN.initialisation(self, y, x_shape, alpha)
        
        self.optimizer = optimizer

    def initialisation(self, y, x_shape, alpha):

        dimensions = self.dimensions
        C_DNN = self.C_DNN
        
        dimensions[str(C_DNN)] = (y.shape[1], dimensions[str(C_DNN)][1], dimensions[str(C_DNN)][2])
        nb_activation = x_shape

        for i in range(1, C_DNN + 1):

            nb_neuron, activation_function, dropout_per = dimensions[str(i)]

            #Dense
            dense = Dense(nb_activation, nb_neuron)

            #Batchnorm
            batchnorm =  BatchNorm(nb_neuron)

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

            self.layers.append(Block(dense, batchnorm, activation, dropout))
            nb_activation = nb_neuron
    
    def get_parameters(self):
        params = []
        for block in self.layers:
            params += block.dense.get_params()
            params += block.batchnorm.get_params()
        return params

    def forward_propagation(self, X, training):

        for block in self.layers:
            X = block.forward(X, training)

        self.logits = X

    def backward_propagation(self, dZ):
        
        for block in reversed(self.layers):
            dZ = block.backward(dZ)
        return dZ
    
    def update(self):
        params = self.get_parameters()
        self.optimizer.update(params)

    def show_information(self):

        dimensions = self.dimensions
        C_DNN = self.C_DNN

        print("")
        print("============================")
        print("    INITIALISATION DNN")
        print("============================")

        print("\nDétail de la convolution :")
        print("Nb activation")
        for c in range(1, C_DNN + 1):
            print(dimensions[str(c)][0], end="")
            if c < C_DNN:
                print("->", end="")
        print("")


        print("")
        for c, block in enumerate(self.layers):
            print("W" + str(c + 1), ":", block.dense.W.shape)
            print("B" + str(c + 1), ":", block.dense.b.shape)
         

        print("")
        print("nb neuron, function, dropout")
        for keys, values in dimensions.items():
            print(keys, values)
        print("")