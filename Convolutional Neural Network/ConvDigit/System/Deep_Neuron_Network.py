
from .Mathematical_function import Linear, ReLU, LeakyReLU, Sigmoide, Tanh
from .Mathematical_function_GPU import ReLU_GPU, LeakyReLU_GPU, Sigmoide_GPU, Tanh_GPU

from .Layer import Dense, BatchNorm, Dropout, Block
from .Layer_GPU import Dense_GPU, BatchNorm_GPU, Dropout_GPU


class DNN():
    
    def __init__(self, x_shape, y_shape, structure, alpha, optimizer, gpu_mode):

        self.structure = structure
        self.layers = []
        self.C_DNN = len(structure)
        self.logits = None
        self.gpu_mode = gpu_mode

        self.initialisation(x_shape, y_shape, alpha)
        
        self.optimizer = optimizer

    def initialisation(self, x_shape, y_shape, alpha):

        structure = self.structure
        C_DNN = self.C_DNN
        gpu_mode = self.gpu_mode

        structure[str(C_DNN)] = (y_shape, structure[str(C_DNN)][1], structure[str(C_DNN)][2])
        nb_activation = x_shape

        LAYER_MAP = {
            "batchNorm": (BatchNorm, BatchNorm_GPU),
            "sigmoide": (Sigmoide, Sigmoide_GPU),      
            "tanh": (Tanh, Tanh_GPU),
            "relu": (ReLU, ReLU_GPU),
            "leaky relu": (LeakyReLU, LeakyReLU_GPU),
            "dropout": (Dropout, Dropout),
            "linear": (Linear, Linear),
            "dense": (Dense, Dense_GPU)
        }
        
        def get_layer(layer_name, gpu_mode=True, *args, **kwargs):
            CPU_class, GPU_class = LAYER_MAP[layer_name]
            LayerClass = GPU_class if gpu_mode else CPU_class
            return LayerClass(*args, **kwargs)

        for i in range(1, C_DNN + 1):
            nb_neuron, activation_function, dropout_per = structure[str(i)]

            dense = get_layer("dense", gpu_mode, nb_activation, nb_neuron)
            batchnorm = get_layer("batchNorm", gpu_mode, nb_neuron)

            # instanciation activation
            activation = get_layer(activation_function, gpu_mode, alpha if activation_function=="leaky relu" else None)
            dropout = get_layer("dropout", gpu_mode, dropout_per)

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

    def set_alpha(self, alpha):

        for block in (self.layers):
            if isinstance(block.activation, LeakyReLU):
                block.activation.alpha = alpha
                
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
        print("")

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