
import numpy as np
from .Mathematical_function import Layer, add_padding

class MaxPooling(Layer):
    
    def __init__(self, k_size, stride, padding):
        self.k_size = k_size
        self.stride = stride
        self.padding = padding
        self.X = None

    def forward(self, X):
        
        padding = self.padding

        # Padding
        if padding > 0:
            X = add_padding(X, padding)

        self.X = X

        k = self.k_size
        s = self.stride
        
        windows = np.lib.stride_tricks.sliding_window_view(
            X, (k, k), axis=(2, 3)
        )
        windows = windows[:, :, ::s, ::s, :, :]

        self.windows = windows 
        return windows.max(axis=(-1, -2))

    def backward(self, dA):
        k = self.k_size
        s = self.stride
        X = self.X
        windows = self.windows
        padding = self.padding

        # mask des max
        max_vals = windows.max(axis=(-1, -2), keepdims=True)
        mask = (windows == max_vals)

        # On broadcast dZ sur les k,k
        dZ_expanded = dA[:, :, :, :, None, None]
        dA_prev = mask * dZ_expanded

        dA_prev_full = np.zeros_like(X)

        H_out, W_out = dA.shape[2], dA.shape[3]

        for h in range(H_out):
            for w in range(W_out):
                h_start = h * s
                h_end   = h_start + k
                w_start = w * s
                w_end   = w_start + k
                dA_prev_full[:, :, h_start:h_end, w_start:w_end] += dA_prev[:, :, h, w, :, :]

        # Removal of padding
        if padding > 0:
            dA_prev_full = dA_prev_full[:, :, :-padding, :-padding]

        return dA_prev_full


class Convolution(Layer):

    def __init__(self, nb_kernel, nb_layer, k_size, stride, o_size, padding):
        
        k_shape = (nb_kernel, nb_layer, k_size, k_size)
        b_shape = (nb_kernel, o_size, o_size)
        
        self.K = np.random.randn(*k_shape) * 0.01
        self.b = np.zeros(b_shape)
        self.X = None

        self.dK = np.zeros(k_shape)
        self.db = np.zeros(b_shape)
        self.stride = stride
        self.padding = padding
        self.windows = None

    def forward(self, X):
        stride  = self.stride
        padding = self.padding

        B, C, H, W     = X.shape
        N, _, Kh, Kw   = self.K.shape

        # Padding
        if padding > 0:
            X = add_padding(X, padding)

        self.X = X  # stocke l'entrée paddée

        # Extraction des fenêtres (cross-correlation)
        windows = np.lib.stride_tricks.sliding_window_view(
            X, (Kh, Kw), axis=(2, 3)
        )
        windows = windows[:, :, ::stride, ::stride, :, :]
        self.windows = windows

        # Dimensions de sortie
        H_out, W_out = windows.shape[2], windows.shape[3]

        # Convolution (produit tensoriel)
        out = np.tensordot(
            self.K,
            windows,
            axes=([1, 2, 3], [1, 4, 5])
        )  # (N, B, H_out, W_out)

        out = np.moveaxis(out, 0, 1)  # → (B, N, H_out, W_out)

        # Ajout du biais
        out += self.b
        
        return out

    def backward(self, dZ):
        stride  = self.stride
        padding = self.padding

        B, N, H_out, W_out = dZ.shape
        _, C, Kh, Kw       = self.K.shape

        X  = self.X
        windows = self.windows

        # ========================
        # Parameter Gradients
        # ========================

        # dK
        self.dK = np.tensordot(
            dZ,
            windows,
            axes=([0, 2, 3], [0, 2, 3])
        )

        # db
        self.db = np.sum(dZ, axis=0)

        # ========================
        # Gradient entry
        # ========================

        # On calcule le gradient projeté à travers les poids
        # On utilise einsum pour multiplier dZ (B, N, H_out, W_out) 
        # par K (N, C, Kh, Kw) -> donne (B, C, H_out, W_out, Kh, Kw)
        dZ_windows = np.einsum('bnhw,nckl->bchwkl', dZ, self.K)
        
        # Reconstruction de dX
        H_in, W_in = self.X.shape[2], self.X.shape[3]
        dX = np.zeros_like(self.X)
        
        # Optimisation cruciale : Utiliser les indices pour vectoriser la sommation
        for h in range(Kh):
            for w in range(Kw):
                h_end = h + H_out * stride
                w_end = w + W_out * stride
                dX[:, :, h:h_end:stride, w:w_end:stride] += dZ_windows[:, :, :, :, h, w]

    
        # Removal of paddin
        if padding > 0:
            dX = dX[:, :, padding:-padding, padding:-padding]

        return dX

    
    def get_params_update(self):
        return [(self.K, self.dK), (self.b, self.db)]
    

    def get_params_save(self):
        return self.K, self.b
    
    def set_params(self, K, b):
        self.K = K
        self.b = b

class BatchNorm(Layer):

    def __init__(self, n_features, eps=1e-5, momentum=0.9):
        self.eps = eps
        self.momentum = momentum
        self.training = False

        self.gamma = np.ones((1, n_features))
        self.beta  = np.zeros((1, n_features))
        
        self.running_mean = np.zeros((1, n_features))
        self.running_var  = np.ones((1, n_features))
    
    def forward(self, X, training):

        self.training = training

        # ===== Detect DNN vs CNN =====
        if X.ndim == 2:
            axes = (0,)
            reshape = (1, -1)
            m = X.shape[0]

        elif X.ndim == 4:
            axes = (0, 2, 3)
            reshape = (1, -1, 1, 1)
            m = X.shape[0] * X.shape[2] * X.shape[3]

        else:
            raise ValueError("Unsupported input shape")

        gamma = self.gamma.reshape(reshape)
        beta  = self.beta.reshape(reshape)

        if self.training:
            self.mu  = np.mean(X, axis=axes, keepdims=True)
            self.var = np.var(X, axis=axes, keepdims=True)

            self.X_centered = X - self.mu
            self.var_eps = self.var + self.eps
            self.std_inv = 1.0 / np.sqrt(self.var_eps)

            self.X_hat = self.X_centered * self.std_inv

            # running stats (always in (1, C))
            self.running_mean = (
                self.momentum * self.running_mean
                + (1 - self.momentum) * self.mu.reshape(1, -1)
            )
            self.running_var = (
                self.momentum * self.running_var
                + (1 - self.momentum) * self.var.reshape(1, -1)
            )

        else:
            mu  = self.running_mean.reshape(reshape)
            var = self.running_var.reshape(reshape)
            self.X_hat = (X - mu) / np.sqrt(var + self.eps)

        return gamma * self.X_hat + beta


    def backward(self, dY):

        if not self.training:
            raise RuntimeError("Backward called in inference mode")

        # ===== Detect DNN vs CNN =====
        if dY.ndim == 2:
            axes = (0,)
            reshape = (1, -1)
            m = dY.shape[0]

        elif dY.ndim == 4:
            axes = (0, 2, 3)
            reshape = (1, -1, 1, 1)
            m = dY.shape[0] * dY.shape[2] * dY.shape[3]

        else:
            raise ValueError("Unsupported input shape")

        gamma = self.gamma.reshape(reshape)

        # ===== Gradients =====
        dX_hat = dY * gamma

        dvar = np.sum(
            dX_hat * self.X_centered * -0.5 * self.var_eps**(-1.5),
            axis=axes, keepdims=True
        )

        dmu = (
            np.sum(dX_hat * -self.std_inv, axis=axes, keepdims=True)
            + dvar * np.sum(-2 * self.X_centered, axis=axes, keepdims=True) / m
        )

        dX = (
            dX_hat * self.std_inv
            + dvar * 2 * self.X_centered / m
            + dmu / m
        )

        # gamma / beta gradients (always in (1, C))
        self.dgamma = np.sum(dY * self.X_hat, axis=axes, keepdims=True).reshape(1, -1)
        self.dbeta  = np.sum(dY, axis=axes, keepdims=True).reshape(1, -1)

        return dX

    def get_params_update(self):
        return [(self.gamma, self.dgamma), (self.beta, self.dbeta)]
    

    def get_params_save(self):
        return self.gamma, self.beta, self.running_mean, self.running_var

    def set_params(self, gamma, beta, running_mean, running_var):
        self.gamma = gamma
        self.beta = beta
        self.running_mean = running_mean
        self.running_var = running_var
    
class Dropout(Layer):

    def __init__(self, dropout_per):
        self.dropout_per = dropout_per
        self.training = False

    def forward(self, A, training):
        
        self.training = training
        if training:
            self.M = (np.random.rand(*A.shape) > self.dropout_per).astype(A.dtype)
            return  self.M * A / (1 - self.dropout_per)
        
        else:
            return A
    
    def backward(self, dZ):
        
        training = self.training

        if training:
            return dZ * self.M / (1 - self.dropout_per)
        
        else:
            return dZ
        
class Block(Layer):

    def __init__(self, dense, batchnorm, activation, dropout):

        self.dense = dense
        self.batchnorm = batchnorm
        self.activation = activation
        self.dropout = dropout

    def forward(self, X, training=True):

        Z = self.dense.forward(X)
        Z = self.batchnorm.forward(Z, training)
        A = self.activation.forward(Z)
        A = self.dropout.forward(A, training)

        return A

    def backward(self, dZ):

        dA = self.dropout.backward(dZ)
        dZ = self.activation.backward(dA)
        dZ = self.batchnorm.backward(dZ)
        dZ = self.dense.backward(dZ)

        return dZ
    

class Dense(Layer):

    def __init__(self, nb_activation, nb_neuron):
        w_shape = (nb_activation, nb_neuron)
        b_shape = (1, nb_neuron)

        #Parameters
        self.W = np.random.randn(*w_shape) * 0.01
        self.b = np.zeros(b_shape)
        
        #Gradient
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        self.Wm = np.zeros(w_shape)
        self.Wv = np.zeros(w_shape)

        self.bm = np.zeros(b_shape)
        self.bv = np.zeros(b_shape)

    def forward(self, X):
        self.X = X
        return np.dot(X, self.W) + self.b

    def backward(self, dZ):
        dW = np.dot(self.X.T, dZ)
        db = np.sum(dZ, axis=0, keepdims=True)
        dA = np.dot(dZ, self.W.T)

        self.dW = dW
        self.db = db

        return dA

    def get_params_update(self):
        return [(self.W, self.dW), (self.b, self.db)]


    def get_params_save(self):
        return self.W, self.b

    def set_params(self, W, b):
        self.W = W
        self.b = b

class Flatten(Layer):

    def __int__(self):
        self.shape = None

    def forward(self, X):
        self.shape = X.shape
        return X.reshape(X.shape[0], -1)
    
    def backward(self, dZ):
        return dZ.reshape(self.shape)