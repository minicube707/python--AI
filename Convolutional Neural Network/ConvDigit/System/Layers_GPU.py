
import cupy as cp

from .Layer import Layer
from .Mathematical_function_GPU import add_padding_GPU
from .Mathematical_function import remove_padding

class MaxPooling_GPU(Layer):
    
    def __init__(self, k_size, stride, padding):
        self.k_size = k_size
        self.stride = stride
        self.padding = padding
        self.X = None
        self.class_ = "MaxPooling"
        
    def forward(self, X):
        
        padding = self.padding

        # ========================
        # Padding (GPU safe)
        # ========================
        if padding > 0:
            X = add_padding_GPU(X, padding)

        self.X = X
        k = self.k_size
        s = self.stride

        B, C, H, W = X.shape

        # ========================
        # Output size
        # ========================
        H_out = (H - k) // s + 1
        W_out = (W - k) // s + 1

        # ========================
        # im2col via as_strided
        # ========================
        shape = (B, C, H_out, W_out, k, k)

        strides = (
            X.strides[0],
            X.strides[1],
            s * X.strides[2],
            s * X.strides[3],
            X.strides[2],
            X.strides[3]
        )

        windows = cp.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)

        B, C, H_out, W_out, k, _ = windows.shape

        # max
        out = windows.max(axis=(-1, -2))

        # argmax (flatten k*k)
        self.argmax = windows.reshape(B, C, H_out, W_out, -1).argmax(axis=-1)

        return out

    def backward(self, dA):

        X = self.X
        k = self.k_size
        s = self.stride
        padding = self.padding
        argmax = self.argmax

        B, C, H_out, W_out = dA.shape

        dX = cp.zeros_like(X)

        # convertir index → (i, j)
        i_idx = argmax // k
        j_idx = argmax % k

        # boucle seulement sur k (OK GPU)
        for i in range(k):
            for j in range(k):

                mask = (i_idx == i) & (j_idx == j)

                dX[:, :, 
                i:i + H_out * s:s,
                j:j + W_out * s:s
                ] += dA * mask

        # remove padding
        dX = remove_padding(dX, padding)
       
        return dX


class Convolution_GPU(Layer):

    def __init__(self, nb_kernel, nb_layer, k_size, stride, o_size, padding):
        
        k_shape = (nb_kernel, nb_layer, k_size, k_size)
        b_shape = (nb_kernel, o_size, o_size)
        
        self.K = cp.random.randn(*k_shape) * 0.01
        self.b = cp.zeros(b_shape)
        self.X = None

        self.dK = cp.zeros(k_shape)
        self.db = cp.zeros(b_shape)
        self.stride = stride
        self.padding = padding
        self.windows = None
        self.class_ = "Convolution"

    def forward(self, X):

        stride  = self.stride
        padding = self.padding

        B, C, H, W = X.shape
        N, _, Kh, Kw = self.K.shape

        # ========================
        # Padding (GPU safe)
        # ========================
        if padding > 0:
            X = add_padding_GPU(X, padding)

        self.X = X

        # ========================
        # Output size
        # ========================
        H_out = (H + padding - Kh) // stride + 1
        W_out = (W + padding - Kw) // stride + 1

        # ========================
        # im2col (vectorisé GPU)
        # ========================
        shape = (B, C, Kh, Kw, H_out, W_out)

        strides = (
            X.strides[0],
            X.strides[1],
            X.strides[2],
            X.strides[3],
            stride * X.strides[2],
            stride * X.strides[3]
        )

        windows = cp.lib.stride_tricks.as_strided(
            X,
            shape=shape,
            strides=strides
        )

        self.windows = windows

        # ========================
        # Convolution (GPU)
        # ========================
        out = cp.einsum('nckl,bcklhw->bnhw', self.K, windows)

        # Ajout biais
        out += self.b

        return out


    def backward(self, dZ):

        stride  = self.stride
        padding = self.padding

        B, N, H_out, W_out = dZ.shape
        _, C, Kh, Kw = self.K.shape

        X = self.X
        windows = self.windows

        # ========================
        # Parameter Gradients
        # ========================

        # dK
        self.dK = cp.tensordot(
            dZ,
            windows,
            axes=([0, 2, 3], [0, 4, 5])
        )

        # db
        self.db = cp.sum(dZ, axis=0)

        # ========================
        # Gradient input (dX)
        # ========================

        # Step 1: appliquer les filtres (équivalent convolution backward)
        # (B, N, H_out, W_out) x (N, C, Kh, Kw)
        dZK = cp.einsum('bnhw,nckl->bchwkl', dZ, self.K)
        # → (B, C, H_out, W_out, Kh, Kw)

        # Step 2: reconstruction (col2im optimisé)
        dX = cp.zeros_like(X)

        for i in range(Kh):
            for j in range(Kw):
                dX[:, :, 
                i:i + H_out * stride:stride,
                j:j + W_out * stride:stride
                ] += dZK[:, :, :, :, i, j]

        # ========================
        # Remove padding
        # ========================
        dX = remove_padding(dX, padding)

        return dX
    
    def get_params_update(self):
        return [(self.K, self.dK), (self.b, self.db)]
    

    def get_params_save(self):
        return cp.asnumpy(self.K), cp.asnumpy(self.b)
    

    def set_params(self, K, b):
        self.K = cp.asarray(K)
        self.b = cp.asarray(b)

class BatchNorm_GPU(Layer):

    def __init__(self, n_features, eps=1e-5, momentum=0.9):
        self.eps = eps
        self.momentum = momentum
        self.training = False

        self.gamma = cp.ones((1, n_features))
        self.beta  = cp.zeros((1, n_features))
        
        self.running_mean = cp.zeros((1, n_features))
        self.running_var  = cp.ones((1, n_features))
        self.class_ = "BatchNorm"
        
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
            self.mu  = cp.mean(X, axis=axes, keepdims=True)
            self.var = cp.var(X, axis=axes, keepdims=True)

            self.X_centered = X - self.mu
            self.var_eps = self.var + self.eps
            self.std_inv = 1.0 / cp.sqrt(self.var_eps)

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
            self.X_hat = (X - mu) / cp.sqrt(var + self.eps)

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

        dvar = cp.sum(
            dX_hat * self.X_centered * -0.5 * self.var_eps**(-1.5),
            axis=axes, keepdims=True
        )

        dmu = (
            cp.sum(dX_hat * -self.std_inv, axis=axes, keepdims=True)
            + dvar * cp.sum(-2 * self.X_centered, axis=axes, keepdims=True) / m
        )

        dX = (
            dX_hat * self.std_inv
            + dvar * 2 * self.X_centered / m
            + dmu / m
        )

        # gamma / beta gradients (always in (1, C))
        self.dgamma = cp.sum(dY * self.X_hat, axis=axes, keepdims=True).reshape(1, -1)
        self.dbeta  = cp.sum(dY, axis=axes, keepdims=True).reshape(1, -1)

        return dX

    def get_params_update(self):
        return [(self.gamma, self.dgamma), (self.beta, self.dbeta)]
    

    def get_params_save(self):
        return cp.asnumpy(self.gamma), cp.asnumpy(self.beta), cp.asnumpy(self.running_mean), cp.asnumpy(self.running_var)

    def set_params(self, gamma, beta, running_mean, running_var):
        self.gamma = cp.asarray(gamma)
        self.beta = cp.asarray(beta)
        self.running_mean = cp.asarray(running_mean)
        self.running_var = cp.asarray(running_var)
    
class Dropout_GPU(Layer):

    def __init__(self, dropout_per):
        self.dropout_per = dropout_per
        self.training = False
        self.class_ = "Dropout"
        
    def forward(self, A, training):
        
        self.training = training

        if training:
            self.M = (cp.random.rand(*A.shape) > self.dropout_per).astype(A.dtype)
            return  self.M * A / (1 - self.dropout_per)
        
        else:
            return A
    
    def backward(self, dZ):
        
        training = self.training

        if training:
            return dZ * self.M / (1 - self.dropout_per)
        
        else:
            return dZ
        
class Dense_GPU(Layer):

    def __init__(self, nb_activation, nb_neuron):
        w_shape = (nb_activation, nb_neuron)
        b_shape = (1, nb_neuron)

        #Parameters
        self.W = cp.random.randn(*w_shape) * 0.01
        self.b = cp.zeros(b_shape)
        
        #Gradient
        self.dW = cp.zeros_like(self.W)
        self.db = cp.zeros_like(self.b)

        self.Wm = cp.zeros(w_shape)
        self.Wv = cp.zeros(w_shape)

        self.bm = cp.zeros(b_shape)
        self.bv = cp.zeros(b_shape)

        self.class_ = "Dense"
        
    def forward(self, X):
        self.X = X
        return cp.dot(X, self.W) + self.b

    def backward(self, dZ):
        dW = cp.dot(self.X.T, dZ)
        db = cp.sum(dZ, axis=0, keepdims=True)
        dA = cp.dot(dZ, self.W.T)

        self.dW = dW
        self.db = db

        return dA

    def get_params_update(self):
        return [(self.W, self.dW), (self.b, self.db)]


    def get_params_save(self):
        return cp.asnumpy(self.W), cp.asnumpy(self.b)

    def set_params(self, W, b):
        self.W = cp.asarray(W)
        self.b = cp.asarray(b)

class GlobalAveragePooling_GPU:
    
    def __init__(self):
        self.class_ = "GlobalAveragePooling"
        
    def forward(self, X):
        self.input_shape = X.shape
        return cp.mean(X, axis=(2, 3))

    def backward(self, grad_output):
      
        batch_size, channels, height, width = self.input_shape
        
        # Chaque pixel reçoit la même part du gradient
        grad_input = grad_output[:, :, None, None] / (height * width)
        
        # Broadcast sur toute la map
        grad_input = cp.ones((batch_size, channels, height, width)) * grad_input
        
        return grad_input