# sigmoid.pyx

# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
from libc.math cimport exp, sqrt, fmin, fmax

def sigmoide(np.ndarray[np.float64_t, ndim=3] X):
    cdef Py_ssize_t i, j, k
    cdef Py_ssize_t d1 = X.shape[0]
    cdef Py_ssize_t d2 = X.shape[1]
    cdef Py_ssize_t d3 = X.shape[2]

    cdef np.ndarray[np.float64_t, ndim=3] result = np.empty((d1, d2, d3), dtype=np.float64)

    cdef double[:,:,:] X_view = X
    cdef double[:,:,:] res_view = result

    for i in range(d1):
        for j in range(d2):
            for k in range(d3):
                res_view[i, j, k] = 1.0 / (1.0 + exp(-X_view[i, j, k]))

    return result



def relu(np.ndarray[np.float64_t, ndim=3] X):

    cdef Py_ssize_t i, j, k
    cdef Py_ssize_t d1 = X.shape[0]
    cdef Py_ssize_t d2 = X.shape[1]
    cdef Py_ssize_t d3 = X.shape[2]

    cdef np.ndarray[np.float64_t, ndim=3] result = np.empty((d1, d2, d3), dtype=np.float64)

    cdef double[:,:,:] X_view = X
    cdef double[:,:,:] res_view = result

    for i in range(d1):
        for j in range(d2):
            for k in range(d3):
                res_view[i, j, k] = X_view[i, j, k] if X_view[i, j, k] > 0.0 else 0.0

    return result



def max_pooling(np.ndarray[np.float64_t, ndim=3] X):

    cdef Py_ssize_t batch = X.shape[0]
    cdef Py_ssize_t flat_h = X.shape[1]
    cdef Py_ssize_t k = X.shape[2]

    cdef Py_ssize_t h = <Py_ssize_t>sqrt(<double>flat_h)
    if h * h != flat_h:
        raise ValueError("Shape[1] must be a perfect square.")

    cdef np.ndarray[np.float64_t, ndim=3] result = np.empty((batch, h, h), dtype=np.float64)
    cdef double[:,:,:] X_view = X
    cdef double[:,:,:] res_view = result
    cdef double max_val

    cdef Py_ssize_t b, i, j, d
    for b in range(batch):
        for i in range(flat_h):
            max_val = X_view[b, i, 0]
            for d in range(1, k):
                if X_view[b, i, d] > max_val:
                    max_val = X_view[b, i, d]
            # Placement dans (h, h)
            res_view[b, i // h, i % h] = max_val

    return result



def correlate(np.ndarray[np.float64_t, ndim=3] A,
              np.ndarray[np.float64_t, ndim=4] K,
              np.ndarray[np.float64_t, ndim=1] b,
              int x_size):

    cdef Py_ssize_t L_A = A.shape[0]
    cdef Py_ssize_t NB_Dot_Product = A.shape[1]
    cdef Py_ssize_t K_Size = A.shape[2]

    cdef Py_ssize_t NB_K = K.shape[0]
    cdef Py_ssize_t L_K = K.shape[1]
    cdef Py_ssize_t one = K.shape[3]  # devrait être 1

    cdef np.ndarray[np.float64_t, ndim=3] Z = np.zeros((NB_K, NB_Dot_Product, one), dtype=np.float64)

    cdef double[:,:,:] A_view = A
    cdef double[:,:,:,:] K_view = K
    cdef double[:] b_view = b
    cdef double[:,:,:] Z_view = Z

    cdef Py_ssize_t i, j, d, k
    cdef double acc

    for i in range(NB_K):
        for d in range(NB_Dot_Product):
            acc = 0.0
            for j in range(L_A):
                for k in range(K_Size):
                    acc += A_view[j, d, k] * K_view[i, j, k, 0]
            Z_view[i, d, 0] = acc + b_view[i]

    # Reshape Z to (NB_K, x_size, x_size)
    cdef np.ndarray[np.float64_t, ndim=3] Z_final = np.empty((NB_K, x_size, x_size), dtype=np.float64)
    cdef double[:,:,:] Zf_view = Z_final

    cdef Py_ssize_t m
    for i in range(NB_K):
        for m in range(x_size * x_size):
            Zf_view[i, m // x_size, m % x_size] = Z_view[i, m, 0]

    # Clip values between -88 and 88
    for i in range(NB_K):
        for j in range(x_size):
            for k in range(x_size):
                Zf_view[i, j, k] = fmax(-88.0, fmin(88.0, Zf_view[i, j, k]))

    return Z_final



def convolution(np.ndarray[np.float64_t, ndim=3] dZ,
                np.ndarray[np.float64_t, ndim=4] K,
                int k_size_sqrt):

    cdef Py_ssize_t nb_kernels = K.shape[0]
    cdef Py_ssize_t nb_channels = K.shape[1]
    cdef Py_ssize_t k_size = k_size_sqrt

    cdef Py_ssize_t out_h = dZ.shape[1] + k_size - 1
    cdef Py_ssize_t out_w = dZ.shape[2] + k_size - 1

    # Padding dZ avec 0
    cdef np.ndarray[np.float64_t, ndim=3] new_dZ = np.zeros((dZ.shape[0],
                                                            dZ.shape[1] + 2*(k_size - 1),
                                                            dZ.shape[2] + 2*(k_size - 1)), dtype=np.float64)

    cdef Py_ssize_t i, j, c, h, w, ki, kj

    # Copier dZ dans new_dZ avec padding
    cdef double[:,:,:] dZ_view = dZ
    cdef double[:,:,:] new_dZ_view = new_dZ

    for i in range(dZ.shape[0]):
        for j in range(dZ.shape[1]):
            for c in range(dZ.shape[2]):
                new_dZ_view[i, j + k_size - 1, c + k_size - 1] = dZ_view[i, j, c]

    # Initialiser la sortie
    cdef np.ndarray[np.float64_t, ndim=3] next_dZ = np.zeros((nb_channels, out_h, out_w), dtype=np.float64)
    cdef double[:,:,:] next_dZ_view = next_dZ

    # Accès aux kernels et flip
    cdef double[:,:,:,:] K_view = K

    # Convolution full
    cdef double s
    for a in range(nb_kernels):
        for b in range(nb_channels):
            for h in range(out_h):
                for w in range(out_w):
                    s = 0.0
                    for ki in range(k_size):
                        for kj in range(k_size):
                            s += new_dZ_view[a, h + ki, w + kj] * K_view[a, b, k_size - 1 - ki, k_size - 1 - kj]
                    next_dZ_view[b, h, w] += s

    return next_dZ



def kernel_activation(np.ndarray[np.float64_t, ndim=3] A,
                      np.ndarray[np.float64_t, ndim=4] K,
                      np.ndarray[np.float64_t, ndim=1] b,
                      int x_size,
                      str mode):
    cdef np.ndarray[np.float64_t, ndim=3] Z

    Z = correlate(A, K, b, x_size)

    if mode == 0:
        Z = relu(Z)
    elif mode == 3:
        Z = sigmoide(Z)
    
    return Z