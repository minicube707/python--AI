import numpy as np

class Adam_CPU:

    def __init__(self, hyperparams):
        self.lr = hyperparams.lr
        self.beta1 =  hyperparams.beta1
        self.beta2 =  hyperparams.beta2
        self.t = 0
        self.state = {}

    def update(self, params):
        self.t += 1

        one_minus_beta1 = 1 - self.beta1
        one_minus_beta2 = 1 - self.beta2
        
        bias_correction1 = 1 - (self.beta1 ** self.t)
        bias_correction2 = 1 - (self.beta2 ** self.t)


        for param, grad in params:
            key = id(param)

            if key not in self.state:
                self.state[key] = {
                    "m": np.zeros_like(param),
                    "v": np.zeros_like(param)
                }

            m = self.state[key]["m"]
            v = self.state[key]["v"]

            # update Adam
            m = self.beta1 * m + one_minus_beta1 * grad
            v = self.beta2 * v + one_minus_beta2 * (grad * grad)

            m_hat = m / bias_correction1
            v_hat = v / bias_correction2

            param -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)

            self.state[key]["m"] = m
            self.state[key]["v"] = v