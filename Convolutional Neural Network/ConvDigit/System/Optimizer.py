import  numpy as np

class Adam:

    def __init__(self, lr, beta1, beta2):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = 0
        self.state = {}

    def update(self, params):
        self.t += 1

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
            m = self.beta1 * m + (1 - self.beta1) * grad
            v = self.beta2 * v + (1 - self.beta2) * (grad * grad)

            m_hat = m / (1 - self.beta1**self.t)
            v_hat = v / (1 - self.beta2**self.t)

            param -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)

            self.state[key]["m"] = m
            self.state[key]["v"] = v