import cupy as cp

class Adam_GPU:

    def __init__(self, hyperparams):
        self.lr = hyperparams.lr
        self.beta1 =  hyperparams.beta1
        self.beta2 =  hyperparams.beta2
        self.t = 0
        self.state = {}

    def update(self, params):
        self.t += 1

        bias_correction1 = 1 - self.beta1 ** self.t
        bias_correction2 = 1 - self.beta2 ** self.t

        one_minus_beta1 = 1 - self.beta1
        one_minus_beta2 = 1 - self.beta2

        for param, grad in params:
            key = id(param)

            if key not in self.state:
                self.state[key] = {
                    "m": cp.zeros_like(param),
                    "v": cp.zeros_like(param)
                }

            m = self.state[key]["m"]
            v = self.state[key]["v"]

            # update moments (IN-PLACE pour perf GPU)
            m *= self.beta1
            m += (1 - self.beta1) * grad

            v *= self.beta2
            v += (1 - self.beta2) * (grad * grad)

            # bias correction
            m_hat = m / bias_correction1
            v_hat = v / bias_correction2

            # update param (in-place)
            param -= self.lr * m_hat / (cp.sqrt(v_hat) + self.eps)