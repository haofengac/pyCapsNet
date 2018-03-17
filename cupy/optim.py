import cupy as cp

class Optimizer:
    def __init__(self):
        self.t = 0
        
    def step(self):
        self.t += 1
        
    def update_val(self, x, dx):
        raise NotImplementedError
        
    def __call__(self, *input, **kwargs):
        return self.update_val(*input, **kwargs)
        
        
class AdamOptimizer(Optimizer):
    def __init__(self, lr=1e-2, beta=(0.9,0.999), eps=1e-8):
        super(AdamOptimizer, self).__init__()
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.m = None
        self.v = None
    
    def update_val(self, x, dx):
        self.m = cp.zeros_like(x)
        self.v = cp.zeros_like(x)
        m,v,lr,eps = self.m,self.v,self.lr,self.eps
        beta1, beta2 = self.beta
        m = beta1 * m + (1 - beta1) * dx
        v = beta2 * v + (1 - beta2) * dx**2
        alpha = lr * cp.sqrt(1 - beta2 ** self.t) / (1 - beta1 ** self.t)
        x -= alpha * (m / (cp.sqrt(v) + eps))
        self.m = m
        self.v = v
        return x
        
        