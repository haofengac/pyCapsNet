
# im2col functions adapted from https://github.com/Burton2000/CS231n-2017/blob/master/assignment2/cs231n/im2col.py

import numpy as np
import time, os


def tile(arr, copy, axis):
    return np.concatenate([arr] * copy, axis=axis)

        
class Module(object):
    def __init__(self, trainable=False):
        self.trainable = trainable
        pass
    
    def forward(self, x):
        raise NotImplementedError
        
    def backward(self, grad, optimizer=None):
        raise NotImplementedError
        
    def __call__(self, *input, **kwargs):
        return self.forward(*input, **kwargs)

    
class Sequence(Module):
    def __init__(self, modules):
        self._modules = modules
        
    def forward(self, inpt):
        t = time.time()
        for module in self._modules:
            inpt = module(inpt)
            cur = time.time()
            t = cur
            if module.trainable:
                self.trainable = True
        return inpt
    
    def backward(self, grad, optimizer=None):
        for module in self._modules[::-1]:
            if module.trainable:
                grad = module.backward(grad, optimizer)
            else:
                grad = module.backward(grad)
            
        return grad
    
    def modules(self):
        return self._modules
    
    def trainable_modules(self):
        return [i for i in self._modules if i.trainable]

    
class Linear(Module):
    def __init__(self, in_channel, out_channel):
        super(Linear, self).__init__(trainable=True)
        std = 1/np.sqrt(in_channel)
        self.w = np.random.uniform(-std, std, (out_channel, in_channel))
        self.b = np.random.uniform(-std, std, (1, out_channel))
        self.x = None
        
    def _set_params(self, params):
        w, b = params
        self.w = w
        self.b = b
        if len(self.b.shape) < 2:
            self.b = self.b[None,:]
        
    def forward(self, x):
        out = x.dot(self.w.T) + self.b
        self.x = x
        return out
    
    def backward(self, grad, optimizer=None):
        dw = (self.x.T @ grad).T
        db = np.sum(grad, axis=0, keepdims=True)
        # update parameters
        if optimizer is not None:
            self.w = optimizer(self.w, dw)
            self.b = optimizer(self.b, db)
        
        dx = grad @ self.w
        dx = np.reshape(dx, self.x.shape)
        return dx
   

class ReLU(Module):
    def __init__(self, alpha=0):
        super(ReLU, self).__init__()
        self.alpha = alpha
        self.x = None
        
    def forward(self, x):
        out = x.copy()
        if self.alpha > 0:
            out[out<0] = self.alpha*x
        else:
            out[out<0] = 0
        self.x = x
        return out
    
    def backward(self, grad):
        dx = grad.copy()
        dx[self.x < 0] = 0
        return dx

class Sigmoid(Module):
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.s = None
        
    def forward(self, x):
        self.s = 1/(1 + np.exp(-x))
        return self.s
    
    def backward(self, grad):
        return grad * (self.s * (1-self.s))

    
class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, pad=0, eps=1e-4):
        super(Conv2d, self).__init__(trainable=True)
        self.ic = in_channels
        self.oc = out_channels
        self.k = kernel_size
        self.s = stride
        self.p = pad
        
        std = 1/(np.sqrt(self.ic* self.k**2))
        self.W = np.random.uniform(-std, std, (self.oc,self.ic,self.k,self.k))
        self.b = np.random.uniform(-std, std, (self.oc, 1))
        
        self.X_col = None
        self.x_shape = None
        
    def _set_params(self, params):
        W, b = params
        self.W = W
        self.b = b
        
    def forward(self, X):
        NF, CF, HF, WF = self.W.shape
        NX, DX, HX, WX = X.shape
        self.x_shape = X.shape
        h_out = int((HX - HF + 2 * self.p) / self.s + 1)
        w_out = int((WX - WF + 2 * self.p) / self.s + 1)

        X_col = self.im2col_indices(X)
        self.X_col = X_col
        W_col = self.W.reshape(NF, -1)

        out = W_col @ self.X_col + self.b
        out = out.reshape(NF, h_out, w_out, NX)
        out = out.transpose(3, 0, 1, 2)

        return out


    def backward(self, dout, optimizer=None):
        NF, CF, HF, WF = self.W.shape

        db = np.sum(dout, axis=(0, 2, 3))
        db = db.reshape(NF, -1)

        dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(NF, -1)
        dW = dout_reshaped @ self.X_col.T
        dW = dW.reshape(self.W.shape)
        
        if optimizer is not None:
            self.b = optimizer(self.b, db)
            self.W = optimizer(self.W, dW)

        W_reshape = self.W.reshape(NF, -1)
        dX_col = W_reshape.T @ dout_reshaped
        dX = self.col2im_indices(dX_col)

        return dX

    def get_im2col_indices(self):
        padding, stride, field_height, field_width, x_shape = self.p, self.s, self.k, self.k, self.x_shape
        N, C, H, W = x_shape
#         assert (H + 2 * padding - field_height) % stride == 0
#         assert (W + 2 * padding - field_height) % stride == 0
        out_height = int((H + 2 * padding - field_height) / stride + 1)
        out_width = int((W + 2 * padding - field_width) / stride + 1)

        i0 = np.repeat(np.arange(field_height), field_width)
        i0 = np.tile(i0, C)
        i1 = stride * np.repeat(np.arange(out_height), out_width)
        j0 = np.tile(np.arange(field_width), field_height * C)
        j1 = stride * np.tile(np.arange(out_width), out_height)
        i = i0.reshape(-1, 1) + i1.reshape(1, -1)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)

        k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)
        
        return (k.astype(np.int), i.astype(np.int), j.astype(np.int))


    def im2col_indices(self, x):
        p, stride, field_height, field_width = self.p, self.s, self.k, self.k
        x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

        k, i, j = self.get_im2col_indices()

        cols = x_padded[:, k, i, j]
        C = x.shape[1]
        cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
        return cols


    def col2im_indices(self, cols):
        field_height, field_width, padding, stride = self.k, self.k, self.p, self.s
        N, C, H, W = self.x_shape
        H_padded, W_padded = H + 2 * padding, W + 2 * padding
        x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
        k, i, j = self.get_im2col_indices()
        cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
        cols_reshaped = cols_reshaped.transpose(2, 0, 1)
        np.scatter_add(x_padded, (slice(None), k, i, j), cols_reshaped)
        if padding == 0:
            return x_padded
        return x_padded[:, :, padding:-padding, padding:-padding]

    
class Softmax(Module):
    def __init__(self, dim=-1):
        super(Softmax, self).__init__()
        self.s = None
        self.dim = dim
        self.squeeze_len = None
        
    def forward(self, x, dim=None):
        if dim is not None:
            self.dim = dim
        if self.dim < 0:
            self.dim = len(x.shape)+self.dim
        self.squeeze_len = x.shape[self.dim]
        y = np.exp(x)
        s = y/np.sum(y, axis=self.dim, keepdims=True)
        self.s = s
        return s
    
    def backward(self, grad): 
        self.s = np.expand_dims(self.s.swapaxes(self.dim,-1), -1)
        grad = np.expand_dims(grad.swapaxes(self.dim,-1), -1)
        mat = self.s @ self.s.swapaxes(-1,-2)
        mat = (-mat + np.eye(mat.shape[-1]) * (mat**0.5))
        grad = mat @ grad
        self.s = self.s.swapaxes(self.dim,-1).squeeze(-1)
        return grad.swapaxes(self.dim,-2).squeeze(-1)
    
    
class Squash(Module):
    def __init__(self, dim=-1):
        super(Squash, self).__init__()
        self.dim = dim
        self.squeeze_len = None
        self.s = None
        
    def forward(self, s):
        self.s = s
        self.squeeze_len = s.shape[self.dim]
        norm2 = np.sum((s)**2, axis=self.dim, keepdims=True)
        return (np.sqrt(norm2) / (1.0 + norm2)) * s
        
    def backward(self, grad):
        norm2 = np.sum((self.s)**2, axis=self.dim, keepdims=True)
        norm = np.sqrt(norm2)
        temp = tile((1/(2*(1.+norm2)*norm) - norm/(1.+norm2)**2), self.squeeze_len, self.dim)
        dnorm2 = np.sum(self.s * temp, axis=-1, keepdims=True)
        factor = norm/(1+norm2)
        return grad * dnorm2 * (2.*self.s) + grad * factor

class MSELoss(Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.x = None
        self.y = None
        
    def forward(self, x, y):
        self.x = x
        self.y = y
        return np.sum((x - y)**2)/float(x.size), 2*(x - y)/float(x.size)


class PrimaryCaps(Module):
    def __init__(self, use_cuda=False, out_channels=32, in_channels=256, mapsize=6, ndim=8, kernel_size=9, stride=2, padding=0):
        super(PrimaryCaps, self).__init__(trainable=True)
        self.ndim = ndim
        self.caps = [Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, pad=padding) for _ in
range(ndim)]
        
        self.out_channels = out_channels
        self.mapsize = mapsize
        self.ncaps = out_channels * mapsize**2
        self.squash = Squash()
        self.x_size = None
    
    def _set_params(self, params):
        for i, c in enumerate(self.caps):
            c._set_params(params[i])
    
    def forward(self, x):
        t = time.time()
        # output (bs, ncaps, ndim)
        self.x_size = x.shape
        u = np.concatenate([cap(x).reshape((x.shape[0], -1, 1)) for cap in self.caps], axis=-1)
        return self.squash(u)
    
    def backward(self, grads, optimizer=None):
        t = time.time()
        grads = self.squash.backward(grads)
        grads = grads.reshape((self.x_size[0],self.out_channels, self.mapsize, self.mapsize,-1))
        grads = np.concatenate([np.expand_dims(self.caps[i].backward(
            grads[:,:,:,:,i], optimizer=optimizer), -1) for i in range(self.ndim)], axis=-1)
        out = np.sum(grads, axis=-1)
        return out
     
        
class Decoder(Module):
    def __init__(self):
        super(Decoder, self).__init__(trainable=True)
        self.net = Sequence([
            Linear(16*10,512),
            ReLU(),
            Linear(512,1024),
            ReLU(),
            Linear(1024,784),
            Sigmoid()
        ])
        self.x_shape = None
    
    def forward(self, x):
        self.x_shape = x.shape
        x = x.reshape(x.shape[0],-1)
        
        return self.net(x)
    
    def _set_params(self, params):
        for i, l in enumerate(self.net.trainable_modules()):
            l._set_params(params[i])
    
    def backward(self, grad, optimizer):
        return self.net.backward(grad, optimizer).reshape(self.x_shape)
    

class DigitCaps(Module):
    def __init__(self, ncaps=10, ncaps_prev=32 * 6 * 6, ndim_prev=8, ndim=16):
        super(DigitCaps, self).__init__(trainable=True)
        self.ndim_prev = ndim_prev
        self.ncaps_prev = ncaps_prev
        self.ncaps = ncaps
        self.route_iter = 2
        self.W = np.random.randn(1, ncaps_prev, ncaps, ndim, ndim_prev)
        self.softmaxs = [Softmax() for _ in range(self.route_iter)]
        self.squashs = [Squash() for _ in range(self.route_iter)]
        self.u_hat = None
        self.bs = None
        self.b = [None] * self.route_iter
        self.v = [None] * self.route_iter
        self.x = None
        
    def _set_params(self, params):
        self.W = params

    def forward(self, x):
        t = time.time()
        self.bs = x.shape[0]
        self.x = x
        x = tile(x[:,:,None,:,None], self.ncaps, 2)
        W = tile(self.W, self.bs, 0)
        u_hat = W @ x
        self.u_hat = u_hat
        b = np.zeros((1, self.ncaps_prev, self.ncaps, 1, 1))

        for r in range(self.route_iter):
            self.b[r] = b
            c = self.softmaxs[r](b, dim=1)

            c = tile(c, self.bs, 0)
            s = np.sum(c * u_hat, axis=1, keepdims=True)
            v = self.squashs[r](s)
            if r == self.route_iter - 1:
                return np.squeeze(v, axis=1)
            
            self.v[r] = v
            p = u_hat.swapaxes(-1, -2) @ tile(v, self.ncaps_prev, 1)
            b = b + np.mean(p, axis=0, keepdims=True)
                
            
    def backward(self, grad, optimizer=None):
        t = time.time()
        grad_accum = np.zeros_like(self.u_hat)
        b_grad_accum = None
        grad = grad[:,None,:,:,:]
        for r in range(self.route_iter)[::-1]:
            if r < self.route_iter-1:
                grad = b_grad_accum
                grad = tile(grad, self.bs, 0)/self.bs
                p_grad = tile(self.v[r], self.ncaps_prev, 1) * grad

                grad_accum += p_grad
                
                grad = self.u_hat * grad
                grad = np.sum(grad, axis=1, keepdims=True)

            grad = self.squashs[r].backward(grad)
            grad = tile(grad, self.ncaps_prev, 1)
            c = self.softmaxs[r].s
            grad_accum += tile(c, self.bs, 0) * grad
            grad = self.u_hat.swapaxes(-1,-2) @ grad
            
            if r > 0:
                grad = np.sum(grad, axis=0, keepdims=True)
                grad = self.softmaxs[r].backward(grad)
                if b_grad_accum is None:
                    b_grad_accum = grad
                else:
                    b_grad_accum += grad
        
        x = tile(self.x[:,:,None,:,None], self.ncaps, 2)
        dW = np.sum(grad_accum @ x.swapaxes(-1,-2), axis=0, keepdims=True)
        if optimizer is not None:
            self.W = optimizer(self.W, dW)

        grad_accum = np.squeeze(self.W.swapaxes(-1,-2) @ grad_accum, axis=-1)
        dx = np.sum(grad_accum, axis=2)
        return dx
    

class CapsNet(Module):
    def __init__(self, use_cuda=False, kernel_size=9, stride=1):       
        super(CapsNet, self).__init__(trainable=True)
        self.net = Sequence([
            Conv2d(1,256,kernel_size=kernel_size,stride=stride),
            ReLU(),
            PrimaryCaps(),
            DigitCaps()
        ])
        self.decoder = Decoder()
        self.x = None
        self.digit_ndim = 16
        self.softmax = Softmax()
        
    def _set_params(self, params):
        for i, m in enumerate(self.net.trainable_modules() + [self.decoder]):
            m._set_params(params)
        
    def forward(self, x):
        x = self.net(x)
        self.x = x
        reconst = self.decoder(x)
        scores = np.sqrt((x ** 2).sum(2)).squeeze()
        return scores, reconst
    
    def backward(self, grad, optimizer):
        scores_grad, reconst_grad = grad

        scores_grad = scores_grad[:,:,None, None]
        t = 0.5 * ((self.x ** 2).sum(2, keepdims=True) ** (-0.5))
        scores_grad *= 0.5 * ((self.x ** 2).sum(2, keepdims=True) ** (-0.5))
        scores_grad = tile(scores_grad, self.digit_ndim, 2) # tile at dimension 2
        scores_grad *= 2*self.x
        t = time.time()

        reconst_grad = self.decoder.backward(reconst_grad, optimizer)    
        grad = scores_grad + reconst_grad

        grad = self.net.backward(grad, optimizer=optimizer)
        return grad
        
        
class CapsLoss(Module):
    def __init__(self):
        super(CapsLoss, self).__init__()
        self.mse_loss = MSELoss()
        self.relu1 = ReLU()
        self.relu2 = ReLU()
        self.reconst_factor = 0.0005
        
        
    def forward(self, norms, labels, reconst, inpt):
        self.labels = labels

        int1 = self.relu1(0.9 - norms)
        int2 = self.relu2(norms - 0.1)
        margin_loss = labels * int1**2 + 0.5*(1-labels) * int2**2
        bs, ndim_prev = margin_loss.shape[0], margin_loss.shape[-1]
        margin_loss = np.sum(margin_loss, axis=-1).mean()
        
        reconst_loss, reconst_grad = self.mse_loss(reconst.reshape(reconst.shape[0],-1), inpt.reshape(inpt.shape[0],-1))
        loss = margin_loss + self.reconst_factor * reconst_loss
        
        margin_grad = np.ones((bs, ndim_prev)) / float(bs)
        margin_grad_pos = -self.relu1.backward(margin_grad * labels * (2*int1))
        margin_grad_neg = self.relu2.backward(margin_grad * 0.5*(1-labels) * (2*int2))

        margin_grad = margin_grad_pos + margin_grad_neg
        reconst_grad *= self.reconst_factor
        
        return loss, (margin_grad, reconst_grad)