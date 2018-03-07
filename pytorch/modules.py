import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
import time, os
from torch.autograd import Variable

def squash(s, dim=-1):
    norm2 = torch.sum(s**2, dim=dim, keepdim=True)
    norm = torch.sqrt(norm2)
    return (norm2 / (1.0 + norm2)) * (s / norm)
    
class PrimaryCaps(nn.Module):
    def __init__(self, use_cuda=False, out_channels=32, in_channels=256, ncaps=32*6*6, ndim=8, kernel_size=9, stride=2, padding=0):
        super(PrimaryCaps, self).__init__()
        self.ncaps = ncaps
        self.ndim = ndim
        self.caps = nn.ModuleList(
            [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding) for _ in
range(ndim)])
    
    def forward(self, x):
        u = torch.cat([cap(x).view(x.size(0), -1, 1) for cap in self.caps], dim=-1)
        # output (bs, ncaps, ndim)
        return squash(u)

    
class DigitCaps(nn.Module):
    def __init__(self, use_cuda=False, ncaps=10, ncaps_prev=32 * 6 * 6, ndim_prev=8, ndim=16):
        super(DigitCaps, self).__init__()
        self.use_cuda = use_cuda
        self.ndim_prev = ndim_prev
        self.ncaps_prev = ncaps_prev
        self.ncaps = ncaps
        self.route_iter = 3
        self.W = nn.Parameter(torch.randn(1, ncaps_prev, ncaps, ndim, ndim_prev))

    def forward(self, x):
        bs = x.size(0)
        x = torch.stack([x] * self.ncaps, dim=2).unsqueeze(-1)
        W = torch.cat([self.W] * bs, dim=0)
        u_hat = W @ x
        
        b = Variable(torch.zeros(1, self.ncaps_prev, self.ncaps, 1))
        if self.use_cuda:
            b = b.cuda()

        for i in range(self.route_iter):
            c = F.softmax(b)
            c = torch.cat([c] * bs, dim=0).unsqueeze(-1)

            s = (c * u_hat).sum(dim=1, keepdim=True)
            v = squash(s)
            
            if i < self.route_iter - 1:
                b = b + torch.matmul(u_hat.transpose(-1, -2), torch.cat([v] * self.ncaps_prev, dim=1)) \
                    .squeeze(-1).mean(dim=0, keepdim=True)
                return v.squeeze(1)
    
            
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(16*10,512),
            nn.ReLU(inplace=True),
            nn.Linear(512,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,784),
            nn.Sigmoid()
        )
    
    def forward(self,x):
        x = x.view(x.size(0),-1)
        x = self.net(x)
        return x
    
class CapsNet(nn.Module):
    def __init__(self, use_cuda=False, kernel_size=9, stride=1):
        super(CapsNet, self).__init__()
        
        self.conv1 = nn.Conv2d(1,256,kernel_size,stride=stride)
        self.primary_caps = PrimaryCaps(use_cuda=use_cuda)
        self.digit_caps = DigitCaps(use_cuda=use_cuda)
        self.decoder = Decoder()
        
    def forward(self, inpt):
        start = time.time()
        x = F.relu(self.conv1(inpt), inplace=True)
        x = self.primary_caps(x)
        x = self.digit_caps(x)
        reconst = self.decoder(x)
        return x, reconst

class CapsLoss(nn.Module):
    def __init__(self):
        super(CapsLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.reconst_factor = 0.0005
    def forward(self, scores, labels, reconst, inpt):
        norms = torch.sqrt(scores).squeeze()
        margin_loss = labels * ( F.relu(0.9 - norms, inplace=True) )**2 + 0.5*(1-labels) * ( F.relu(norms - 0.1, inplace=True) )**2
        margin_loss = margin_loss.sum(dim=-1).mean()
        reconst_loss = self.mse_loss(reconst.view(reconst.size(0),-1), inpt.view(inpt.size(0),-1))
        return margin_loss + self.reconst_factor * reconst_loss

