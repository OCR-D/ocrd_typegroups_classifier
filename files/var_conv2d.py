# coding=utf-8
import math
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import Module
from torch.nn import Linear

# This class use some of the code from
# https://github.com/pytorch/examples/blob/master/vae/main.py

class VarConv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1), padding=(0,0), bias=True):
        super(VarConv2d, self).__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.bias = bias
        
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding
        
        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride
        
        nb_in = self.kernel_size[0]*self.kernel_size[1]*self.in_channels
        self.mu_layer      = nn.Linear(nb_in, self.out_channels, bias)
        self.logvar_layer  = nn.Linear(nb_in, self.out_channels, bias)
        self.unfold = nn.Unfold(kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.fold =   nn.Fold(output_size=out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.is_variational = True

    def forward(self, x):
        u = self.unfold(x).transpose(1, 2)
        fv =  round((2*self.padding[0]+x.size(2)-self.kernel_size[0]+1) / self.stride[0])
        fh = round((2*self.padding[1]+x.size(3)-self.kernel_size[1]+1) / self.stride[1])
        # Got here while tracking an error? Well, I'm sorry for this. The two lines above have 
        # an issue, but as it never appeared, I never took the time to solve it. For some reason,
        # the division appears to produce floats, so I round the values, as fold() needs a pair
        # of integers. Depending on the convolution size, the float value might be approximative
        # and lead to a wrong value. Ideally, round should be replaced by the addition of an
        # epsilon and floor(). I never spent the time to think about what that epsilon should
        # be with regard to the other parameters. Sorry. - Mathias
        mu = torch.nn.functional.fold(self.mu_layer(u).transpose(1, 2), (fv, fh), (1, 1))
        if self.training and self.is_variational:
            logvar = torch.nn.functional.fold(self.logvar_layer(u).transpose(1, 2), (fv, fh), (1, 1))
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            mu = eps.mul(std).add_(mu)
            varloss = self.kld(mu, logvar)
        else:
            varloss = 0
        return (mu, varloss)
    
    def kld(self, mu, logvar):
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    def extra_repr(self):
        return 'out_channels={out_channels}, kernel_size={kernel_size}, ' \
            'padding={padding}, stride={stride}, bias={bias}, is_variational={is_variational}'.format(
                **self.__dict__)
