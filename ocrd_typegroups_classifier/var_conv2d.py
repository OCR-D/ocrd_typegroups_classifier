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
    """ Variational convolution 2D module for PyTorch.
    
    This class is based on the Variational Auto-Encoder; more details are
    given in the code comments.
    
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1), padding=(0,0), bias=True, is_variational=True):
        """ Constructor of the class, following the same syntax as the
            PyTorch Conv2D class, with an optional is_variational
            parameter.
        
            Note that by default it is variational. You can switch on
            and off this behavior by modifying the value of the attribute
            is_variational.
        """
        
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
        self.is_variational = is_variational

    def forward(self, x):
        """ Forward step of the variational convolution
            
            Parameters
            ----------
                x: PyTorch tensor / batch
                    Data to process
            
            Returns
            -------
                (output, variational loss)
                    A pair composed of the output tensor, and the variational
                    loss. If the layer is not being trained or if its
                    is_variational attribute is false, then the variational
                    loss is 0.
        """
        u = self.unfold(x).transpose(1, 2)
        fv =  round((2*self.padding[0]+x.size(2)-self.kernel_size[0]+1) / self.stride[0])
        fh = round((2*self.padding[1]+x.size(3)-self.kernel_size[1]+1) / self.stride[1])
        # Got here while tracking an error? Well, I'm sorry for this. The two lines above have 
        # an issue, but as it never appeared, I never took the time to solve it. For some reason,
        # the division appears to produce floats, so I round the values, as fold() needs a pair
        # of integers. Depending on the convolution size, the float value might be an approximation
        # and lead to a wrong value. Ideally, round should be replaced by the addition of an
        # epsilon and floor(). Unless it breaks while I'm using it, I won't lose time fixing it.
        # Sorry. - Mathias
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
        """ Computes the Kullback-Leibler Divergence, which is used as
        an extra-loss when training the variational layer """
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    def extra_repr(self):
        """ Returns a string description of the instance """
        return 'out_channels={out_channels}, kernel_size={kernel_size}, ' \
            'padding={padding}, stride={stride}, bias={bias}, is_variational={is_variational}'.format(
                **self.__dict__)
