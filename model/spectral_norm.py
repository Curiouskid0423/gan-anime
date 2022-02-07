"""
Spectral-Norm GAN discriminator from the original paper (slightly modified).
Currently applied to a DC-GAN backbone.
"""

import torch
from torch import nn
from torch.nn.utils import spectral_norm
from model.dcgan import weights_init

class Discriminator(nn.Module):
    """
    Input shape: (N, 3, 64, 64)
    Output shape: (N, )
    """
    def __init__(self, in_dim, dim=64):
        super(Discriminator, self).__init__()
        # color_channels = 3 # RGB as of now.
        leak_ratio = 0.2
        def conv_bn_lrelu(in_dim, out_dim, kernel_size=5):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size, 2, 2),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(leak_ratio),
            )
        def conv_sn_lrelu(in_dim, out_dim, kernel_size, stride):
            return nn.Sequential(
                spectral_norm(nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding=2)),
                nn.LeakyReLU(leak_ratio)
            )

        self.conv1 = conv_bn_lrelu(in_dim, dim, 3) 
        self.conv2 = conv_sn_lrelu(dim, dim, 4, 2)
        self.conv3 = conv_sn_lrelu(dim, dim*2, 4, 2) 
        self.conv4 = conv_sn_lrelu(dim*2, dim*2, 3, 1) 
        self.conv5 = conv_sn_lrelu(dim*2, dim*4, 4, 1) 
        self.conv5_2 = conv_sn_lrelu(dim*4, dim*4, 3, 1) 
        self.conv6 = conv_sn_lrelu(dim*4, dim*8, 4, 1) 

        self.layers = nn.Sequential(
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4,
            self.conv5,
            self.conv5_2,
            self.conv6,
            nn.Conv2d(dim * 8, 1, 4),
        )

        self.apply(weights_init)

    def forward(self, x):
        y = self.layers(x)        
        return y.view(-1)