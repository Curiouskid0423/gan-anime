"""
Trainer file for StyleGAN.
Define all the hyperparameters.
"""
import os
import torch
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model.stylegan.synthesis_net import StyledGenerator
from model.stylegan.discriminator import Discriminator
from data.data_loader import *
import matplotlib.pyplot as plt

class StyleTrainer:

    def __init__(self, workspace_dir='.', ckpt_save_path=None, log_save_path=None, 
                z_dim=100, batch_size=64, lr=1e-4):
        self.checkpoint_save_path = ckpt_save_path

        """ Hyperparameter definition for StyleGAN """
        self.batch_size = batch_size
        # self.z_dim = z_dim

    def requires_grad(self, model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag

    def accumulate(self, model1, model2, decay=0.999):
        """
        In Progressive GAN the author used running average of generator when inferring 
        samples. accumulate() calculates and saves the running average to g_running.
        """
        par1 = dict(model1.named_parameters())
        par2 = dict(model2.named_parameters())

        for k in par1.keys():
            par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)

    # def training_loop(self, lr )