'''
Trainer file for DC_GAN with spectral norm.
Define all the hyperparameters.
'''
import os
import torch
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model.dcgan import Discriminator, Generator
from model.spectral_norm import Discriminator as SN_Discriminator
from data.data_loader import *
import matplotlib.pyplot as plt

class Trainer:

    def __init__(self, workspace_dir='.', n_epoch=50, ckpt_save_path=None, log_save_path=None, z_dim=100, batch_size=64, lr=1e-4):
        
        # Google Drive path for saving checkpoints
        self.checkpoint_save_paths = ckpt_save_path 

        """ Hyperparameter definitions """
        # Medium Baseline: WGAN, 50 epoch, n_critic=5, clip_value=0.01
        self.batch_size = batch_size
        self.z_dim = z_dim  # Latent space dimension
        self.z_sample = Variable(torch.randn(100, z_dim)).cuda()
        self.lr = lr
        self.n_epoch = n_epoch
        self.n_critic = 4
        self.clip_value = 0.01
        log_dir = log_save_path if log_save_path else os.path.join(workspace_dir, 'logs')
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        """ END of hyperparameter definitions"""

        # Models. Set to "train" mode
        generator = Generator(in_dim=z_dim).cuda()
        # discriminator = Discriminator(3).cuda()
        discriminator = SN_Discriminator(3).cuda()
        self.G = generator
        self.D = discriminator
        # Loss and Optimizer
        # self.opt_D = torch.optim.RMSprop(discriminator.parameters(), lr=lr)
        # self.opt_G = torch.optim.RMSprop(generator.parameters(), lr=lr)
        lr_sngan = 0.002 # Temporarily use 2e-4 for lr for SN-GAN
        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=lr_sngan, betas=(0, 0.9)) 
        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=lr_sngan, betas=(0, 0.9)) 

        # DataLoader
        dataset = get_dataset(os.path.join(workspace_dir, 'faces'))
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    def load_checkpoint(self, states_path):
        """ 
        Load states when the model is identical with that of the checkpoint
        """
        if states_path:
            states = torch.load(states_path)
            self.G.load_state_dict(states['G'])
            self.D.load_state_dict(states['D'])
            self.opt_G.load_state_dict(states['opt_G'])
            self.opt_D.load_state_dict(states['opt_D'])
            print('Model loaded from %s' % states_path)

    def load_checkpoint_partial(self, states_path):
        """ 
        Load only part of the states (manually chosen) when 
        the loaded model is not the same.
        """
        if states_path:
            states = torch.load(states_path)
            self.D.load_state_dict(states['D'])
            self.opt_D.load_state_dict(states['opt_D'])
            g_states = {}
            for name, param in states["G"].items():
                if 'add_on_one_layer' not in name:
                    g_states[name] = param
            self.G.state_dict().update(g_states)
            self.G.load_state_dict(self.G.state_dict())
            # ignore opt_G states for now
            print('Model loaded from %s' % states_path)

            
    def load_checkpoint_separate(self, G_states_path, D_states_path):
        """ 
        Load only part of the states (manually chosen) when 
        the loaded model is not the same.
        """
        if G_states_path or D_states_path:
            g_states, d_states = torch.load(G_states_path), torch.load(D_states_path)
            self.G.load_state_dict(g_states['G'])
            self.opt_G.load_state_dict(g_states['opt_G'])
            self.D.load_state_dict(d_states['D'])
            self.opt_D.load_state_dict(d_states['opt_D'])
            print(f'Model loaded from {G_states_path} and {D_states_path}')

    def training_loop(self, steps=0, prev_ckpt_path=None, g_ckpt_path=None, fine_tune=False):
        self.G.train() 
        self.D.train()
        d_loss_history, g_loss_history = [], []

        """
        Checkpoint loading options
        1.  Load in checkpoint when the models are identical. 
        2.  Load in checkpoints "separately" for D and G, when the models are identical.
        3.  Load in checkpoint for "fine-tuning" when the new model is "modified".
        """
        if prev_ckpt_path:
            if fine_tune:
                self.load_checkpoint_partial(prev_ckpt_path)
                print("Partial checkpoint loaded. Fine-tuning.")
            elif not g_ckpt_path: # Default use.
                self.load_checkpoint(prev_ckpt_path)
                print("Loaded the entire checkpoint for both G and D.")
            elif g_ckpt_path:
                self.load_checkpoint_separate(g_ckpt_path, prev_ckpt_path)
                print("Loaded checkpoints for G and D separately.")
            
        for e, epoch in enumerate(range(self.n_epoch)):
            for i, data in enumerate(self.dataloader):
                imgs = data.cuda()
                bs = imgs.size(0)

                # ============================================
                #  Train D
                # ============================================
                z = Variable(torch.randn(bs, self.z_dim)).cuda()
                r_imgs = Variable(imgs).cuda()
                f_imgs = self.G(z)

                r_label = torch.ones((bs)).cuda()
                f_label = torch.zeros((bs)).cuda()

                # Model forwarding
                r_logit = self.D(r_imgs.detach())
                f_logit = self.D(f_imgs.detach())

                # WGAN Loss
                loss_D = -torch.mean(self.D(r_imgs)) + torch.mean(self.D(f_imgs))

                # Model backwarding
                self.D.zero_grad()
                loss_D.backward()
                self.opt_D.step()

                """ Clip weights of discriminator. """
                for p in self.D.parameters():
                    p.data.clamp_(-self.clip_value, self.clip_value)

                # ============================================
                #  Train G
                # ============================================
                if steps % self.n_critic == 0:
                    # Generate some fake images.
                    z = Variable(torch.randn(bs, self.z_dim)).cuda()
                    f_imgs = self.G(z)
                    # Model forwarding
                    f_logit = self.D(f_imgs)
                    # Compute the loss for the generator.
                    loss_G = -torch.mean(self.D(f_imgs))

                    d_loss_history.append(loss_D)
                    g_loss_history.append(loss_G)
                    
                    self.G.zero_grad()
                    loss_G.backward()
                    self.opt_G.step()

                steps += 1

                if steps % 100 == 0:
                    print(f'Loss_D :: {round(loss_D.item(), 4)} | Loss_G :: {round(loss_G.item(), 4)} | Epoch :: {e+1:03d} | Step :: {steps:05d}')

            self.G.eval()
            f_imgs_sample = (self.G(self.z_sample).data + 1) / 2.0
            filename = os.path.join(self.log_dir, f'Epoch_{epoch+1:03d}.jpg')
            torchvision.utils.save_image(f_imgs_sample, filename, nrow=10)
            print(f' | Save some samples to {filename}.')

            self.G.train()

            if (e+1) % 10 == 0 or e == 0:
                # Save the checkpoints.
                states = {
                    "G": self.G.state_dict(), "D": self.D.state_dict(), 
                    "opt_G": self.opt_G.state_dict(), "opt_D": self.opt_D.state_dict()
                }
                torch.save(states, os.path.join(self.checkpoint_save_paths, 'latest_ckpt.pth'))
                torch.save(states, os.path.join(self.checkpoint_save_paths, f'ckpt_{e+1}.pth'))

        return d_loss_history, g_loss_history