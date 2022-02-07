'''
Main driver file. 
2021.05.06
'''

import random
import numpy as np
import matplotlib.pyplot as plt
import torch 
import torchvision
from model.trainer import Trainer

class Driver:

    def __init__(self, ckpt_cloud_save_path, log_cloud_save_path=None):
        self.seed = 2021
        if not torch.cuda.is_available():
            print("Cuda device not available. Please connect to GPU before training.")
            return
        self.trainer = Trainer(ckpt_save_path=ckpt_cloud_save_path, log_save_path=log_cloud_save_path)
        self.show_image_data()

    def same_seeds(self, seed):
        # Fix random module for Python, NumPy, and PyTorch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    
    def show_image_data(self):
        dataset = self.trainer.dataset
        images = [(dataset[i]+1)/2 for i in range(16)]
        grid_img = torchvision.utils.make_grid(images, nrow=4)
        plt.figure(figsize=(10,10))
        plt.imshow(grid_img.permute(1, 2, 0))
        plt.show()

    def train(self, prev_ckpt=None, g_ckpt_path=None, log_path=None):
        """ Pass in the Google Drive path of previous checkpoint if available. """
        self.same_seeds(self.seed) 
        # Actual training starts!
        d_loss_history, g_loss_history = self.trainer.training_loop(
            prev_ckpt_path=prev_ckpt, g_ckpt_path=g_ckpt_path, fine_tune=False
        )
        # return d_loss_history, g_loss_history
