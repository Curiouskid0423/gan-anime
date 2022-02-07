"""
Inference code to generate faces for testing results.
"""

import torch
import torchvision
import os
from torch.autograd import Variable
from model.dcgan import Generator
from model.trainer import Trainer
import matplotlib.pyplot as plt

def inference(workspace_dir='.', inference_ckpt_path=None, cloud_log_path=None, n_output=1000):
    if inference_ckpt_path is None:
        print("Inference checkpoint not found. Please provide the model for inference use.")
        return
    trainer = Trainer()
    z_dim = trainer.z_dim
    
    # G = Generator(z_dim)
    G = Generator(z_dim)
    ckpt = torch.load(inference_ckpt_path)
    G.load_state_dict(ckpt['G'])
    G.eval()
    print(G.cuda())

    # Generate 1000 images and make a grid to save them.
    z_sample = Variable(torch.randn(n_output, z_dim)).cuda()
    imgs_sample = (G(z_sample).data + 1) / 2.0
    log_dir =  cloud_log_path if cloud_log_path else os.path.join(workspace_dir, 'logs')
    filename = os.path.join(log_dir, 'result.jpg')
    torchvision.utils.save_image(imgs_sample, filename, nrow=10)

    # Show 32 of the images.
    grid_img = torchvision.utils.make_grid(imgs_sample[:32].cpu(), nrow=10)
    plt.figure(figsize=(10,10))
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.show()

    # Save the generated images.
    os.makedirs('output', exist_ok=True)
    for i in range(1000):
        torchvision.utils.save_image(imgs_sample[i], f'output/{i+1}.jpg')
    