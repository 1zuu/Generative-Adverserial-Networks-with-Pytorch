import os, torch
import numpy as np
import torch.nn as nn
from IPython import display
import matplotlib.pyplot as plt
from variables import *

def set_seed():
    np.random.seed(seed)
    torch.manual_seed(seed)

def plot_multiple_images(images, title, n_cols=None):
    display.clear_output(wait=False)  
    n_cols = n_cols or len(images)
    n_rows = (len(images) - 1) // n_cols + 1

    if images.shape[-1] == 1:
        images = np.squeeze(images, axis=-1)

    fig = plt.figure(figsize=(n_cols, n_rows))
    
    for index, image in enumerate(images):
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(image.numpy().reshape(image_dim,image_dim), cmap="binary")
        plt.axis("off")
    title = "{}_{}".format(title[0], title[1])
    plt.savefig('results/{}.png'.format(title))   

def weights_initialization(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)

def gradient_penalty(gradient):
    gradient = gradient.view(len(gradient), -1)
    gradient_norm = gradient.norm(2, dim=1)
    penalty = torch.mean((1 - gradient_norm).pow(2))
    return penalty