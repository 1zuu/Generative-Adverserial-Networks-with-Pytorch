import os
import numpy as np
import tensorflow as tf
from IPython import display
import matplotlib.pyplot as plt
from variables import *

def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)

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