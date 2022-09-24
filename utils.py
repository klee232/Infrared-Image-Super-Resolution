import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


def load_image(path):
    return np.array(Image.open(path))


def plot_sample(lr, sr, hr):
    plt.figure(figsize=(20, 10))

    images = [lr, sr, hr]
    titles = ['LR', f'SR (x{sr.shape[0] // lr.shape[0]})', 'HR']


    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, 3, i+1)
        plt.imshow(img,cmap='gray')
        plt.title(title)
        plt.xticks([])
        plt.yticks([])
