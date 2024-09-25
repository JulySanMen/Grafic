import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage.color import rgb2gray
from skimage.io import imread

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-i', '--image', help='Imagen de entrada', required=True)
arguments = vars(argument_parser.parse_args())

image = imread(arguments['image'])
gray = rgb2gray(image)
width, height = gray.shape

# Escalado
scaling_matrix = np.array([[0.75, 0, 0], [0, 1.25, 0], [0, 0, 1]])
scaled_image = ndimage.affine_transform(gray, scaling_matrix)

# Rotacion
theta = np.pi / 6
rotation_matrix = (np.array([[1, 0, width / 2], [0, 1, height / 2], [0, 0, 1]]) @
                   np.array([[np.cos(theta), np.sin(theta), 0], [np.sin(theta), -np.cos(theta), 0], [0, 0, 1]]) @
                   np.array([[1, 0, -width / 2], [0, 1, -height / 2], [0, 0, 1]]))
rotated_image = ndimage.affine_transform(gray, rotation_matrix)

# Cizallado
shearing_matrix = np.array([[1, 0.5, 0], [0.5, 1, 0], [0, 0, 1]])
sheared_image = ndimage.affine_transform(gray, shearing_matrix)

# Traslación
tx, ty = 50, 30
translation_matrix = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
translated_image = ndimage.affine_transform(gray, translation_matrix)

titles = ['Original', 'Escalada', 'Rotada', 'Cizallada', 'Trasladada']
images = [gray, scaled_image, rotated_image, sheared_image, translated_image]


plt.figure(figsize=(20, 10))
for i, title, image in zip(range(1, 6), titles, images):
    plt.subplot(230 + i)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.title(title, size=20)

plt.show()
