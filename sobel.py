import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar imagen y convertir a escala de grises
imagen = cv2.imread('ima.jpeg', cv2.IMREAD_GRAYSCALE)

# Aplicar el filtro de Sobel en el eje X y Y
sobelx = cv2.Sobel(imagen, cv2.CV_64F, 1, 0, ksize=3)  # Sobel en eje X
sobely = cv2.Sobel(imagen, cv2.CV_64F, 0, 1, ksize=3)  # Sobel en eje Y

# Magnitud del gradiente combinando ambos ejes
sobel_combined = np.sqrt(sobelx**2 + sobely**2)

# Mostrar las imágenes originales y procesadas
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1), plt.imshow(imagen, cmap='gray'), plt.title('Imagen Original')
plt.subplot(1, 3, 2), plt.imshow(sobelx, cmap='gray'), plt.title('Sobel X')
plt.subplot(1, 3, 3), plt.imshow(sobely, cmap='gray'), plt.title('Sobel Y')
plt.show()

plt.imshow(sobel_combined, cmap='gray'), plt.title('Sobel Combinado')
plt.show()
