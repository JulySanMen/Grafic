import cv2
import numpy as np
import matplotlib.pyplot as plt
#resaltar bordes en imagenes  borrosas


imagen = cv2.imread('ima.jpeg', cv2.IMREAD_GRAYSCALE)
imagen_blur = cv2.GaussianBlur(imagen, (5, 5), 0)

# Aplicar Sobel
sobelx_blur = cv2.Sobel(imagen_blur, cv2.CV_64F, 1, 0, ksize=3)
sobely_blur = cv2.Sobel(imagen_blur, cv2.CV_64F, 0, 1, ksize=3)

sobel_combined_blur = np.sqrt(sobelx_blur**2 + sobely_blur**2)

# Mostrar la imagen borrosa y el resultado de Sobel
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1), plt.imshow(imagen_blur, cmap='gray'), plt.title('Imagen Borrosa')
plt.subplot(1, 2, 2), plt.imshow(sobel_combined_blur, cmap='gray'), plt.title('Sobel en Imagen Borrosa')
plt.show()