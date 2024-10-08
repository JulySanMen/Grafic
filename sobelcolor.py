import cv2
import numpy as np
import matplotlib.pyplot as plt
# Cargar imagen en color
imagen_color = cv2.imread('ima.jpeg')

# Separar los canales de color
b, g, r = cv2.split(imagen_color)

# Aplicar Sobel en cada canal
sobel_b = cv2.Sobel(b, cv2.CV_64F, 1, 1, ksize=3)
sobel_g = cv2.Sobel(g, cv2.CV_64F, 1, 1, ksize=3)
sobel_r = cv2.Sobel(r, cv2.CV_64F, 1, 1, ksize=3)

# Combinar los canales nuevamente
sobel_combined_color = cv2.merge([sobel_b, sobel_g, sobel_r])

plt.imshow(cv2.cvtColor(sobel_combined_color.astype(np.uint8), cv2.COLOR_BGR2RGB))
plt.title('Sobel en Imagen a Color')
plt.show()

#resaltar bordes en imagenes  borrosas


imagen = cv2.imread('des.avif', cv2.IMREAD_GRAYSCALE)
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
