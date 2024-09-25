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
