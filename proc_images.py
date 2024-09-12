from PIL import Image
import numpy as np

imagen = Image.open('/home/julyfranco/Descargas/gato.jpeg')
#imagen = imagen.convert('L')
matriz = np.array(imagen)
print(matriz)
