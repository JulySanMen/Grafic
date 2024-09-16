import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

import matplotlib
matplotlib.use('Agg')  # Usa el backend 'Agg' para entornos sin interfaz gráfica
import matplotlib.pyplot as plt

# Cargar y preparar los datos
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train, axis=-1).astype('float32') / 255.0
x_test = np.expand_dims(x_test, axis=-1).astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Definir el filtro de Sobel
def sobel_filter():
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    return sobel_x, sobel_y

def apply_sobel_filter(image):
    sobel_x, sobel_y = sobel_filter()
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    
    # Aplicar el filtro Sobel en x y y
    sobel_x_filter = tf.image.sobel_edges(image)[..., 0]  # Extraer el canal X
    sobel_y_filter = tf.image.sobel_edges(image)[..., 1]  # Extraer el canal Y
    
    # Combinar las imágenes en una sola (puedes usar la magnitud del gradiente para esto)
    filtered_image = tf.sqrt(tf.square(sobel_x_filter) + tf.square(sobel_y_filter))
    
    return filtered_image.numpy()


def show_image_and_filter(image):
    filtered_image = apply_sobel_filter(image[None, ...])
    plt.figure(figsize=(10, 5))
    
    # Mostrar imagen original
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image.squeeze(), cmap='gray')
    plt.axis('off')
    
    # Mostrar imagen después de aplicar el filtro
    plt.subplot(1, 2, 2)
    plt.title("After Sobel Filter")
    plt.imshow(filtered_image.squeeze(), cmap='gray')
    plt.axis('off')
    
    # Guardar las imágenes en archivos
    plt.savefig('original_vs_sobel.png')
    plt.close()


# Crear un modelo de red neuronal convolucional simple
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
model.fit(x_train, y_train, epochs=5, validation_split=0.1)

# Evaluar el modelo
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc}')

# Mostrar una imagen y su filtro aplicado
show_image_and_filter(x_test[0])
