import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# dataset
celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

# Expandir las dimensiones de los datos de entrada
celsius = np.expand_dims(celsius, axis=-1)
fahrenheit = np.expand_dims(fahrenheit, axis=-1)


#configuración

capaoculta1 = tf.keras.layers.Dense(units=3)
capaoculta2 = tf.keras.layers.Dense(units=3)
capasalida = tf.keras.layers.Dense(units=1)
modelo = tf.keras.Sequential([capaoculta1, capaoculta2, capasalida])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

#entrenamiento

print("Comenzando entrenamiento...")
historial = modelo.fit(celsius, fahrenheit, epochs=1000, verbose=False)
print("Modelo entrenado!")


plt.xlabel("# Epoca")
plt.ylabel("Magnitud de pérdida")
plt.plot(historial.history["loss"])
plt.show()

#probar el modelo
print("Hagamos una predicción!")
input_data = np.array([[100.0]])
resultado = modelo.predict(input_data)
print("El resultado es " + str(resultado) + " fahrenheit!")
# El resultado es [[211.7477]] fahrenheit! muy cerca

# comprobar pesos
print("Variables internas del modelo")
#print(capa.get_weights())
print(capaoculta1.get_weights())
print(capaoculta2.get_weights())
print(capasalida.get_weights())