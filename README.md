# README

## Descripción del Proyecto

Este proyecto demuestra cómo entrenar un modelo de red neuronal simple utilizando TensorFlow para convertir grados Celsius a Fahrenheit. El modelo se entrena con un conjunto de datos de ejemplo y luego se utiliza para hacer predicciones.

## Requisitos

- Python 3.6+
- TensorFlow 2.x
- NumPy
- Matplotlib

## Instalación

1. Clona este repositorio en tu máquina local.
2. Instala las dependencias necesarias usando pip:

    ```bash
    pip install tensorflow numpy matplotlib
    ```

## Uso

1. Configuración del entorno:

    El código desactiva algunas opciones y reduce la verbosidad de los mensajes de TensorFlow configurando las variables de entorno:

    ```python
    import os
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    ```

2. Importar las bibliotecas necesarias:

    ```python
    import tensorflow as tf
    import numpy as np
    import matplotlib.pyplot as plt
    ```

3. Definir el conjunto de datos:

    ```python
    celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
    fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

    celsius = np.expand_dims(celsius, axis=-1)
    fahrenheit = np.expand_dims(fahrenheit, axis=-1)
    ```

4. Configurar el modelo:

    ```python
    capaoculta1 = tf.keras.layers.Dense(units=3)
    capaoculta2 = tf.keras.layers.Dense(units=3)
    capasalida = tf.keras.layers.Dense(units=1)
    modelo = tf.keras.Sequential([capaoculta1, capaoculta2, capasalida])

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(0.1),
        loss='mean_squared_error'
    )
    ```

5. Entrenar el modelo:

    ```python
    print("Comenzando entrenamiento...")
    historial = modelo.fit(celsius, fahrenheit, epochs=1000, verbose=False)
    print("Modelo entrenado!")
    ```

6. Visualizar la pérdida:

    ```python
    plt.xlabel("# Epoca")
    plt.ylabel("Magnitud de pérdida")
    plt.plot(historial.history["loss"])
    plt.show()
    ```

7. Realizar una predicción:

    ```python
    print("Hagamos una predicción!")
    input_data = np.array([[100.0]])
    resultado = modelo.predict(input_data)
    print("El resultado es " + str(resultado) + " fahrenheit!")
    ```

8. Inspeccionar los pesos del modelo:

    ```python
    print("Variables internas del modelo")
    print(capaoculta1.get_weights())
    print(capaoculta2.get_weights())
    print(capasalida.get_weights())
    ```

## Ejecución

Para ejecutar el código, simplemente corre el script en tu entorno Python. El script entrenará el modelo, visualizará la pérdida durante el entrenamiento y realizará una predicción de prueba, además de imprimir los pesos internos del modelo.

```bash
python script_name.py
```

## Contribución

Si deseas contribuir a este proyecto, por favor, abre un issue o envía un pull request con tus sugerencias y mejoras.

## Licencia

Este proyecto está licenciado bajo la Licencia MIT. Consulta el archivo LICENSE para obtener más detalles.
