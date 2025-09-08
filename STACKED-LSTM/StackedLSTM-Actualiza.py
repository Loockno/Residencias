import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# (Opcional) Verifica si se está usando eager execution:
print("Eager execution:", tf.executing_eagerly())

# Si por alguna razón no se está usando, puedes forzarlo.
tf.config.run_functions_eagerly(True)

# Cargar el modelo guardado y recompilarlo (en caso de que se pierda la compilación)
modelo = load_model("modelo_stacked_lstm_SM100.h5", compile=False)
modelo.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
print("Modelo cargado:")
modelo.summary()

# Cargar el nuevo dataset preprocesado
nuevo_df = pd.read_csv("dataset_procesadoWednesday-21-02-2018.csv")

# Separar las características y la etiqueta
X_new = nuevo_df.drop("Label", axis=1).values
y_new = nuevo_df["Label"].values

# Definir la longitud de la secuencia (la misma usada en entrenamiento)
time_steps = 10

# Crear secuencias para el LSTM con los nuevos datos
nuevo_generator = TimeseriesGenerator(X_new, y_new, length=time_steps, batch_size=64)

# Continuar el entrenamiento del modelo con los nuevos datos
additional_epochs = 1
history2 = modelo.fit(nuevo_generator, epochs=additional_epochs)

# Guardar el modelo actualizado
modelo.save("modelo_stacked_lstm_SM101.h5")
print("Modelo actualizado guardado en 'modelo_stacked_lstm_SM101.h5'.")
