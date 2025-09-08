import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# Cargar el dataset preprocesado
df = pd.read_csv("dataset_procesadoWednesday-21-02-2018.csv")

# Separar las características y la etiqueta
# Se asume que 'Label' es la columna objetivo y las demás son características
X = df.drop("Label", axis=1).values
y = df["Label"].values

# Definir la longitud de la secuencia (ventana temporal)
time_steps = 10

# Crear secuencias para el LSTM con TODOS los datos para el entrenamiento
generator = TimeseriesGenerator(X, y, length=time_steps, batch_size=64)

# Definir el modelo Stacked LSTM
model = Sequential()
# Primera capa LSTM con return_sequences=True para apilar otra LSTM
model.add(LSTM(64, activation="tanh", return_sequences=True, input_shape=(time_steps, X.shape[1])))
model.add(Dropout(0.2))
# Segunda capa LSTM
model.add(LSTM(32, activation="tanh", return_sequences=False))
model.add(Dropout(0.2))
# Capa densa final para clasificación binaria
model.add(Dense(1, activation="sigmoid"))

# Compilar el modelo
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Mostrar resumen del modelo
model.summary()

# Entrenar el modelo
epochs = 30
history = model.fit(generator, epochs=epochs)

# Guardar el modelo entrenado
model.save("PRUEBA_MODELO.h5")
print("Modelo guardado en 'PRUEBA_MODELO.h5'.")
