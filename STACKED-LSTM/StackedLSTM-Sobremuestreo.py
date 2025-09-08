import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from imblearn.over_sampling import SMOTE

# 1. Cargar el dataset preprocesado
df = pd.read_csv("dataset_procesado2.csv")

# 2. Separar características (X) y etiquetas (y)
X = df.drop("Label", axis=1).values
y = df["Label"].values

# 3. Aplicar sobremuestreo SMOTE antes de generar secuencias
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# 4. Definir la longitud de la secuencia (ventana temporal)
time_steps = 10

# 5. Generar secuencias con los datos sobremuestreados
generator = TimeseriesGenerator(X_res, y_res, length=time_steps, batch_size=64)

# 6. Definir el modelo Stacked LSTM
model = Sequential()
# Primera capa LSTM
model.add(LSTM(64, activation="tanh", return_sequences=True, input_shape=(time_steps, X.shape[1])))
model.add(Dropout(0.2))
# Segunda capa LSTM
model.add(LSTM(32, activation="tanh", return_sequences=False))
model.add(Dropout(0.2))
# Capa de salida para clasificación binaria
model.add(Dense(1, activation="sigmoid"))

# 7. Compilar el modelo
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Mostrar resumen del modelo
model.summary()

# 8. Entrenar el modelo
epochs = 20
history = model.fit(generator, epochs=epochs)

# 9. Guardar el modelo entrenado
model.save("modelo_stacked_lstm_Sobremuestreo.h5")
print("Modelo guardado en 'modelo_stacked_lstm_Sobremuestreo.h5'.")
