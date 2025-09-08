import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# Cargar el dataset
ruta_archivo = "dataset_minimoWednesday-21-02-2018.csv"

# Definir columnas de interés
columnas_numericas = [
    "Dst Port", "Protocol", "Flow Duration", "Tot Fwd Pkts",
    "Tot Bwd Pkts", "TotLen Fwd Pkts", "TotLen Bwd Pkts",
    "Flow Byts/s", "Flow Pkts/s", "Flow IAT Mean"
]

df = pd.read_csv(ruta_archivo, usecols=columnas_numericas + ["Label"], low_memory=False)

# Convertir columnas numéricas a float (evitando errores)
for col in columnas_numericas:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Reemplazar valores "Infinity" por NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Verificar y manejar valores nulos
print("Valores nulos antes de tratamiento:\n", df.isnull().sum())

# Reemplazar nulos con la mediana de cada columna
df.fillna(df.median(numeric_only=True), inplace=True)

# Normalización con MinMaxScaler
scaler = MinMaxScaler()
df[columnas_numericas] = scaler.fit_transform(df[columnas_numericas])

# Convertir la columna 'Label' a 0 y 1
df["Label"] = df["Label"].apply(lambda x: 1 if x != "Benign" else 0)

# Crear dataset para LSTM con ventanas de tiempo
X = df[columnas_numericas].values
y = df["Label"].values

time_steps = 10  # Longitud de la secuencia para LSTM
generator = TimeseriesGenerator(X, y, length=time_steps, batch_size=64)

print(f"Total de muestras generadas: {len(generator)}")

# Guardar el dataset preprocesado
df.to_csv("dataset_procesadoWednesday-21-02-2018.csv", index=False)
print("Dataset procesado guardado como 'dataset_procesadoWednesday-21-02-2018.csv'.")
