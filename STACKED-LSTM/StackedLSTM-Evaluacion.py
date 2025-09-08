import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.metrics import classification_report, confusion_matrix

# Cargar el modelo guardado
model = load_model("modelo_stacked_lstm_SM100.h5")
print("Modelo cargado:")
model.summary()

# Cargar el dataset preprocesado
df_test = pd.read_csv("dataset_procesado.csv")

# Separar características y etiqueta
X_test = df_test.drop("Label", axis=1).values
y_test = df_test["Label"].values

# Definir la longitud de la secuencia (la misma usada en entrenamiento)
time_steps = 10

# Crear el generador para el dataset de prueba
test_generator = TimeseriesGenerator(X_test, y_test, length=time_steps, batch_size=64)

# Evaluar el modelo sobre el conjunto de prueba
loss, accuracy = model.evaluate(test_generator)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

# Obtener predicciones (las salidas son probabilidades)
y_pred_prob = model.predict(test_generator)
# Convertir probabilidades a etiquetas (umbral 0.5) 
y_pred = (y_pred_prob > 0.5).astype(int)

# Extraer las etiquetas verdaderas del generador
y_true = []
for i in range(len(test_generator)):
    _, y_batch = test_generator[i]
    y_true.extend(y_batch)
y_true = np.array(y_true)

# Imprimir métricas de evaluación
print("Classification Report:")
print(classification_report(y_true, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# Generar un Excel con las predicciones
# Debido a la creación de secuencias, los primeros 'time_steps' registros no tienen predicción.
# Se asignan las predicciones a los registros a partir del índice 'time_steps'.
pred_indices = np.arange(time_steps, len(X_test))
df_pred = df_test.iloc[pred_indices].copy()

# Ajustar el tamaño de df_pred para que coincida con y_pred
# Esto es necesario porque el generador crea un número de muestras menor que el total de registros.
df_pred = df_pred.iloc[:len(y_pred)]
df_pred['Predicted_Label'] = y_pred.flatten()

# Guardar en un Excel
df_pred.to_excel("100.xlsx", index=False)
print("Predicciones guardadas en '100.xlsx'.")
