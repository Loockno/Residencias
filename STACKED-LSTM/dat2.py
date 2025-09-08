import pandas as pd

# Definir la ruta del archivo CSV
ruta_archivo = r"Wednesday-21-02-2018_TrafficForML_CICFlowMeter-copia.csv"

# Leer el archivo CSV con pandas
data = pd.read_csv(ruta_archivo, low_memory=False)

# Definir las columnas indispensables
columns_min = [
    "Dst Port", "Protocol","Timestamp", "Flow Duration", "Tot Fwd Pkts", 
    "Tot Bwd Pkts", "TotLen Fwd Pkts", "TotLen Bwd Pkts",
    "Flow Byts/s", "Flow Pkts/s", "Flow IAT Mean", "Label"
]

# Filtrar solo las columnas necesarias
data_min = data[columns_min]

# Guardar el nuevo conjunto de datos en un archivo CSV
data_min.to_csv("dataset_minimoWednesday-21-02-2018.csv", index=False)

print("Archivo 'dataset_minimoWednesday-21-02-2018.csv' creado con las columnas indispensables.")
