import pandas as pd
import re
import os

# Expresiones regulares para parsear los logs
access_regex = re.compile(
    r'(\S+) - - \[(.+?)\] "(\w+) (.+?) (HTTP/.+?)" (\d{3}) (\d+|-) "(.+?)" "(.+?)"'
)

ssl_request_regex = re.compile(
    r'\[(.+?)\] (\S+) (\S+) (\S+) "(\w+) (.+?) (HTTP/.+?)" (\d+)'
)


def parse_logs(log_files):
    """
    Lee archivos de log y extrae la información en formato estructurado.
    """
    parsed_data = []

    for log_file, log_type in log_files:
        if not os.path.exists(log_file):
            print(f"Advertencia: El archivo {log_file} no se encontró. Omitiendo.")
            continue
            
        print(f"Procesando {log_file} como tipo '{log_type}'...")
        with open(log_file, 'r') as f:
            for line in f:
                data = {
                    "ip": None, "timestamp_str": None, "method": None, "url": None,
                    "http_version": None, "status_code": None, "response_size": None,
                    "referer": None, "user_agent": None, "tls_version": None,
                    "cipher_suite": None, "log_source": log_type
                }
                
                if log_type == 'access':
                    match = access_regex.match(line)
                    if match:
                        g = match.groups()
                        data.update({
                            "ip": g[0], "timestamp_str": g[1], "method": g[2],
                            "url": g[3], "http_version": g[4], "status_code": g[5],
                            "response_size": g[6], "referer": g[7], "user_agent": g[8]
                        })
                        parsed_data.append(data)
                        
                elif log_type == 'ssl_request':
                    match = ssl_request_regex.match(line)
                    if match:
                        g = match.groups()
                        data.update({
                            "timestamp_str": g[0], "ip": g[1], "tls_version": g[2],
                            "cipher_suite": g[3], "method": g[4], "url": g[5],
                            "http_version": g[6], "response_size": g[7],
                            "status_code": "200"
                        })
                        parsed_data.append(data)
                        
    return parsed_data


archivos_a_procesar = [
    ('Logs/access_log', 'access'),
   # ('Logs/access_log-20250921', 'access'),
   # ('Logs/access_log-20250928', 'access'),
   # ('Logs/access_log-20251005', 'access'),
   # ('Logs/access_log-20251012', 'access'),
   # ('Logs/ssl_request_log', 'ssl_request'),
   # ('Logs/ssl_request_log-20250921', 'ssl_request'),
   # ('Logs/ssl_request_log-20250928', 'ssl_request'),
   # ('Logs/ssl_request_log-20251005', 'ssl_request'),
   # ('Logs/ssl_request_log-20251012', 'ssl_request'),
]

datos_extraidos = parse_logs(archivos_a_procesar)

if datos_extraidos:
    print(f"\nSe extrajeron {len(datos_extraidos)} registros.")
    
    print("Creando DataFrame y transformando tipos de datos...")
    
    df = pd.DataFrame(datos_extraidos)

    # Conversión de timestamp a formato datetime
    df['timestamp'] = pd.to_datetime(df['timestamp_str'], 
                                     format='%d/%b/%Y:%H:%M:%S %z', 
                                     errors='coerce')
    
    # Conversión de campos numéricos
    df['status_code'] = pd.to_numeric(df['status_code'], errors='coerce').fillna(0).astype(int)
    df['response_size'] = df['response_size'].replace('-', '0')
    df['response_size'] = pd.to_numeric(df['response_size'], errors='coerce').fillna(0).astype(int)
    
    # Limpieza de comillas
    df['user_agent'] = df['user_agent'].str.strip('"')
    df['referer'] = df['referer'].str.strip('"')
    
    df = df.drop(columns=['timestamp_str'])

    print("\nDataFrame procesado correctamente")
    
    print("\nInformación del DataFrame:")
    df.info()
    
    print("\nPrimeras 5 filas:")
    print(df.head())
    
    df.to_csv('logs_unificados.csv', index=False)
    print("\nDatos guardados en 'logs_unificados.csv'")

else:
    print("No se extrajeron datos. Revisa los archivos de log y las expresiones regulares.")