# Importar las librerías necesarias
import gdown
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Descargar el archivo desde Google Drive
file_id = '15aPwYiUWm3Zz2XFraAkTf3-IzArnLnFQ'
destination = 'fitness_tracker_dataset.csv'  # Nombre del archivo
url = f'https://drive.google.com/uc?id={file_id}'
gdown.download(url, destination, quiet=False)

# Cargar el dataset en un DataFrame
data = pd.read_csv(destination)

# Verificar las primeras filas del dataset
print(data.head())

# Preprocesamiento de datos
# Selección de las columnas relevantes
features = ['steps', 'distance_km', 'active_minutes', 'sleep_hours', 'heart_rate_avg']
target = 'calories_burned'

X = data[features]
y = data[target]

# Dividir el dataset en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Construir el modelo de red neuronal
model = Sequential()
model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))  # Salida de un solo valor (calorías quemadas)

# Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo
history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=16, validation_split=0.2, verbose=1)

# Evaluar el modelo en el conjunto de prueba
test_loss = model.evaluate(X_test_scaled, y_test)
print(f'Test Loss (MSE): {test_loss}')

# Hacer predicciones
y_pred = model.predict(X_test_scaled)

# Calcular el MSE y MAE para las predicciones
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')

# Graficar la pérdida durante el entrenamiento
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.show()

# Fin del código
