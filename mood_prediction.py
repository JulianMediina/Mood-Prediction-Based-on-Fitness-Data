# Importar las bibliotecas necesarias
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Cargar el dataset
file_path = 'fitness_tracker_dataset.csv'
data = pd.read_csv(file_path)

# Mostrar las primeras filas del dataset para exploración
print(data.head())

# Información general del dataset
print(data.info())

# Revisar si hay valores nulos
print(data.isnull().sum())

# Eliminar filas o columnas con muchos valores nulos, si es necesario
# data = data.dropna()

# Análisis exploratorio: Visualizar la correlación entre las variables
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Matriz de correlación')
plt.show()

# Selección de características (X) y variable objetivo (y)
X = data[['steps', 'distance_km', 'active_minutes', 'sleep_hours', 'heart_rate_avg']]
y = data['calories_burned']

# Dividir el dataset en conjunto de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar las características para mejorar el rendimiento del modelo
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Construir el modelo de red neuronal
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),  # Capa de entrada
    Dense(32, activation='relu'),  # Capa oculta
    Dense(1)  # Capa de salida (predicción de un valor continuo - calorías quemadas)
])

# Compilar el modelo (usamos mean_squared_error porque es un problema de regresión)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

# Entrenar el modelo
history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=16, validation_split=0.2, verbose=1)

# Evaluar el modelo en el conjunto de prueba
test_loss, test_mae = model.evaluate(X_test_scaled, y_test)
print(f'Pérdida en el conjunto de prueba (MSE): {test_loss}')
print(f'Error absoluto medio en el conjunto de prueba (MAE): {test_mae}')

# Visualizar el historial de entrenamiento (pérdida y MAE)
plt.figure(figsize=(12, 5))

# Gráfico de la pérdida (MSE)
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Pérdida de Entrenamiento (MSE)')
plt.plot(history.history['val_loss'], label='Pérdida de Validación (MSE)')
plt.title('Pérdida (MSE) a lo largo de las épocas')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()

# Gráfico del error absoluto medio (MAE)
plt.subplot(1, 2, 2)
plt.plot(history.history['mean_absolute_error'], label='MAE de Entrenamiento')
plt.plot(history.history['val_mean_absolute_error'], label='MAE de Validación')
plt.title('MAE a lo largo de las épocas')
plt.xlabel('Épocas')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.show()

# Hacer predicciones con el conjunto de prueba
predictions = model.predict(X_test_scaled)

# Comparar algunas predicciones con los valores reales
comparison_df = pd.DataFrame({'Real': y_test, 'Predicción': predictions.flatten()})
print(comparison_df.head())
