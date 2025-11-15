# 🤖 Guía de Predicción de Demanda Eléctrica - V2

## 📋 Descripción General

Este módulo implementa **predicción de demanda eléctrica** para las próximas 24 horas utilizando múltiples enfoques de Machine Learning con **k-fold cross-validation** para garantizar robustez.

## 🧠 Modelos Disponibles

### 1. LSTM (Long Short-Term Memory) - **Modelo Principal**

**Arquitectura:**
```
Input (168 horas × 1 feature)
    ↓
LSTM Layer 1: 64 units + Dropout (0.2)
    ↓
LSTM Layer 2: 32 units + Dropout (0.2)
    ↓
Dense Layer: 32 neurons (ReLU)
    ↓
Output: 1 neuron (predicción siguiente hora)
```

**Características:**
- Framework: TensorFlow 2.x / Keras
- Lookback window: 168 horas (7 días)
- Normalización: MinMaxScaler
- Training/Validation split: 80/20
- Early stopping: patience=10 epochs
- Loss function: MSE
- Optimizer: Adam

**Ventajas:**
- ✅ Captura dependencias temporales de largo plazo
- ✅ Maneja patrones no lineales complejos
- ✅ Robusto ante ruido en los datos
- ✅ Predicciones más precisas que modelos estadísticos tradicionales

**Desventajas:**
- ⏱️ Entrenamiento más lento (2-5 minutos)
- 🧮 Requiere más recursos computacionales
- ❓ Menos interpretable que Prophet

### 2. Prophet - **Fallback**

**Características:**
- Framework: Facebook/Meta Prophet
- Estacionalidad: yearly, weekly, daily
- Changepoint detection: automático
- Intervalo de confianza: 95%

**Ventajas:**
- ⚡ Entrenamiento rápido (<30 segundos)
- 📊 Interpretable (componentes visualizables)
- 🔧 Maneja missing values automáticamente

**Se activa si:**
- TensorFlow no está instalado
- LSTM falla por datos insuficientes
- Usuario configura `use_lstm=False`

## 🔄 K-Fold Cross-Validation

### Implementación para Series Temporales

**Método: Time Series Split**
```
Fold 1:  [Train: 0→33%]  [Test: 33→36%]
Fold 2:  [Train: 0→66%]  [Test: 66→69%]
Fold 3:  [Train: 0→100%] [Test: 100→103%]
```

**Diferencia vs K-Fold estándar:**
- ❌ K-Fold estándar: mezcla aleatoriamente (rompe orden temporal)
- ✅ Time Series Split: respeta orden cronológico (realista)

**Configuración:**
- `n_splits`: 3-5 folds (recomendado 3 para datos <1000 registros)
- `test_size`: 24 horas por fold
- Métricas: MAE, RMSE, MAPE + desviación estándar

**Importancia:**
- Valida que el modelo funcione en diferentes períodos temporales
- Detecta overfitting
- Estima error real en producción

## 📊 Métricas de Evaluación

### Métricas de Ajuste Histórico

1. **MAE (Mean Absolute Error)**
   - Definición: Error absoluto promedio en MW
   - Fórmula: `(1/n) * Σ|y_true - y_pred|`
   - Interpretación: "En promedio, el modelo se equivoca ±X MW"

2. **RMSE (Root Mean Squared Error)**
   - Definición: Raíz del error cuadrático medio
   - Fórmula: `√((1/n) * Σ(y_true - y_pred)²)`
   - Interpretación: Penaliza errores grandes más fuertemente que MAE

3. **MAPE (Mean Absolute Percentage Error)**
   - Definición: Error porcentual promedio
   - Fórmula: `(100/n) * Σ|(y_true - y_pred)/y_true|`
   - Interpretación: Error relativo independiente de escala
   - **Criterios:**
     - < 5%: Excelente
     - 5-10%: Bueno
     - 10-20%: Moderado
     - > 20%: Revisar modelo

### Métricas de Cross-Validation

- **CV_MAE**: Promedio de MAE en todos los folds
- **CV_RMSE**: Promedio de RMSE en todos los folds
- **CV_MAPE**: Promedio de MAPE en todos los folds
- **Desviación estándar**: Estabilidad del modelo entre folds

**Ejemplo de salida:**
```
CV MAE:  1,234.50 MW (±123.45)
CV RMSE: 1,890.20 MW (±200.10)
CV MAPE: 4.8% (±0.5%)
```

## 🚀 Uso

### Instalación de Dependencias

```bash
# Opción 1: Solo LSTM (recomendado)
pip install tensorflow scikit-learn pandas numpy sqlalchemy

# Opción 2: LSTM + Prophet (fallback)
pip install tensorflow scikit-learn pandas numpy sqlalchemy prophet
```

### Ejemplo Básico

```python
from prediction_model_v2 import forecast_demand_pipeline_v2, print_forecast_summary

# Ejecutar predicción con configuración por defecto
forecast_df, model_info, metrics = forecast_demand_pipeline_v2(
    region_code='CISO',      # Región específica o None para todas
    days_back=180,           # Días de datos históricos
    forecast_hours=24,       # Horas a predecir
    use_lstm=True,           # Usar LSTM
    use_prophet_fallback=True,  # Fallback a Prophet si LSTM falla
    perform_cv=True,         # Ejecutar cross-validation
    n_splits=3               # Número de folds
)

# Imprimir resumen
if forecast_df is not None:
    print_forecast_summary(forecast_df, metrics)
    
    # Guardar predicciones
    forecast_df.to_csv('prediccion_24h.csv', index=False)
```

### Configuración Avanzada

```python
# Configuración para datos abundantes (>1000 registros)
forecast_df, model_info, metrics = forecast_demand_pipeline_v2(
    region_code='MISO',
    days_back=365,        # 1 año de datos
    forecast_hours=48,    # Predecir 48 horas
    use_lstm=True,
    perform_cv=True,
    n_splits=5            # 5 folds para más robustez
)

# Configuración solo Prophet (rápida)
forecast_df, model_info, metrics = forecast_demand_pipeline_v2(
    region_code='PJM',
    days_back=90,
    forecast_hours=24,
    use_lstm=False,              # Desactivar LSTM
    use_prophet_fallback=True,   # Usar Prophet directamente
    perform_cv=False             # Sin CV (más rápido)
)
```

## 📁 Estructura del Output

### DataFrame de Forecast
```python
forecast_df.columns
# ['ds', 'yhat', 'yhat_lower', 'yhat_upper']

forecast_df.head()
#                   ds        yhat  yhat_lower  yhat_upper
# 0 2025-11-14 14:00:00  24500.45    23275.43    25725.47
# 1 2025-11-14 15:00:00  25200.12    23940.11    26460.13
# ...
```

### Diccionario de Métricas
```python
metrics = {
    'MAE': 1234.56,         # Error absoluto medio
    'RMSE': 1890.23,        # Raíz del error cuadrático medio
    'MAPE': 4.8,            # Error porcentual (%)
    'model': 'LSTM',        # Modelo usado
    'CV_MAE': 1250.30,      # CV: Error absoluto medio
    'CV_RMSE': 1920.45,     # CV: RMSE
    'CV_MAPE': 0.049,       # CV: MAPE (como proporción, no %)
    'CV_MAE_std': 100.20,   # Desviación estándar de CV_MAE
    'CV_RMSE_std': 150.30,  # Desviación estándar de CV_RMSE
    'CV_MAPE_std': 0.005    # Desviación estándar de CV_MAPE
}
```

### Información del Modelo
```python
model_info = {
    'model': <keras.Model object>,  # Modelo entrenado
    'scaler': <MinMaxScaler>,       # Scaler (solo LSTM)
    'model_type': 'LSTM',           # 'LSTM' o 'Prophet'
    'lookback': 168,                # Ventana de entrada (horas)
    'trained_on': 4320,             # Número de registros
    'region': 'CISO'                # Región de predicción
}
```

## ⚙️ Configuración Recomendada por Escenario

### Escenario 1: Producción (Máxima Precisión)
```python
forecast_df, model_info, metrics = forecast_demand_pipeline_v2(
    region_code='CISO',
    days_back=365,         # 1 año completo
    forecast_hours=24,
    use_lstm=True,
    perform_cv=True,
    n_splits=5             # 5 folds
)
```

### Escenario 2: Desarrollo Rápido
```python
forecast_df, model_info, metrics = forecast_demand_pipeline_v2(
    region_code='CISO',
    days_back=90,          # 3 meses
    forecast_hours=24,
    use_lstm=True,
    perform_cv=False,      # Sin CV
    n_splits=0
)
```

### Escenario 3: Datos Limitados
```python
forecast_df, model_info, metrics = forecast_demand_pipeline_v2(
    region_code='CISO',
    days_back=30,          # 1 mes
    forecast_hours=24,
    use_lstm=False,        # Prophet es mejor con pocos datos
    use_prophet_fallback=True,
    perform_cv=False
)
```

## 🔧 Troubleshooting

### Error: "TensorFlow no está instalado"
```bash
pip install tensorflow
```

### Error: "Datos insuficientes"
- Mínimo: 200 registros (200 horas)
- Recomendado: 1000+ registros (1000+ horas)
- Verifica que la ingesta histórica se haya completado

### Warning: "CV omitido"
- Causa: Menos de 500 registros
- Solución: Cargar más datos históricos o desactivar CV

### Performance lenta
- LSTM en CPU es ~10x más lento que en GPU
- Reduce `epochs` de 50 a 30
- Reduce `lookback` de 168 a 72
- Usa Prophet como alternativa rápida

### MAPE muy alto (>15%)
- Aumenta `days_back` para más datos de entrenamiento
- Verifica outliers en los datos
- Considera features adicionales (temperatura, día de semana, etc.)

## 📚 Referencias

- **LSTM:** Hochreiter & Schmidhuber (1997) - "Long Short-Term Memory"
- **Time Series CV:** Hyndman & Athanasopoulos - "Forecasting: Principles and Practice"
- **Prophet:** Taylor & Letham (2018) - "Forecasting at Scale"
- **TensorFlow:** https://www.tensorflow.org/
- **Keras:** https://keras.io/

## 📝 Changelog

### V2.0 (2025-11-14)
- ✨ Agregado modelo LSTM con TensorFlow
- ✨ Implementado k-fold cross-validation para series temporales
- ✨ Prophet como fallback automático
- ✨ Métricas extendidas (CV_MAE, CV_RMSE, CV_MAPE + std)
- 🐛 Resuelto problema de timezone con Prophet
- 📊 Mejorado logging y feedback visual

### V1.0 (2025-11-13)
- ✨ Implementación inicial con Prophet
- ✨ Métricas básicas (MAE, RMSE, MAPE)
