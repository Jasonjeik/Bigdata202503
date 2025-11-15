"""
Módulo de predicción de demanda energética con múltiples enfoques:
1. Prophet (Facebook/Meta) - Para captura de estacionalidad
2. LSTM con TensorFlow - Deep Learning para series temporales
3. K-Fold Cross Validation adaptado para series temporales
"""

import os
import json
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
from pathlib import Path
from joblib import dump, load
import warnings
warnings.filterwarnings('ignore')
import traceback

# Intentar imports de Prophet
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("⚠️ Prophet no disponible")

# Intentar imports de TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.preprocessing import MinMaxScaler
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow no disponible. Instala con: pip install tensorflow")

from sklearn.metrics import mean_absolute_error, mean_squared_error

# =========================
# CONFIGURACIÓN BD
# =========================
DB_USER = "prj1_admin"
DB_PASS = "Bigdataproyecto1"
DB_HOST = "bigdataproyecto1.postgres.database.azure.com"
DB_PORT = "5432"
DB_NAME = "proyecto1"
SCHEMA = "eia"

def get_engine():
    """Crea engine de SQLAlchemy para conexión a PostgreSQL."""
    return create_engine(
        f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
        connect_args={
            "sslmode": "require",
            "options": f"-csearch_path={SCHEMA}"
        }
    )

# =========================
# CARGA DE DATOS
# =========================

def load_demand_data(region_code=None, days_back=180):
    """
    Carga datos de demanda desde rto_region_data (tabla raw con más historia).
    
    Returns:
        DataFrame con columnas: timestamp, demand_mw
    """
    engine = get_engine()
    cutoff = datetime.utcnow() - timedelta(days=days_back)
    
    if region_code:
        query = text("""
            SELECT period as timestamp, value as demand_mw
            FROM rto_region_data
            WHERE period >= :cutoff
              AND respondent = :region
              AND type = 'D'
              AND value IS NOT NULL
            ORDER BY period
        """)
        df = pd.read_sql(query, engine, params={"cutoff": cutoff, "region": region_code})
    else:
        query = text("""
            SELECT period as timestamp, SUM(value) as demand_mw
            FROM rto_region_data
            WHERE period >= :cutoff
              AND type = 'D'
              AND value IS NOT NULL
            GROUP BY period
            ORDER BY period
        """)
        df = pd.read_sql(query, engine, params={"cutoff": cutoff})
    
    # Convertir timestamp a datetime sin timezone
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    if df['timestamp'].dt.tz is not None:
        df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')
    df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S')
    
    return df

# =========================
# GESTIÓN DE MODELOS (SERIALIZACIÓN)
# =========================

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models_v2")

def _region_key(region_code: str | None) -> str:
    return (region_code or 'ALL').replace("/", "_").replace(" ", "_")

def _ensure_model_dir():
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

def _paths_for_region(region_code: str | None):
    rk = _region_key(region_code)
    return {
        'lstm_model': os.path.join(MODEL_DIR, f"{rk}_lstm_model.h5"),
        'lstm_scaler': os.path.join(MODEL_DIR, f"{rk}_lstm_scaler.joblib"),
        'prophet': os.path.join(MODEL_DIR, f"{rk}_prophet.joblib"),
        'meta': os.path.join(MODEL_DIR, f"{rk}_meta.json"),
    }

def _has_saved_artifacts(region_code: str | None) -> bool:
    """Verifica si existen artefactos guardados para la región sin cargarlos."""
    paths = _paths_for_region(region_code)
    try:
        if not os.path.exists(paths['meta']):
            return False
        with open(paths['meta'], 'r', encoding='utf-8') as f:
            meta = json.load(f)
        mtype = meta.get('model_type')
        if mtype == 'LSTM':
            return os.path.exists(paths['lstm_model']) and os.path.exists(paths['lstm_scaler'])
        if mtype == 'Prophet':
            return os.path.exists(paths['prophet'])
        return False
    except Exception:
        return False

def save_best_model(region_code, model_type, model, scaler, lookback, sigma, trained_on, metrics: dict):
    """Guarda el mejor modelo encontrado junto con metadatos por región."""
    _ensure_model_dir()
    paths = _paths_for_region(region_code)
    meta = {
        'region': region_code or 'ALL',
        'model_type': model_type,
        'lookback': int(lookback) if lookback is not None else None,
        'sigma': float(sigma) if sigma is not None else None,
        'trained_on': int(trained_on),
        'saved_at': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        'metrics': {
            'MAE': float(metrics.get('MAE', np.nan)),
            'RMSE': float(metrics.get('RMSE', np.nan)),
            'MAPE': float(metrics.get('MAPE', np.nan)),
        }
    }
    if model_type == 'LSTM':
        model.save(paths['lstm_model'])
        if scaler is not None:
            dump(scaler, paths['lstm_scaler'])
    elif model_type == 'Prophet':
        dump(model, paths['prophet'])
    with open(paths['meta'], 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return paths

def load_saved_model(region_code):
    """Carga el modelo pre-entrenado por región si existe. Retorna tuple o None."""
    paths = _paths_for_region(region_code)
    try:
        if os.path.exists(paths['meta']):
            with open(paths['meta'], 'r', encoding='utf-8') as f:
                meta = json.load(f)
            mtype = meta.get('model_type')
            if mtype == 'LSTM' and os.path.exists(paths['lstm_model']) and os.path.exists(paths['lstm_scaler']):
                # Intento 1: carga normal
                try:
                    mdl = keras.models.load_model(paths['lstm_model'])
                except Exception:
                    # Intento 2: cargar sin compilar (mayor tolerancia entre versiones)
                    print(f"⚠️ Falló carga compilada de LSTM '{paths['lstm_model']}'. Probando compile=False...")
                    mdl = keras.models.load_model(paths['lstm_model'], compile=False)
                scl = load(paths['lstm_scaler'])
                return {
                    'model_type': 'LSTM',
                    'model': mdl,
                    'scaler': scl,
                    'lookback': meta.get('lookback', 168),
                    'sigma': meta.get('sigma'),
                    'trained_on': meta.get('trained_on', 0),
                    'meta': meta
                }
            if mtype == 'Prophet' and os.path.exists(paths['prophet']):
                mdl = load(paths['prophet'])
                return {
                    'model_type': 'Prophet',
                    'model': mdl,
                    'scaler': None,
                    'lookback': None,
                    'sigma': meta.get('sigma'),
                    'trained_on': meta.get('trained_on', 0),
                    'meta': meta
                }
    except Exception as e:
        print(f"⚠️ Error cargando modelo guardado para región {region_code or 'ALL'}: {e}")
        traceback.print_exc()
        return None
    return None

# =========================
# PREPARACIÓN DE DATOS PARA LSTM
# =========================

def create_sequences(data, lookback=168):
    """
    Crea secuencias para entrenamiento de LSTM.
    
    Args:
        data: Array 1D de valores de demanda
        lookback: Ventana de tiempo (horas) para predicción (default 168 = 7 días)
    
    Returns:
        X: Secuencias de entrada (samples, lookback, 1)
        y: Valores objetivo (samples,)
    """
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(data[i])
    return np.array(X), np.array(y)

# =========================
# MODELO 1: LSTM CON TENSORFLOW
# =========================

def build_lstm_model(lookback=168, units=64):
    """
    Construye modelo LSTM para predicción de series temporales.
    
    Args:
        lookback: Tamaño de la ventana de entrada
        units: Número de unidades LSTM
    
    Returns:
        Modelo Keras compilado
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow no está instalado")
    
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(units // 2, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_lstm_model(df, lookback=168, epochs=30, batch_size=32):
    """
    Entrena modelo LSTM con datos de demanda.
    
    Returns:
        model: Modelo entrenado
        scaler: Scaler ajustado
        history: Historia de entrenamiento
    """
    print(f"🔧 Preparando datos para LSTM (lookback={lookback} horas)...")
    
    # Normalizar datos
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df['demand_mw'].values.reshape(-1, 1))
    
    # Crear secuencias
    X, y = create_sequences(data_scaled.flatten(), lookback)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    # Split train/val (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"📊 Datos de entrenamiento: {X_train.shape[0]} secuencias")
    print(f"📊 Datos de validación: {X_val.shape[0]} secuencias")
    
    # Construir y entrenar modelo
    model = build_lstm_model(lookback)
    
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    print(f"\n🤖 Entrenando LSTM (epochs={epochs})...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0
    )
    
    print(f"✅ Entrenamiento completado (mejor val_loss: {min(history.history['val_loss']):.4f})")
    
    return model, scaler, history

def predict_lstm(model, scaler, last_sequence, forecast_hours=24):
    """
    Genera predicciones con LSTM para las próximas N horas.
    
    Args:
        model: Modelo LSTM entrenado
        scaler: MinMaxScaler ajustado
        last_sequence: Últimas 'lookback' horas de demanda (array)
        forecast_hours: Horas a predecir
    
    Returns:
        Array con predicciones
    """
    predictions = []
    current_seq = last_sequence.copy()
    
    for _ in range(forecast_hours):
        # Predecir siguiente valor
        current_seq_scaled = scaler.transform(current_seq.reshape(-1, 1))
        X_pred = current_seq_scaled.reshape(1, len(current_seq), 1)
        pred_scaled = model.predict(X_pred, verbose=0)[0, 0]
        pred = scaler.inverse_transform([[pred_scaled]])[0, 0]
        
        predictions.append(pred)
        
        # Actualizar secuencia (rolling window)
        current_seq = np.append(current_seq[1:], pred)
    
    return np.array(predictions)

# =========================
# MODELO 2: PROPHET (FALLBACK)
# =========================

def train_prophet_model(df):
    """
    Entrena modelo Prophet como alternativa.
    
    Returns:
        Modelo Prophet entrenado o None si falla
    """
    if not PROPHET_AVAILABLE:
        return None
    
    try:
        # Preparar datos para Prophet
        df_prophet = df.rename(columns={'timestamp': 'ds', 'demand_mw': 'y'})
        
        # Remover outliers
        q1 = df_prophet['y'].quantile(0.01)
        q99 = df_prophet['y'].quantile(0.99)
        df_prophet = df_prophet[(df_prophet['y'] >= q1) & (df_prophet['y'] <= q99)]
        
        print(f"📊 Entrenando Prophet con {len(df_prophet)} observaciones...")
        
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0,
            interval_width=0.95
        )
        
        model.fit(df_prophet)
        print("✅ Prophet entrenado exitosamente")
        return model
        
    except Exception as e:
        print(f"⚠️ Prophet falló: {e}")
        return None

def predict_prophet(model, periods_ahead=24):
    """Genera predicciones con Prophet."""
    if model is None:
        return None
    
    future = model.make_future_dataframe(periods=periods_ahead, freq='H')
    forecast = model.predict(future)
    return forecast.tail(periods_ahead)

# =========================
# INTERVALOS / VARIANZA POR REGIÓN
# =========================

def compute_sigma(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Desviación estándar de residuales."""
    if y_true is None or y_pred is None:
        return np.nan
    res = (y_true - y_pred)
    return float(np.nanstd(res))

def intervals_from_sigma(yhat: np.ndarray, sigma: float, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    """Intervalos para horizonte multi-paso asumiendo error ~ sqrt(h)."""
    if sigma is None or np.isnan(sigma):
        lower = yhat * 0.95
        upper = yhat * 1.05
        return lower, upper
    steps = np.arange(1, len(yhat) + 1)
    scale = np.sqrt(steps)
    half_width = z * sigma * scale
    lower = yhat - half_width
    upper = yhat + half_width
    return lower, upper

# =========================
# K-FOLD CROSS VALIDATION PARA SERIES TEMPORALES
# =========================

def time_series_cv_split(data, n_splits=5, test_size=24):
    """
    Crea splits de validación cruzada respetando el orden temporal.
    
    Args:
        data: DataFrame con datos temporales
        n_splits: Número de folds
        test_size: Tamaño del conjunto de prueba (horas)
    
    Yields:
        train_idx, test_idx para cada fold
    """
    n_samples = len(data)
    indices = np.arange(n_samples)
    
    # Calcular tamaño de cada fold
    fold_size = (n_samples - test_size) // n_splits
    
    for i in range(n_splits):
        # Train: desde inicio hasta (i+1)*fold_size
        train_end = (i + 1) * fold_size
        test_end = min(train_end + test_size, n_samples)
        
        if test_end - train_end < test_size:
            continue
        
        train_idx = indices[:train_end]
        test_idx = indices[train_end:test_end]
        
        yield train_idx, test_idx

def cross_validate_lstm(df, n_splits=5, lookback=168, epochs=30):
    """
    Realiza k-fold cross-validation con LSTM.
    
    Returns:
        Dict con métricas promedio: CV_MAE, CV_RMSE, CV_MAPE
    """
    print(f"\n🔄 Iniciando {n_splits}-Fold Cross Validation (LSTM)...")
    
    fold_metrics = {'mae': [], 'rmse': [], 'mape': []}
    
    for fold, (train_idx, test_idx) in enumerate(time_series_cv_split(df, n_splits, test_size=24), 1):
        print(f"\n📂 Fold {fold}/{n_splits}")
        
        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_test = df.iloc[test_idx].reset_index(drop=True)
        
        # Entrenar modelo
        model, scaler, _ = train_lstm_model(df_train, lookback, epochs, batch_size=32)
        
        # Predecir
        last_seq = df_train['demand_mw'].tail(lookback).values
        predictions = predict_lstm(model, scaler, last_seq, forecast_hours=len(df_test))
        
        # Calcular métricas
        y_true = df_test['demand_mw'].values
        y_pred = predictions[:len(y_true)]
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        fold_metrics['mae'].append(mae)
        fold_metrics['rmse'].append(rmse)
        fold_metrics['mape'].append(mape)
        
        print(f"   MAE: {mae:,.2f} MW | RMSE: {rmse:,.2f} MW | MAPE: {mape:.2f}%")
    
    # Promedios
    cv_mae = np.mean(fold_metrics['mae'])
    cv_rmse = np.mean(fold_metrics['rmse'])
    cv_mape = np.mean(fold_metrics['mape'])
    
    print(f"\n✅ Cross-Validation Completado:")
    print(f"   CV MAE:  {cv_mae:,.2f} MW (±{np.std(fold_metrics['mae']):,.2f})")
    print(f"   CV RMSE: {cv_rmse:,.2f} MW (±{np.std(fold_metrics['rmse']):,.2f})")
    print(f"   CV MAPE: {cv_mape:.2f}% (±{np.std(fold_metrics['mape']):.2f})")
    
    return {
        'CV_MAE': cv_mae,
        'CV_RMSE': cv_rmse,
        'CV_MAPE': cv_mape / 100,  # Como proporción
        'CV_MAE_std': np.std(fold_metrics['mae']),
        'CV_RMSE_std': np.std(fold_metrics['rmse']),
        'CV_MAPE_std': np.std(fold_metrics['mape']) / 100
    }

# =========================
# PIPELINE PRINCIPAL
# =========================

def forecast_demand_pipeline_v2(region_code=None, days_back=180, forecast_hours=24,
                                 use_lstm=True, use_prophet_fallback=True,
                                 perform_cv=True, n_splits=3,
                                 force_retrain: bool = False):
    """
    Pipeline completo de predicción con múltiples enfoques.
    
    Args:
        region_code: Región específica o None para todas
        days_back: Días históricos para entrenamiento
        forecast_hours: Horas a predecir (default 24)
        use_lstm: Usar LSTM como modelo principal
        use_prophet_fallback: Si LSTM falla, usar Prophet
        perform_cv: Ejecutar cross-validation
        n_splits: Número de folds para CV
    
    Returns:
        forecast_df: DataFrame con predicciones
        model_info: Dict con modelo y metadata
        metrics: Dict con métricas de desempeño
    """
    print("="*70)
    print("🚀 PIPELINE DE PREDICCIÓN DE DEMANDA ELÉCTRICA V2")
    print("="*70)
    
    # 1. Cargar datos
    print(f"\n📥 Cargando datos (últimos {days_back} días)...")
    df = load_demand_data(region_code, days_back)
    
    if len(df) < 200:
        print(f"❌ Datos insuficientes: {len(df)} registros (mínimo 200)")
        return None, None, None
    
    print(f"✅ {len(df)} registros cargados")
    print(f"   Rango: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"   Demanda promedio: {df['demand_mw'].mean():,.2f} MW")
    
    model_used = None
    model = None
    scaler = None
    predictions = None
    lookback = None
    saved = None

    # 1.1 Intentar cargar modelo pre-entrenado
    if not force_retrain:
        saved = load_saved_model(region_code)
        if saved is not None:
            model_used = saved['model_type']
            model = saved['model']
            scaler = saved.get('scaler')
            lookback = saved.get('lookback')
            print(f"📦 Usando modelo pre-entrenado para región {region_code or 'ALL'}: {model_used}")
    
    # 2. Predicción con modelo cargado o entrenamiento
    if model_used == 'LSTM' and model is not None and scaler is not None:
        try:
            lookback = int(lookback or 168)
            last_sequence = df['demand_mw'].tail(lookback).values
            if len(last_sequence) < lookback:
                raise ValueError("Historial insuficiente para lookback del modelo guardado")
            predictions = predict_lstm(model, scaler, last_sequence, forecast_hours)
            print("✅ Predicciones generadas con LSTM (pre-entrenado)")
        except Exception as e:
            print(f"⚠️ Falló predicción con modelo guardado LSTM: {e}")
            model_used = None
            model = None
            scaler = None

    elif model_used == 'Prophet' and model is not None:
        try:
            forecast_prophet = predict_prophet(model, forecast_hours)
            predictions = forecast_prophet['yhat'].values
            print("✅ Predicciones generadas con Prophet (pre-entrenado)")
        except Exception as e:
            print(f"⚠️ Falló predicción con modelo guardado Prophet: {e}")
            model_used = None
            model = None

    # 2.b Si no hay modelo cargado, entrenar candidatos
    if model_used is None:
        lstm_candidate = None
        lstm_scaler = None
        lstm_metrics = None
        lstm_lookback = None
        prophet_candidate = None
        prophet_metrics = None

        # Entrenar LSTM
        if use_lstm and TENSORFLOW_AVAILABLE:
            try:
                print(f"\n🤖 Método 1: LSTM con TensorFlow")
                lstm_lookback = min(168, len(df) // 3)
                lstm_candidate, lstm_scaler, _ = train_lstm_model(df, lstm_lookback, epochs=30)
                last_sequence = df['demand_mw'].tail(lstm_lookback).values
                lstm_pred = predict_lstm(lstm_candidate, lstm_scaler, last_sequence, forecast_hours)
                # Métricas de holdout
                test_size = min(24, len(df) // 10)
                df_test = df.tail(test_size + lstm_lookback)
                last_seq = df_test['demand_mw'].head(lstm_lookback).values
                test_pred = predict_lstm(lstm_candidate, lstm_scaler, last_seq, test_size)
                y_true = df_test['demand_mw'].tail(test_size).values
                y_pred = test_pred
                lstm_mae = mean_absolute_error(y_true, y_pred)
                lstm_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                lstm_mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                lstm_metrics = {'MAE': lstm_mae, 'RMSE': lstm_rmse, 'MAPE': lstm_mape}
                print(f"✅ LSTM listo. Holdout → MAE {lstm_mae:.2f}, RMSE {lstm_rmse:.2f}, MAPE {lstm_mape:.2f}%")
            except Exception as e:
                print(f"⚠️ LSTM falló: {e}")
                lstm_candidate = None
                lstm_scaler = None

        # Entrenar Prophet
        if use_prophet_fallback and PROPHET_AVAILABLE:
            try:
                print(f"\n🤖 Método 2: Prophet")
                prophet_candidate = train_prophet_model(df)
                if prophet_candidate is not None:
                    # Evaluar en últimas forecast_hours horas
                    y_true_p = df['demand_mw'].tail(min(forecast_hours, len(df))).values
                    fcst = predict_prophet(prophet_candidate, forecast_hours)
                    y_pred_p = fcst['yhat'].values[:len(y_true_p)]
                    p_mae = mean_absolute_error(y_true_p, y_pred_p)
                    p_rmse = np.sqrt(mean_squared_error(y_true_p, y_pred_p))
                    p_mape = np.mean(np.abs((y_true_p - y_pred_p) / y_true_p)) * 100
                    prophet_metrics = {'MAE': p_mae, 'RMSE': p_rmse, 'MAPE': p_mape}
                    print(f"✅ Prophet listo. Holdout → MAE {p_mae:.2f}, RMSE {p_rmse:.2f}, MAPE {p_mape:.2f}%")
            except Exception as e:
                print(f"⚠️ Prophet falló: {e}")
                prophet_candidate = None

        # Seleccionar mejor por RMSE (fallback a MAPE)
        selected = None
        if lstm_candidate is not None and prophet_candidate is not None:
            if lstm_metrics['RMSE'] <= prophet_metrics['RMSE']:
                selected = ('LSTM', lstm_candidate, lstm_scaler, lstm_lookback, lstm_metrics)
            else:
                selected = ('Prophet', prophet_candidate, None, None, prophet_metrics)
        elif lstm_candidate is not None:
            selected = ('LSTM', lstm_candidate, lstm_scaler, lstm_lookback, lstm_metrics)
        elif prophet_candidate is not None:
            selected = ('Prophet', prophet_candidate, None, None, prophet_metrics)

        if selected is None:
            print("\n❌ No se pudo entrenar ningún modelo")
            return None, None, None

        model_used, model, scaler, lookback, selected_metrics = selected
        print(f"🏆 Modelo seleccionado: {model_used}")

        # Generar predicciones con el modelo seleccionado
        if model_used == 'LSTM':
            last_sequence = df['demand_mw'].tail(lookback).values
            predictions = predict_lstm(model, scaler, last_sequence, forecast_hours)
        else:
            fcst = predict_prophet(model, forecast_hours)
            predictions = fcst['yhat'].values
        try:
            print(f"\n🤖 Método 1: LSTM con TensorFlow")
            lookback = min(168, len(df) // 3)  # 7 días o 1/3 de los datos
            
            model, scaler, history = train_lstm_model(df, lookback, epochs=30)
            
            # Generar predicciones
            last_sequence = df['demand_mw'].tail(lookback).values
            predictions = predict_lstm(model, scaler, last_sequence, forecast_hours)
            
            model_used = 'LSTM'
            print(f"✅ Predicciones generadas con LSTM")
            
        except Exception as e:
            print(f"⚠️ LSTM falló: {e}")
            model = None
    
    # 3. Fallback a Prophet si LSTM falló
    if model_used is None and use_prophet_fallback and PROPHET_AVAILABLE:
        try:
            print(f"\n🤖 Método 2 (Fallback): Prophet")
            prophet_model = train_prophet_model(df)
            
            if prophet_model is not None:
                forecast_prophet = predict_prophet(prophet_model, forecast_hours)
                predictions = forecast_prophet['yhat'].values
                model = prophet_model
                model_used = 'Prophet'
                print(f"✅ Predicciones generadas con Prophet")
                
        except Exception as e:
            print(f"⚠️ Prophet también falló: {e}")
    
    if predictions is None:
        print("\n❌ Ningún modelo pudo generar predicciones")
        return None, None, None
    
    # 4. Crear DataFrame de forecast
    last_timestamp = df['timestamp'].max()
    future_timestamps = pd.date_range(
        start=last_timestamp + timedelta(hours=1),
        periods=forecast_hours,
        freq='H'
    )
    forecast_df = pd.DataFrame({
        'ds': future_timestamps,
        'yhat': predictions,
    })
    
    # 5. Métricas de ajuste histórico
    print(f"\n📈 Evaluando ajuste en datos históricos...")
    
    # Si usamos modelo pre-entrenado, usar métricas guardadas
    if saved is not None and not force_retrain:
        saved_metrics = saved.get('meta', {}).get('metrics', {})
        mae = saved_metrics.get('MAE', 0.0)
        rmse = saved_metrics.get('RMSE', 0.0)
        mape = saved_metrics.get('MAPE', 0.0)
        print(f"📦 Usando métricas del modelo pre-entrenado")
    else:
        # Calcular métricas para modelo recién entrenado
        if model_used == 'LSTM':
            # Evaluar en últimas 24 horas
            test_size = min(24, len(df) // 10)
            df_test = df.tail(test_size + lookback)
            last_seq = df_test['demand_mw'].head(lookback).values
            test_pred = predict_lstm(model, scaler, last_seq, test_size)
            y_true = df_test['demand_mw'].tail(test_size).values
            y_pred = test_pred
            
        else:  # Prophet
            y_true = df['demand_mw'].tail(forecast_hours).values
            y_pred = predictions[:len(y_true)]
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'model': model_used
    }
    
    print(f"   MAE:  {mae:,.2f} MW")
    print(f"   RMSE: {rmse:,.2f} MW")
    print(f"   MAPE: {mape:.2f}%")
    
    # 6. Cross Validation (solo si entrenamos modelo nuevo)
    if saved is not None and not force_retrain:
        print(f"\n⏭️ CV omitido (usando modelo pre-entrenado)")
    elif perform_cv and len(df) >= 500 and model_used == 'LSTM':
        print(f"\n🔄 Ejecutando Cross-Validation (modelo nuevo)...")
        cv_metrics = cross_validate_lstm(df, n_splits, lookback, epochs=30)
        metrics.update(cv_metrics)
    else:
        print(f"\n⚠️ CV omitido (datos insuficientes, modelo no es LSTM, o deshabilitado)")
    
    # 7. Calcular varianza por región y construir intervalos
    if saved is not None and not force_retrain:
        # Usar sigma guardado del modelo pre-entrenado
        region_sigma = saved.get('sigma', rmse)  # Fallback a RMSE si no hay sigma guardado
    else:
        # Calcular sigma de datos recién evaluados
        region_sigma = compute_sigma(y_true, y_pred)
    if model_used == 'Prophet':
        # Si el modelo Prophet genera intervalos nativos, úsalos; si no, usa sigma
        try:
            fc_prophet = predict_prophet(model, forecast_hours)
            forecast_df['yhat_lower'] = fc_prophet['yhat_lower'].values
            forecast_df['yhat_upper'] = fc_prophet['yhat_upper'].values
        except Exception:
            lower, upper = intervals_from_sigma(forecast_df['yhat'].values, region_sigma)
            forecast_df['yhat_lower'] = lower
            forecast_df['yhat_upper'] = upper
    else:
        lower, upper = intervals_from_sigma(forecast_df['yhat'].values, region_sigma)
        forecast_df['yhat_lower'] = lower
        forecast_df['yhat_upper'] = upper

    # 8. Guardar el mejor modelo si acabamos de entrenar o si no existía
    if saved is None or force_retrain:
        try:
            save_best_model(
                region_code=region_code,
                model_type=model_used,
                model=model,
                scaler=scaler,
                lookback=lookback,
                sigma=region_sigma,
                trained_on=len(df),
                metrics=metrics,
            )
            print("💾 Modelo guardado como mejor para la región")
        except Exception as e:
            print(f"⚠️ No se pudo guardar el modelo: {e}")

    model_info = {
        'model': model,
        'scaler': scaler if model_used == 'LSTM' else None,
        'model_type': model_used,
        'lookback': lookback if model_used == 'LSTM' else None,
        'trained_on': len(df),
        'region': region_code or 'ALL',
        'sigma': region_sigma,
        'pretrained_used': saved is not None and not force_retrain
    }
    
    return forecast_df, model_info, metrics

# =========================
# FUNCIONES DE UTILIDAD
# =========================

def print_forecast_summary(forecast_df, metrics):
    """Imprime resumen de predicciones y métricas."""
    print("\n" + "="*70)
    print("✅ PREDICCIÓN COMPLETADA")
    print("="*70)
    
    print(f"\n🔮 Predicciones para las próximas {len(forecast_df)} horas:")
    print(forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head(10).to_string(index=False))
    
    print(f"\n📊 Métricas de Desempeño ({metrics['model']}):")
    print(f"   MAE:  {metrics['MAE']:,.2f} MW")
    print(f"   RMSE: {metrics['RMSE']:,.2f} MW")
    print(f"   MAPE: {metrics['MAPE']:.2f}%")
    
    if 'CV_MAE' in metrics:
        print(f"\n🔄 Métricas de Cross-Validation:")
        print(f"   CV MAE:  {metrics['CV_MAE']:,.2f} MW (±{metrics.get('CV_MAE_std', 0):,.2f})")
        print(f"   CV RMSE: {metrics['CV_RMSE']:,.2f} MW (±{metrics.get('CV_RMSE_std', 0):,.2f})")
        print(f"   CV MAPE: {metrics['CV_MAPE']:.2%} (±{metrics.get('CV_MAPE_std', 0):.2%})")
