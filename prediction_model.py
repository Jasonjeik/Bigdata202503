"""
Módulo de predicción de demanda energética usando modelos de series temporales.
Soporta Prophet para capturar estacionalidad y tendencias en datos de demanda eléctrica.
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("⚠️ Prophet no está instalado. Instala con: pip install prophet")

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
# FUNCIONES DE CARGA DE DATOS
# =========================

def load_demand_timeseries(region_code=None, days_back=90):
    """
    Carga serie temporal de demanda desde eia_aggregated_realtime.
    
    Args:
        region_code: Código de región específica (ej: 'CISO', 'MISO'). Si None, agrega todas las regiones.
        days_back: Cantidad de días históricos a cargar (default 90 días para entrenamiento).
    
    Returns:
        DataFrame con columnas ['ds', 'y'] para Prophet (ds=timestamp, y=demand).
    """
    engine = get_engine()
    
    cutoff = datetime.utcnow() - timedelta(days=days_back)
    
    if region_code:
        query = text("""
            SELECT period as ds, demand_mw as y, region
            FROM eia_aggregated_realtime
            WHERE period >= :cutoff
              AND region = :region
              AND demand_mw IS NOT NULL
            ORDER BY period
        """)
        df = pd.read_sql(query, engine, params={"cutoff": cutoff, "region": region_code})
    else:
        # Agregar todas las regiones
        query = text("""
            SELECT period as ds, SUM(demand_mw) as y
            FROM eia_aggregated_realtime
            WHERE period >= :cutoff
              AND demand_mw IS NOT NULL
            GROUP BY period
            ORDER BY period
        """)
        df = pd.read_sql(query, engine, params={"cutoff": cutoff})
    
    # Convertir a datetime y remover timezone (Prophet no lo soporta)
    df['ds'] = pd.to_datetime(df['ds'])
    if df['ds'].dt.tz is not None:
        df['ds'] = df['ds'].dt.tz_convert(None)
    
    return df

def load_demand_from_raw(region_code=None, days_back=180):
    """
    Carga serie temporal de demanda directamente desde rto_region_data (tabla cruda).
    Útil si eia_aggregated_realtime no tiene suficientes datos históricos.
    
    Args:
        region_code: Código de región (respondent). Si None, agrega todas.
        days_back: Días históricos a cargar.
    
    Returns:
        DataFrame con columnas ['ds', 'y'].
    """
    engine = get_engine()
    
    cutoff = datetime.utcnow() - timedelta(days=days_back)
    
    if region_code:
        query = text("""
            SELECT period as ds, value as y
            FROM rto_region_data
            WHERE period >= :cutoff
              AND respondent = :region
              AND type = 'D'
              AND value IS NOT NULL
            ORDER BY period
        """)
        df = pd.read_sql(query, engine, params={"cutoff": cutoff, "region": region_code})
    else:
        # Agregar todas las regiones
        query = text("""
            SELECT period as ds, SUM(value) as y
            FROM rto_region_data
            WHERE period >= :cutoff
              AND type = 'D'
              AND value IS NOT NULL
            GROUP BY period
            ORDER BY period
        """)
        df = pd.read_sql(query, engine, params={"cutoff": cutoff})
    
    # Convertir a datetime y remover timezone (Prophet no lo soporta)
    df['ds'] = pd.to_datetime(df['ds'])
    if df['ds'].dt.tz is not None:
        df['ds'] = df['ds'].dt.tz_convert('UTC').dt.tz_localize(None)
    
    return df

# =========================
# MODELOS DE PREDICCIÓN
# =========================

def train_prophet_model(df, yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True):
    """
    Entrena modelo Prophet para predicción de demanda.
    
    Args:
        df: DataFrame con columnas ['ds', 'y'] (timestamp, demanda).
        yearly_seasonality: Capturar estacionalidad anual.
        weekly_seasonality: Capturar estacionalidad semanal (días de la semana).
        daily_seasonality: Capturar estacionalidad diaria (horas del día).
    
    Returns:
        Modelo Prophet entrenado.
    """
    if not PROPHET_AVAILABLE:
        raise ImportError("Prophet no está instalado. Instala con: pip install prophet")
    
    # Preparar datos: eliminar outliers extremos (opcional)
    df_clean = df.copy()
    q1 = df_clean['y'].quantile(0.01)
    q99 = df_clean['y'].quantile(0.99)
    df_clean = df_clean[(df_clean['y'] >= q1) & (df_clean['y'] <= q99)]
    
    # CRÍTICO: Asegurar que 'ds' no tenga timezone (Prophet no lo soporta)
    # Estrategia: convertir a string ISO y re-parsear sin timezone
    df_clean = df_clean.reset_index(drop=True)
    
    # Convertir a objetos datetime de Python (sin timezone awareness)
    # Primero convertir a UTC si tiene timezone, luego extraer valores sin tz
    if hasattr(df_clean['ds'].dtype, 'tz') and df_clean['ds'].dtype.tz is not None:
        # Tiene timezone - convertir a UTC primero
        df_clean['ds'] = df_clean['ds'].dt.tz_convert('UTC')
    
    # Convertir a strings y re-parsear (fuerza naive datetime)
    df_clean['ds'] = df_clean['ds'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df_clean['ds'] = pd.to_datetime(df_clean['ds'], format='%Y-%m-%d %H:%M:%S')
    
    # Verificación final
    assert df_clean['ds'].dt.tz is None, "ERROR: ds todavía tiene timezone"
    
    print(f"📊 Entrenando modelo con {len(df_clean)} observaciones...")
    
    # Configurar Prophet
    model = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=daily_seasonality,
        changepoint_prior_scale=0.05,  # Flexibilidad en cambios de tendencia
        seasonality_prior_scale=10.0,  # Énfasis en estacionalidad
        interval_width=0.95  # Intervalo de confianza al 95%
    )
    
    # Ajustar modelo
    model.fit(df_clean)
    
    print(f"✅ Modelo entrenado exitosamente")
    
    return model

def predict_demand(model, periods_ahead=24):
    """
    Genera predicciones de demanda para las próximas N horas.
    
    Args:
        model: Modelo Prophet entrenado.
        periods_ahead: Horas futuras a predecir (default 24).
    
    Returns:
        DataFrame con predicciones: ['ds', 'yhat', 'yhat_lower', 'yhat_upper'].
    """
    # Crear dataframe futuro
    future = model.make_future_dataframe(periods=periods_ahead, freq='H')
    
    # Predecir
    forecast = model.predict(future)
    
    # Filtrar solo períodos futuros
    last_date = model.history['ds'].max()
    forecast_future = forecast[forecast['ds'] > last_date][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].reset_index(drop=True)
    
    return forecast_future

def get_model_performance(model, df):
    """
    Evalúa performance del modelo en datos históricos.
    
    Returns:
        Dict con métricas: MAE, RMSE, MAPE.
    """
    # Asegurar que df no tenga timezone
    df_eval = df.copy()
    if df_eval['ds'].dt.tz is not None:
        df_eval['ds'] = df_eval['ds'].dt.tz_convert('UTC').dt.tz_localize(None)
    
    # Predecir sobre datos históricos
    forecast = model.predict(df_eval[['ds']])
    
    # Calcular métricas
    y_true = df_eval['y'].values
    y_pred = forecast['yhat'].values
    
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    }

def cross_validate_model(model, df, horizon='24 hours', period='12 hours', initial='72 hours'):
    """
    Realiza validación cruzada (k-fold) del modelo Prophet.
    
    Args:
        model: Modelo Prophet entrenado.
        df: DataFrame con datos históricos ['ds', 'y'].
        horizon: Horizonte de predicción (default 24 horas).
        period: Período entre ventanas de validación (default 12 horas).
        initial: Tamaño mínimo del conjunto de entrenamiento inicial (default 72 horas).
    
    Returns:
        Tuple (cv_results_df, metrics_dict) con resultados de CV y métricas promedio.
    """
    try:
        from prophet.diagnostics import cross_validation, performance_metrics
        
        print(f"🔄 Ejecutando validación cruzada...")
        print(f"   Horizonte: {horizon}")
        print(f"   Período: {period}")
        print(f"   Training inicial: {initial}")
        
        # Ejecutar cross-validation
        df_cv = cross_validation(
            model, 
            initial=initial,
            period=period,
            horizon=horizon,
            parallel="processes"  # Paralelizar para mayor velocidad
        )
        
        # Calcular métricas de performance
        df_metrics = performance_metrics(df_cv)
        
        # Calcular promedios
        cv_metrics = {
            'CV_MAE': df_metrics['mae'].mean(),
            'CV_RMSE': df_metrics['rmse'].mean(),
            'CV_MAPE': df_metrics['mape'].mean(),
            'CV_Coverage': df_metrics['coverage'].mean()
        }
        
        print(f"✅ Validación cruzada completada")
        print(f"   MAE promedio:  {cv_metrics['CV_MAE']:.2f}")
        print(f"   RMSE promedio: {cv_metrics['CV_RMSE']:.2f}")
        print(f"   MAPE promedio: {cv_metrics['CV_MAPE']:.2%}")
        print(f"   Cobertura (95% CI): {cv_metrics['CV_Coverage']:.2%}")
        
        return df_cv, cv_metrics
        
    except Exception as e:
        print(f"⚠️ No se pudo completar validación cruzada: {e}")
        print("   Continuando sin CV...")
        return None, {}

# =========================
# PIPELINE COMPLETO
# =========================

def forecast_demand_pipeline(region_code=None, days_back=180, forecast_hours=24, use_raw_data=True):
    """
    Pipeline completo: carga datos, entrena modelo, genera predicciones.
    
    Args:
        region_code: Región específica o None para agregar todas.
        days_back: Días históricos para entrenamiento.
        forecast_hours: Horas futuras a predecir.
        use_raw_data: Si True usa rto_region_data, si False usa eia_aggregated_realtime.
    
    Returns:
        Tuple (forecast_df, model, metrics) con predicciones, modelo y métricas de performance.
    """
    print(f"\n{'='*70}")
    print(f"  PREDICCIÓN DE DEMANDA ENERGÉTICA - {region_code or 'TODAS LAS REGIONES'}")
    print(f"{'='*70}\n")
    
    # 1. Cargar datos
    print(f"📥 Cargando datos históricos ({days_back} días)...")
    if use_raw_data:
        df = load_demand_from_raw(region_code, days_back)
    else:
        df = load_demand_timeseries(region_code, days_back)
    
    if df.empty or len(df) < 48:
        print("❌ Datos insuficientes para entrenar modelo (mínimo 48 observaciones)")
        return None, None, None
    
    print(f"✅ {len(df)} observaciones cargadas")
    print(f"   Período: {df['ds'].min()} → {df['ds'].max()}")
    print(f"   Demanda promedio: {df['y'].mean():.2f} MW")
    
    # 2. Entrenar modelo
    print(f"\n🤖 Entrenando modelo Prophet...")
    model = train_prophet_model(df)
    
    # 3. Evaluar performance con datos históricos
    print(f"\n📈 Evaluando performance del modelo (datos históricos)...")
    metrics = get_model_performance(model, df)
    print(f"   MAE:  {metrics['MAE']:.2f} MW")
    print(f"   RMSE: {metrics['RMSE']:.2f} MW")
    print(f"   MAPE: {metrics['MAPE']:.2f}%")
    
    # 4. Validación cruzada (si hay suficientes datos)
    if len(df) >= 100:  # Mínimo 100 observaciones para CV
        print(f"\n🔄 Validación Cruzada (K-Fold)...")
        cv_results, cv_metrics = cross_validate_model(
            model, df,
            horizon='24 hours',
            period='12 hours',
            initial='72 hours'
        )
        # Agregar métricas de CV al dict principal
        metrics.update(cv_metrics)
    else:
        print(f"\n⚠️ Datos insuficientes para validación cruzada (mínimo 100 observaciones)")
        print(f"   Se usarán solo las métricas de ajuste histórico")
    
    # 5. Generar predicciones
    print(f"\n🔮 Generando predicciones para las próximas {forecast_hours} horas...")
    forecast = predict_demand(model, periods_ahead=forecast_hours)
    
    print(f"✅ Predicciones generadas:")
    print(f"   Inicio: {forecast['ds'].min()}")
    print(f"   Fin:    {forecast['ds'].max()}")
    print(f"   Demanda promedio predicha: {forecast['yhat'].mean():.2f} MW")
    
    print(f"\n{'='*70}\n")
    
    return forecast, model, metrics

# =========================
# EJEMPLO DE USO
# =========================

if __name__ == "__main__":
    # Ejemplo: Predicción para California ISO (CISO)
    forecast, model, metrics = forecast_demand_pipeline(
        region_code='CISO',
        days_back=180,
        forecast_hours=24,
        use_raw_data=True
    )
    
    if forecast is not None:
        print("\n📋 Primeras 10 predicciones:")
        print(forecast.head(10).to_string(index=False))
        
        # Guardar predicciones a CSV (opcional)
        output_file = "demand_forecast_24h.csv"
        forecast.to_csv(output_file, index=False)
        print(f"\n💾 Predicciones guardadas en: {output_file}")
