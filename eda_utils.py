"""
eda_utils.py
~~~~~~~~~~~~~~~~

Este módulo proporciona funciones reutilizables para realizar un análisis
exploratorio de datos (EDA) sobre los datos de la EIA ingesta en PostgreSQL.
Las funciones están diseñadas para mantener el flujo de trabajo simple y
legible, facilitando tareas comunes como la verificación de valores
faltantes, imputación por mediana o moda, detección de valores atípicos,
construcción de tablas agregadas por minuto y región, detección de
anomalías mediante Isolation Forest y cálculo de rankings de déficit.

Para evitar dependencias innecesarias, este módulo utiliza únicamente
`pandas`, `numpy` y `scikit‑learn` (que están disponibles en el entorno).

Uso básico:

    from eda_utils import (
        remove_high_missing, impute_missing, identify_outliers,
        aggregate_region_data, detect_anomalies, top_deficit_regions
    )

    # Cargar datos en DataFrames df_region, df_fuel, df_inter.
    df_region = remove_high_missing(df_region, threshold=0.8)
    df_region = impute_missing(df_region)
    # repetir para df_fuel y df_inter
    agg = aggregate_region_data(df_region, df_fuel, df_inter)
    agg = detect_anomalies(agg, contamination=0.02)
    top5 = top_deficit_regions(agg, n=5)

Todas las funciones devuelven nuevos DataFrames; no modifican los
argumentos de entrada in situ.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from typing import Dict, Optional, List

def remove_high_missing(df: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    """Elimina filas con un porcentaje de valores faltantes superior a ``threshold``.

    :param df: DataFrame de entrada.
    :param threshold: Porcentaje máximo de nulos permitido por fila (0‑1).
    :return: Nuevo DataFrame sin filas con demasiados nulos.

    Esta función calcula el ratio de valores nulos por fila y elimina las
    filas en las que dicho ratio supera el ``threshold``.  Elimina pocas
    filas si los datos están completos y actúa como primera línea de defensa
    contra registros corruptos.
    """
    if df.empty:
        return df.copy()
    # proporción de nulos por fila
    null_ratio = df.isna().mean(axis=1)
    mask = null_ratio <= threshold
    return df.loc[mask].reset_index(drop=True)


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Imputa valores faltantes en un DataFrame usando mediana y moda.

    Columnas numéricas se rellenan con la mediana; columnas
    categóricas (tipo objeto) se rellenan con la moda (valor más frecuente).

    :param df: DataFrame con valores faltantes.
    :return: Nuevo DataFrame con valores imputados.
    """
    if df.empty:
        return df.copy()
    result = df.copy()
    for col in result.columns:
        if result[col].isna().any():
            if pd.api.types.is_numeric_dtype(result[col]):
                # usar la mediana para numéricos; si todo es NaN, se ignora
                median = result[col].median()
                if pd.notna(median):
                    result[col] = result[col].fillna(median)
            else:
                # para strings/categorías, usar la moda; si varias modas, tomar la primera
                mode_val = result[col].mode(dropna=True)
                if not mode_val.empty:
                    result[col] = result[col].fillna(mode_val.iloc[0])
    return result


def identify_outliers(
    df: pd.DataFrame,
    col: str,
    method: str = "iqr",
    z_thresh: float = 3.0
) -> pd.DataFrame:
    """Marca valores atípicos en una columna numérica.

    Puede utilizar el método del rango intercuartil (IQR) o
    una regla basada en el z‑score.  Devuelve un DataFrame con una nueva
    columna booleana ``is_outlier`` que indica si cada fila es atípica
    respecto a la columna ``col``.

    :param df: DataFrame de entrada.
    :param col: Nombre de la columna numérica a evaluar.
    :param method: 'iqr' (por defecto) o 'zscore'.
    :param z_thresh: Umbral de z‑score para considerar un valor atípico si
                     ``method='zscore'``.
    :return: DataFrame con columna ``is_outlier``.
    """
    if df.empty or col not in df.columns:
        out = df.copy()
        out["is_outlier"] = False
        return out
    series = pd.to_numeric(df[col], errors="coerce")
    outlier_flags = pd.Series(False, index=df.index)
    if method == "zscore":
        # calc z-score normalizado
        mean_val = series.mean()
        std_val = series.std(ddof=0)
        if std_val > 0:
            z_scores = (series - mean_val) / std_val
            outlier_flags = z_scores.abs() > z_thresh
    else:
        # método IQR
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_flags = (series < lower) | (series > upper)
    out = df.copy()
    out["is_outlier"] = outlier_flags.fillna(False)
    return out


def aggregate_region_data(
    df_region: pd.DataFrame,
    df_fuel: pd.DataFrame,
    df_interchange: pd.DataFrame,
    coord_map: Optional[Dict[str, Dict[str, float]]] = None
) -> pd.DataFrame:
    """Construye una tabla agregada minuto a minuto por región.

    Esta función produce una tabla con índice (region, period) y columnas
    de energía generada por cada fuente (`fueltype`), demanda ("demand"),
    generación total (``total_generation``), energía recibida y enviada
    (``energy_received``, ``energy_sent``) y un déficit calculado como
    ``deficit = demand - total_generation``.  Si se proporcionan
    coordenadas, también las incluye como columnas ``lat`` y ``lon``.

    :param df_region: DataFrame con datos de demanda y generación.  Debe
        contener columnas ``period`` (timestamptz), ``respondent`` (region),
        ``type`` y ``value``.
    :param df_fuel: DataFrame con datos de generación por tipo de combustible.
        Debe contener columnas ``period``, ``respondent``, ``fueltype`` y
        ``value``.
    :param df_interchange: DataFrame con intercambio entre regiones.  Debe
        contener columnas ``period``, ``fromba``, ``toba`` y ``value``.
    :param coord_map: Diccionario opcional con claves = region y valores
        ``{"lat": float, "lon": float}`` para incluir latitud y
        longitud en el resultado.
    :return: DataFrame agregado con un registro por minuto y región.
    """
    from pandas.api.types import is_datetime64_any_dtype
    
    # Copias para no modificar los originales
    df_reg = df_region.copy()
    df_ful = df_fuel.copy()
    df_int = df_interchange.copy()

    # -- Normalizar period (manejo de zona horaria) --
    for d in [df_reg, df_ful, df_int]:
        if not is_datetime64_any_dtype(d['period']):
            d['period'] = pd.to_datetime(d['period'], errors='coerce')
        # Si es datetime con tz, eliminar la zona horaria
        if hasattr(d['period'].dt, 'tz') and d['period'].dt.tz is not None:
            d['period'] = d['period'].dt.tz_localize(None)

    # Agregación de df_region: separar demanda y generación neta (NG) o total
    # Sumamos por periodo y región para cada tipo
    reg_group = df_reg.groupby(["period", "respondent", "type"])["value"].sum().reset_index()
    # Pivot para obtener columnas demand y generation
    # definimos generation como tipo 'NG' (Net Generation) y demand como 'D'
    reg_pivot = reg_group.pivot_table(
        index=["period", "respondent"],
        columns="type",
        values="value",
        aggfunc="sum"
    ).reset_index()
    # renombrar columnas: tipo 'D' -> demand, 'NG' -> total_generation, otros sin cambio
    if 'D' in reg_pivot.columns:
        reg_pivot = reg_pivot.rename(columns={'D': 'demand'})
    if 'NG' in reg_pivot.columns:
        reg_pivot = reg_pivot.rename(columns={'NG': 'total_generation'})
    # otras columnas se dejan como están
    # Agregación de df_fuel: pivot por fueltype
    fuel_group = df_ful.groupby(["period", "respondent", "fueltype"])["value"].sum().reset_index()
    fuel_pivot = fuel_group.pivot_table(
        index=["period", "respondent"],
        columns="fueltype",
        values="value",
        aggfunc="sum"
    )
    fuel_pivot = fuel_pivot.reset_index()
    # Agregación de df_interchange: energía recibida y enviada
    # energía enviada: sum(value) para filas donde respondent = fromba
    received = df_int.groupby(["period", "fromba"])["value"].sum().reset_index()
    received = received.rename(columns={"fromba": "respondent", "value": "energy_received"})
    # energía recibida: sum(value) para filas donde respondent = toba
    sent = df_int.groupby(["period", "toba"])["value"].sum().reset_index()
    sent = sent.rename(columns={"toba": "respondent", "value": "energy_sent"})
    # combinar todas
    # merge reg_pivot y fuel_pivot
    agg = pd.merge(reg_pivot, fuel_pivot, on=["period", "respondent"], how="outer")
    # merge sent y received
    agg = pd.merge(agg, sent, on=["period", "respondent"], how="left")
    agg = pd.merge(agg, received, on=["period", "respondent"], how="left")
    # rellenar NaN con 0 para columnas numéricas
    num_cols = agg.select_dtypes(include=[np.number]).columns
    agg[num_cols] = agg[num_cols].fillna(0)
    # calcular total_generation si no existe sumando fueltypes (excluye demand)
    if 'total_generation' not in agg.columns:
        # sumamos todas las columnas que no son demand, energy_sent, energy_received
        value_cols = [c for c in agg.columns if c not in ["period", "respondent", "demand", "energy_sent", "energy_received"]]
        agg['total_generation'] = agg[value_cols].sum(axis=1)
    # calcular deficit
    agg['deficit'] = agg['demand'] - agg['total_generation']
    # incorporar coordenadas si se proporcionan
    if coord_map:
        # Extraer lat/lon desde center (tupla) y convertir boundary a JSON string
        agg['lat'] = agg['respondent'].map(lambda x: coord_map.get(x, {}).get('center', (0.0, 0.0))[0])
        agg['lon'] = agg['respondent'].map(lambda x: coord_map.get(x, {}).get('center', (0.0, 0.0))[1])
        agg['boundary'] = agg['respondent'].map(lambda x: str(coord_map.get(x, {}).get('boundary', [])))
        agg['region_name'] = agg['respondent'].map(lambda x: coord_map.get(x, {}).get('name', ''))
    agg = agg.sort_values(["period", "respondent"]).reset_index(drop=True)
    return agg


def detect_anomalies(
    df: pd.DataFrame,
    contamination: float = 0.02,
    feature_col: str = "deficit"
) -> pd.DataFrame:
    """Detecta anomalías utilizando Isolation Forest.

    Entrena un modelo de aislamiento sobre la columna ``feature_col`` para
    identificar registros inusuales.  Añade dos columnas: ``anomaly_score``
    con el valor devuelto por ``decision_function`` (puntajes negativos
    indican anomalías más fuertes), y ``is_anomaly`` (booleano).  El
    parámetro ``contamination`` controla la fracción de anomalías esperada.

    :param df: DataFrame que contiene la columna con la característica.
    :param contamination: Fracción de anomalías (0 < contamination < 0.5).
    :param feature_col: Nombre de la columna numérica a analizar
        (por defecto ``deficit``).
    :return: Nuevo DataFrame con columnas de anomalía.
    """
    if df.empty or feature_col not in df.columns:
        out = df.copy()
        out['anomaly_score'] = np.nan
        out['is_anomaly'] = False
        return out
    # convertir a numérico
    feature = pd.to_numeric(df[feature_col], errors='coerce').values.reshape(-1, 1)
    # manejar valores NaN reemplazando con 0
    nan_mask = np.isnan(feature)
    feature[nan_mask] = 0
    # ajustar IsolationForest
    try:
        clf = IsolationForest(contamination=contamination, random_state=42)
        clf.fit(feature)
        scores = clf.decision_function(feature)
        is_anom = clf.predict(feature) == -1
    except Exception:
        # si falla, devolver columnas vacías
        scores = np.zeros(len(df))
        is_anom = np.array([False] * len(df))
    out = df.copy()
    out['anomaly_score'] = scores
    out['is_anomaly'] = is_anom
    return out


def top_deficit_regions(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Devuelve el top ``n`` de regiones con mayor déficit acumulado.

    El déficit se define como la suma de la columna ``deficit`` por región.
    :param df: DataFrame con una columna ``respondent`` y ``deficit``.
    :param n: Número de regiones a devolver.
    :return: DataFrame con columnas ``respondent`` y ``total_deficit`` ordenado
        de mayor a menor déficit.
    """
    if df.empty or 'respondent' not in df.columns or 'deficit' not in df.columns:
        return pd.DataFrame(columns=['respondent', 'total_deficit'])
    agg = (
        df.groupby('respondent')['deficit']
        .sum()
        .reset_index(name='total_deficit')
        .sort_values('total_deficit', ascending=False)
        .head(n)
    )
    return agg.reset_index(drop=True)


__all__ = [
    'remove_high_missing',
    'impute_missing',
    'identify_outliers',
    'aggregate_region_data',
    'detect_anomalies',
    'top_deficit_regions',
]