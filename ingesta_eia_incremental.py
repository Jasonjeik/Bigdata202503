import os
import math
import datetime as dt
import pandas as pd
import requests
import sqlalchemy
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool
import time
import sys
from time import sleep

# =========================
# CONFIGURACIÓN CON TU BD
# =========================
API_KEY = "YOAXAO8j6vBYbOPycDJH0yCfwWgzpK94LQLaZ1hT"
BASE_URL = "https://api.eia.gov/v2/electricity/rto/"
PAGE = 5000

DB_USER = "prj1_admin"
DB_PASS = "Bigdataproyecto1"
DB_HOST = "bigdataproyecto1.postgres.database.azure.com"
DB_PORT = "5432"
DB_NAME = "proyecto1"
SCHEMA  = "eia"

engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
    connect_args={
        "sslmode": "require",
        "options": f"-csearch_path={SCHEMA}",
        "connect_timeout": 30,
        "keepalives": 1,
        "keepalives_idle": 30,
        "keepalives_interval": 10,
        "keepalives_count": 5
    },
    pool_pre_ping=True,  # Verifica conexiones antes de usarlas
    pool_recycle=3600,   # Recicla conexiones cada hora
    pool_size=5,
    max_overflow=10
)

# =========================
# AUXILIARES
# =========================
def test_connection(max_retries=5, delay=5):
    """Valida conectividad con PostgreSQL con reintentos."""
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[Intento {attempt}/{max_retries}] Probando conexión a {DB_HOST}...")
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1;")).scalar()
                if result == 1:
                    print(f"[OK] Conexion exitosa a PostgreSQL")
                    return True
        except Exception as e:
            print(f"[ERROR] Error en intento {attempt}: {e}")
            if attempt < max_retries:
                print(f"[WAIT] Esperando {delay} segundos antes de reintentar...")
                sleep(delay)
            else:
                print(f"\n[WARN] No se pudo conectar despues de {max_retries} intentos.")
                print("\nVerifica:")
                print("  1. Servidor PostgreSQL está en ejecución (Azure Portal)")
                print("  2. Tu IP está permitida en Firewall de Azure")
                print("  3. Credenciales son correctas")
                print(f"  4. Host: {DB_HOST}")
                print(f"  5. Database: {DB_NAME}")
                return False
    return False

def utc_hour_str_now():
    # 'YYYY-MM-DDTHH' en UTC (zona horaria explícita)
    now = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0)
    return now.strftime("%Y-%m-%dT%H")

def add_hour(eia_period_str):
    """eia_period_str: 'YYYY-MM-DDTHH' o datetime -> +1 hora como string."""
    # Si ya es datetime, convertir a string primero
    if isinstance(eia_period_str, dt.datetime):
        eia_period_str = eia_period_str.strftime("%Y-%m-%dT%H")
    
    t = dt.datetime.strptime(eia_period_str, "%Y-%m-%dT%H")
    t2 = t + dt.timedelta(hours=1)
    return t2.strftime("%Y-%m-%dT%H")

def get_last_period(table_name: str, max_retries=3) -> str | None:
    """Devuelve el max(period) como string 'YYYY-MM-DDTHH', o None si la tabla está vacía."""
    q = text(f'SELECT MAX(period) FROM {SCHEMA}."{table_name}";')
    
    for attempt in range(1, max_retries + 1):
        try:
            with engine.begin() as conn:
                r = conn.execute(q).scalar()
                
                # Si es None, devolver None
                if r is None:
                    return None
                
                # Si es datetime, convertir a string
                if isinstance(r, dt.datetime):
                    return r.strftime("%Y-%m-%dT%H")
                
                # Si ya es string, devolverlo tal cual
                return str(r)
                
        except Exception as e:
            print(f"[WARN] Error al obtener last_period (intento {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                sleep(2)
            else:
                print(f"[ERROR] No se pudo obtener last_period para {table_name}")
                return None

def eia_paged_request(endpoint: str, base_params: dict):
    """Generador que pagina contra la EIA devolviendo dataframes consecutivos."""
    offset = 0
    while True:
        params = dict(base_params)
        params["offset"] = offset
        params["length"] = PAGE

        resp = requests.get(BASE_URL + endpoint, params=params, timeout=180)
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")

        data = resp.json().get("response", {}).get("data", [])
        if not data:
            break

        yield pd.DataFrame(data)
        offset += PAGE

def upsert_from_df(df: pd.DataFrame, table: str, key_cols: list[str]):
    """Inserta/actualiza df en tabla con ON CONFLICT (key_cols)."""
    if df.empty:
        return

    # Hacer una copia para no modificar el original
    df = df.copy()
    
    # Convertir columna period a timestamp si existe
    if 'period' in df.columns:
        df['period'] = pd.to_datetime(df['period'], errors='coerce')
    
    # Convertir columna value a numeric si existe
    if 'value' in df.columns:
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
    
    # Cargar a staging
    stg = f"_stg_{table}"
    with engine.begin() as conn:
        conn.execute(text(f'DROP TABLE IF EXISTS {SCHEMA}."{stg}";'))
    df.to_sql(stg, engine, schema=SCHEMA, if_exists="replace", index=False)

    # Construir UPSERT dinámico con casts necesarios
    all_cols = list(df.columns)
    # Citar columnas (maneja guiones)
    def q(c): return f'"{c}"'
    
    # SELECT con casts para tipos específicos
    select_cols = []
    for c in all_cols:
        if c == 'period':
            select_cols.append(f'{q(c)}::timestamptz')
        elif c == 'value':
            select_cols.append(f'{q(c)}::numeric')
        else:
            select_cols.append(q(c))
    select_csv = ", ".join(select_cols)
    
    cols_csv = ", ".join(q(c) for c in all_cols)
    excl_set = ", ".join(f'{q(c)} = EXCLUDED.{q(c)}' for c in all_cols if c not in key_cols)
    keys_csv = ", ".join(q(k) for k in key_cols)

    sql = f'''
        INSERT INTO {SCHEMA}."{table}" ({cols_csv})
        SELECT {select_csv} FROM {SCHEMA}."{stg}"
        ON CONFLICT ({keys_csv}) DO UPDATE
        SET {excl_set};
        DROP TABLE IF EXISTS {SCHEMA}."{stg}";
    '''
    with engine.begin() as conn:
        conn.execute(text(sql))

def ingest_region(historical_mode: bool = False):
    table = "rto_region_data"
    keys = ["period", "respondent", "type"]

    end_dt = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0)
    last_p = None if historical_mode else get_last_period(table)

    if historical_mode:
        start = "2024-01-01T00"
        end = utc_hour_str_now()
        print(f"📥 MODO HISTÓRICO: Cargando datos desde {start} hasta {end}")
        base_params = {
            "api_key": API_KEY,
            "frequency": "hourly",
            "data[0]": "value",
            "facets[type][]": ["D", "NG", "TI"],
            "start": start,
            "end": end,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc"
        }
    else:
        # INCREMENTAL: NO usar start/end (bug en API), traer recientes con DESC sort
        print(f"[INCR] INCREMENTAL region last_period={last_p} -> traer datos recientes sin filtro start/end")
        base_params = {
            "api_key": API_KEY,
            "frequency": "hourly",
            "data[0]": "value",
            "facets[type][]": ["D", "NG", "TI"],
            "sort[0][column]": "period",
            "sort[0][direction]": "desc"  # DESC para traer más recientes primero
        }

    print(f"Region: modo={'historico' if historical_mode else 'incremental'}")
    total_rows = 0
    page_num = 0
    # En modo incremental limitar a primeras 2 páginas (suficiente para últimas ~24h con DESC sort)
    # El dataset de interchange tiene muchos registros por hora (pares BA),
    # por lo que 2 páginas pueden no alcanzar a cubrir la hora más reciente.
    # Ampliamos a 8 páginas (~40k filas) en modo incremental para capturar
    # varias horas recientes incluso cuando el API ordena por múltiples columnas.
    max_pages = 999 if historical_mode else 8
    for df in eia_paged_request("region-data/data/", base_params):
        page_num += 1
        if not df.empty:
            per_min = df['period'].min() if 'period' in df.columns else None
            per_max = df['period'].max() if 'period' in df.columns else None
            print(f"  página {page_num}: periods {per_min} → {per_max} ({len(df)} filas)")
        total_rows += len(df)
        upsert_from_df(df, table, keys)
        if page_num >= max_pages:
            break
    
    # Verificar último period cargado en BD
    final_max = get_last_period(table)
    print(f"Region OK: +{total_rows} filas ingresadas. MAX period en BD ahora: {final_max}")
    
    # Advertir si hay gap entre BD y hora actual
    if final_max:
        final_max_dt = dt.datetime.strptime(final_max, "%Y-%m-%dT%H").replace(tzinfo=dt.timezone.utc) if isinstance(final_max, str) else final_max
        if final_max_dt.tzinfo is None:
            final_max_dt = final_max_dt.replace(tzinfo=dt.timezone.utc)
        gap_hours = int((end_dt - final_max_dt).total_seconds() / 3600)
        if gap_hours > 1:
            print(f"[GAP] La API aun no publico las ultimas {gap_hours} horas (desde {final_max} hasta {end_dt.strftime('%Y-%m-%dT%H')}). Esto es normal; se cargaran en proximos ciclos.")


def ingest_fueltype(historical_mode: bool = False):
    table = "rto_fueltype_data"
    keys = ["period", "respondent", "fueltype"]

    end_dt = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0)
    last_p = None if historical_mode else get_last_period(table)

    if historical_mode:
        start = "2024-01-01T00"
        end = utc_hour_str_now()
        print(f"📥 MODO HISTÓRICO: Cargando datos desde {start} hasta {end}")
        base_params = {
            "api_key": API_KEY,
            "frequency": "hourly",
            "data[0]": "value",
            "start": start,
            "end": end,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc"
        }
    else:
        print(f"[INCR] INCREMENTAL fueltype last_period={last_p} -> traer recientes sin start/end")
        base_params = {
            "api_key": API_KEY,
            "frequency": "hourly",
            "data[0]": "value",
            "sort[0][column]": "period",
            "sort[0][direction]": "desc"
        }

    print(f"FuelType: modo={'historico' if historical_mode else 'incremental'}")
    total_rows = 0
    page_num = 0
    max_pages = 999 if historical_mode else 2
    for df in eia_paged_request("fuel-type-data/data/", base_params):
        page_num += 1
        if not df.empty:
            per_min = df['period'].min() if 'period' in df.columns else None
            per_max = df['period'].max() if 'period' in df.columns else None
            print(f"  pagina {page_num}: periods {per_min} -> {per_max} ({len(df)} filas)")
        total_rows += len(df)
        upsert_from_df(df, table, keys)
        if page_num >= max_pages:
            break
    
    final_max = get_last_period(table)
    print(f"FuelType OK: +{total_rows} filas ingresadas. MAX period en BD ahora: {final_max}")
    
    if final_max:
        final_max_dt = dt.datetime.strptime(final_max, "%Y-%m-%dT%H").replace(tzinfo=dt.timezone.utc) if isinstance(final_max, str) else final_max
        if final_max_dt.tzinfo is None:
            final_max_dt = final_max_dt.replace(tzinfo=dt.timezone.utc)
        gap_hours = int((end_dt - final_max_dt).total_seconds() / 3600)
        if gap_hours > 1:
            print(f"[GAP] fueltype: API no publico ultimas {gap_hours}h (desde {final_max} hasta {end_dt.strftime('%Y-%m-%dT%H')}).")

def ingest_interchange(historical_mode: bool = False):
    table = "rto_interchange_data"
    keys = ["period", "fromba", "toba"]

    end_dt = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0)
    last_p = None if historical_mode else get_last_period(table)

    if historical_mode:
        start = "2022-01-01T00"
        end = utc_hour_str_now()
        print(f"[HIST] MODO HISTORICO: Cargando datos desde {start} hasta {end}")
        base_params = {
            "api_key": API_KEY,
            "frequency": "hourly",
            "data[0]": "value",
            "start": start,
            "end": end,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc"
        }
    else:
        print(f"[INCR] INCREMENTAL interchange last_period={last_p} -> traer recientes sin start/end")
        base_params = {
            "api_key": API_KEY,
            "frequency": "hourly",
            "data[0]": "value",
            "sort[0][column]": "period",
            "sort[0][direction]": "desc"
        }

    print(f"Interchange: modo={'historico' if historical_mode else 'incremental'}")
    total_rows = 0
    page_num = 0
    max_pages = 999 if historical_mode else 2
    for df in eia_paged_request("interchange-data/data/", base_params):
        page_num += 1
        if not df.empty:
            per_min = df['period'].min() if 'period' in df.columns else None
            per_max = df['period'].max() if 'period' in df.columns else None
            print(f"  pagina {page_num}: periods {per_min} -> {per_max} ({len(df)} filas)")
        total_rows += len(df)
        upsert_from_df(df, table, keys)
        if page_num >= max_pages:
            break
    
    final_max = get_last_period(table)
    print(f"Interchange OK: +{total_rows} filas ingresadas. MAX period en BD ahora: {final_max}")
    
    if final_max:
        final_max_dt = dt.datetime.strptime(final_max, "%Y-%m-%dT%H").replace(tzinfo=dt.timezone.utc) if isinstance(final_max, str) else final_max
        if final_max_dt.tzinfo is None:
            final_max_dt = final_max_dt.replace(tzinfo=dt.timezone.utc)
        gap_hours = int((end_dt - final_max_dt).total_seconds() / 3600)
        if gap_hours > 1:
            print(f"[GAP] interchange: API no publico ultimas {gap_hours}h (desde {final_max} hasta {end_dt.strftime('%Y-%m-%dT%H')}).")


# --- Bucle programador ---
def run_eda_aggregation():
    """Ejecuta el pipeline EDA y guarda tabla agregada para Streamlit."""
    try:
        # Importar eda_utils
        from eda_utils import (
            remove_high_missing, impute_missing, identify_outliers,
            aggregate_region_data, detect_anomalies, top_deficit_regions
        )
        
        print(f"[{dt.datetime.now()}] Ejecutando EDA y agregación...")
        
        # Cargar últimas 24 horas de datos
        # Usar formato simplificado sin microsegundos
        end_time = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0)
        start_time = end_time - dt.timedelta(hours=24)
        
        # Formatear como 'YYYY-MM-DD HH:00:00' para compatibilidad con timestamptz
        start_str = start_time.strftime("%Y-%m-%d %H:00:00")
        end_str = end_time.strftime("%Y-%m-%d %H:00:00")
        
        print(f"  Rango de datos: {start_str} -> {end_str}")
        
        # Queries con filtro temporal
        q_region = f'''
            SELECT * FROM {SCHEMA}.rto_region_data 
            WHERE period >= '{start_str}'::timestamptz 
            AND period <= '{end_str}'::timestamptz
        '''
        q_fuel = f'''
            SELECT * FROM {SCHEMA}.rto_fueltype_data 
            WHERE period >= '{start_str}'::timestamptz 
            AND period <= '{end_str}'::timestamptz
        '''
        q_inter = f'''
            SELECT * FROM {SCHEMA}.rto_interchange_data 
            WHERE period >= '{start_str}'::timestamptz 
            AND period <= '{end_str}'::timestamptz
        '''
        
        df_region = pd.read_sql(q_region, engine)
        df_fuel = pd.read_sql(q_fuel, engine)
        df_inter = pd.read_sql(q_inter, engine)
        
        print(f"  📊 Registros cargados: region={len(df_region)}, fuel={len(df_fuel)}, interchange={len(df_inter)}")
        
        # Convertir tipos
        for df in [df_region, df_fuel, df_inter]:
            if 'period' in df.columns:
                df['period'] = pd.to_datetime(df['period'], errors='coerce')
            if 'value' in df.columns:
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
        
        # Limpiar
        df_region = impute_missing(remove_high_missing(df_region))
        df_fuel = impute_missing(remove_high_missing(df_fuel))
        df_inter = impute_missing(remove_high_missing(df_inter))
        
        # Detectar outliers
        if not df_region.empty and 'value' in df_region.columns:
            df_region = identify_outliers(df_region, 'value')
        if not df_fuel.empty and 'value' in df_fuel.columns:
            df_fuel = identify_outliers(df_fuel, 'value')
        if not df_inter.empty and 'value' in df_inter.columns:
            df_inter = identify_outliers(df_inter, 'value')
        
        # Diccionario coordenadas (pequeño subset)
        # Diccionario con geocercas (boundaries) y centros
        coord_map = {
            "YAD": {"name": "Alcoa Power Generating, Inc. - Yadkin Division", "center": (35.7853, -81.3748), "boundary": [[[ -81.5, 35.6 ], [ -81.2, 35.6 ], [ -81.2, 35.9 ], [ -81.5, 35.9 ], [ -81.5, 35.6 ]]]},
            "AZPS": {"name": "Arizona Public Service Company", "center": (33.4484, -112.0740), "boundary": [[[ -114.8, 37.0 ], [ -109.0, 37.0 ], [ -109.0, 31.3 ], [ -114.8, 31.3 ], [ -114.8, 37.0 ]]]},
            "DEAA": {"name": "Arlington Valley, LLC", "center": (32.7700, -113.0600), "boundary": [[[ -113.2, 32.6 ], [ -112.9, 32.6 ], [ -112.9, 32.9 ], [ -113.2, 32.9 ], [ -113.2, 32.6 ]]]},
            "AECI": {"name": "Associated Electric Cooperative, Inc.", "center": (37.2089, -93.2923), "boundary": [[[ -95.8, 40.6 ], [ -89.1, 40.6 ], [ -89.1, 36.0 ], [ -95.8, 36.0 ], [ -95.8, 40.6 ]]]},
            "AVRN": {"name": "Avangrid Renewables, LLC", "center": (45.5231, -122.6765), "boundary": [[[ -124.0, 49.0 ], [ -109.0, 49.0 ], [ -109.0, 31.0 ], [ -124.0, 31.0 ], [ -124.0, 49.0 ]]]},
            "AVA": {"name": "Avista Corporation", "center": (47.6588, -117.4260), "boundary": [[[ -119.0, 49.0 ], [ -116.0, 49.0 ], [ -116.0, 46.5 ], [ -119.0, 46.5 ], [ -119.0, 49.0 ]]]},
            "BANC": {"name": "Balancing Authority of Northern California", "center": (38.5816, -121.4944), "boundary": [[[ -124.4, 42.0 ], [ -120.0, 42.0 ], [ -120.0, 38.0 ], [ -124.4, 38.0 ], [ -124.4, 42.0 ]]]},
            "BPAT": {"name": "Bonneville Power Administration", "center": (45.6387, -122.7574), "boundary": [[[ -124.8, 49.0 ], [ -110.0, 49.0 ], [ -110.0, 42.0 ], [ -124.8, 42.0 ], [ -124.8, 49.0 ]]]},
            "CAL": {"name": "California", "center": (36.7783, -119.4179), "boundary": [[[ -124.4, 42.0 ], [ -114.1, 42.0 ], [ -114.1, 32.5 ], [ -124.4, 32.5 ], [ -124.4, 42.0 ]]]},
            "CISO": {"name": "California Independent System Operator", "center": (38.6686, -121.1023), "boundary": [[[ -124.0, 41.0 ], [ -114.5, 41.0 ], [ -114.5, 33.0 ], [ -124.0, 33.0 ], [ -124.0, 41.0 ]]]},
            "CAR": {"name": "Carolinas", "center": (35.1958, -80.8930), "boundary": [[[ -84.3, 36.6 ], [ -75.4, 36.6 ], [ -75.4, 33.8 ], [ -84.3, 33.8 ], [ -84.3, 36.6 ]]]},
            "CENT": {"name": "Central", "center": (39.7392, -104.9903), "boundary": [[[ -109.0, 49.0 ], [ -90.0, 49.0 ], [ -90.0, 36.5 ], [ -109.0, 36.5 ], [ -109.0, 49.0 ]]]},
            "HST": {"name": "City of Homestead", "center": (25.4687, -80.4776), "boundary": [[[ -80.6, 25.3 ], [ -80.3, 25.3 ], [ -80.3, 25.6 ], [ -80.6, 25.6 ], [ -80.6, 25.3 ]]]},
            "TPWR": {"name": "City of Tacoma, Department of Public Utilities, Light Division", "center": (47.2529, -122.4443), "boundary": [[[ -122.6, 47.0 ], [ -122.2, 47.0 ], [ -122.2, 47.4 ], [ -122.6, 47.4 ], [ -122.6, 47.0 ]]]},
            "TAL": {"name": "City of Tallahassee", "center": (30.4383, -84.2807), "boundary": [[[ -84.5, 30.3 ], [ -84.1, 30.3 ], [ -84.1, 30.6 ], [ -84.5, 30.6 ], [ -84.5, 30.3 ]]]},
            "SCEG": {"name": "Dominion Energy South Carolina, Inc.", "center": (33.9486, -81.9624), "boundary": [[[ -83.4, 35.2 ], [ -78.5, 35.2 ], [ -78.5, 32.0 ], [ -83.4, 32.0 ], [ -83.4, 35.2 ]]]},
            "DUK": {"name": "Duke Energy Carolinas", "center": (35.2271, -80.8431), "boundary": [[[ -84.3, 36.6 ], [ -75.4, 36.6 ], [ -75.4, 33.8 ], [ -84.3, 33.8 ], [ -84.3, 36.6 ]]]},
            "FPC": {"name": "Duke Energy Florida, Inc.", "center": (28.5383, -81.3792), "boundary": [[[ -87.6, 31.0 ], [ -79.9, 31.0 ], [ -79.9, 24.5 ], [ -87.6, 24.5 ], [ -87.6, 31.0 ]]]},
            "CPLE": {"name": "Duke Energy Progress East", "center": (35.2271, -80.8431), "boundary": [[[ -80.0, 36.6 ], [ -75.4, 36.6 ], [ -75.4, 33.8 ], [ -80.0, 33.8 ], [ -80.0, 36.6 ]]]},
            "CPLW": {"name": "Duke Energy Progress West", "center": (35.2271, -80.8431), "boundary": [[[ -84.3, 36.6 ], [ -80.0, 36.6 ], [ -80.0, 33.8 ], [ -84.3, 33.8 ], [ -84.3, 36.6 ]]]},
            "FPL": {"name": "Florida Power & Light Co.", "center": (26.8892, -80.1108), "boundary": [[[ -87.6, 30.7 ], [ -79.9, 30.7 ], [ -79.9, 25.0 ], [ -82.0, 25.0 ], [ -87.6, 30.7 ]]]},
            "GRID": {"name": "Gridforce Energy Management, LLC", "center": (45.5231, -122.6765), "boundary": [[[ -107.0, 36.5 ], [ -93.5, 36.5 ], [ -93.5, 25.8 ], [ -107.0, 25.8 ], [ -107.0, 36.5 ]], [[ -124.0, 46.0 ], [ -121.0, 46.0 ], [ -121.0, 45.5 ], [ -124.0, 45.5 ], [ -124.0, 46.0 ]]]},
            "IPCO": {"name": "Idaho Power Company", "center": (43.6150, -116.2023), "boundary": [[[ -117.0, 49.0 ], [ -111.0, 49.0 ], [ -111.0, 42.0 ], [ -117.0, 42.0 ], [ -117.0, 49.0 ]]]},
            "IID": {"name": "Imperial Irrigation District", "center": (33.1130, -115.5711), "boundary": [[[ -118.0, 34.0 ], [ -114.0, 34.0 ], [ -114.0, 32.5 ], [ -118.0, 32.5 ], [ -118.0, 34.0 ]]]},
            "ISNE": {"name": "ISO New England", "center": (42.3656, -71.2606), "boundary": [[[ -73.7, 47.5 ], [ -66.9, 47.5 ], [ -66.9, 41.0 ], [ -73.7, 41.0 ], [ -73.7, 47.5 ]]]},
            "JEA": {"name": "JEA", "center": (30.3240, -81.6557), "boundary": [[[ -82.0, 30.1 ], [ -81.3, 30.1 ], [ -81.3, 30.6 ], [ -82.0, 30.6 ], [ -82.0, 30.1 ]]]},
            "LGEE": {"name": "LG&E and KU Services Company", "center": (38.2527, -85.7585), "boundary": [[[ -89.6, 39.2 ], [ -81.9, 39.2 ], [ -81.9, 36.5 ], [ -89.6, 36.5 ], [ -89.6, 39.2 ]]]},
            "LDWP": {"name": "Los Angeles Department of Water and Power", "center": (34.0522, -118.2437), "boundary": [[[ -119.0, 34.8 ], [ -117.8, 34.8 ], [ -117.8, 33.7 ], [ -119.0, 33.7 ], [ -119.0, 34.8 ]]]},
            "MISO": {"name": "Midcontinent ISO", "center": (39.7635, -86.1576), "boundary": [[[ -104.0, 49.0 ], [ -80.0, 49.0 ], [ -80.0, 29.0 ], [ -104.0, 29.0 ], [ -104.0, 49.0 ]]]},
            "NEVP": {"name": "Nevada Power Company", "center": (36.1699, -115.1398), "boundary": [[[ -120.0, 39.0 ], [ -114.0, 39.0 ], [ -114.0, 35.0 ], [ -120.0, 35.0 ], [ -120.0, 39.0 ]]]},
            "NYIS": {"name": "New York ISO", "center": (42.7334, -73.8863), "boundary": [[[ -79.8, 45.0 ], [ -71.8, 45.0 ], [ -71.8, 40.5 ], [ -79.8, 40.5 ], [ -79.8, 45.0 ]]]},
            "PACE": {"name": "PacifiCorp East", "center": (40.7608, -111.8910), "boundary": [[[ -114.0, 45.0 ], [ -104.0, 45.0 ], [ -104.0, 37.0 ], [ -114.0, 37.0 ], [ -114.0, 45.0 ]]]},
            "PACW": {"name": "PacifiCorp West", "center": (40.7608, -111.8910), "boundary": [[[ -124.0, 49.0 ], [ -117.0, 49.0 ], [ -117.0, 42.0 ], [ -124.0, 42.0 ], [ -124.0, 49.0 ]]]},
            "PJM": {"name": "PJM Interconnection", "center": (40.1194, -75.5253), "boundary": [[[ -90.0, 42.5 ], [ -74.5, 42.5 ], [ -74.5, 36.5 ], [ -90.0, 36.5 ], [ -90.0, 42.5 ]]]},
            "PGE": {"name": "Portland General Electric", "center": (45.5231, -122.6765), "boundary": [[[ -124.0, 46.3 ], [ -121.5, 46.3 ], [ -121.5, 45.5 ], [ -124.0, 45.5 ], [ -124.0, 46.3 ]]]},
            "PSCO": {"name": "Public Service Company of Colorado", "center": (39.7392, -104.9903), "boundary": [[[ -109.0, 41.0 ], [ -102.0, 41.0 ], [ -102.0, 37.0 ], [ -109.0, 37.0 ], [ -109.0, 41.0 ]]]},
            "PNM": {"name": "Public Service Company of New Mexico", "center": (35.0853, -106.6056), "boundary": [[[ -109.0, 37.0 ], [ -103.0, 37.0 ], [ -103.0, 31.3 ], [ -109.0, 31.3 ], [ -109.0, 37.0 ]]]},
            "CHPD": {"name": "PUD Chelan County", "center": (47.5390, -120.5012), "boundary": [[[ -121.0, 48.0 ], [ -120.0, 48.0 ], [ -120.0, 47.3 ], [ -121.0, 47.3 ], [ -121.0, 48.0 ]]]},
            "GCPD": {"name": "PUD Grant County", "center": (47.2330, -119.5412), "boundary": [[[ -120.0, 47.8 ], [ -119.0, 47.8 ], [ -119.0, 46.9 ], [ -120.0, 46.9 ], [ -120.0, 47.8 ]]]},
            "DOPD": {"name": "PUD Douglas County", "center": (47.9700, -119.5400), "boundary": [[[ -120.0, 48.2 ], [ -119.5, 48.2 ], [ -119.5, 47.7 ], [ -120.0, 47.7 ], [ -120.0, 48.2 ]]]},
            "PSEI": {"name": "Puget Sound Energy", "center": (47.6062, -122.3321), "boundary": [[[ -124.0, 49.0 ], [ -120.5, 49.0 ], [ -120.5, 47.0 ], [ -124.0, 47.0 ], [ -124.0, 49.0 ]]]},
            "SRP": {"name": "Salt River Project", "center": (33.4255, -111.9400), "boundary": [[[ -113.0, 34.0 ], [ -111.5, 34.0 ], [ -111.5, 33.0 ], [ -113.0, 33.0 ], [ -113.0, 34.0 ]]]},
            "SCL": {"name": "Seattle City Light", "center": (47.6080, -122.3352), "boundary": [[[ -122.5, 47.8 ], [ -122.0, 47.8 ], [ -122.0, 47.5 ], [ -122.5, 47.5 ], [ -122.5, 47.8 ]]]},
            "SEC": {"name": "Seminole Electric Cooperative", "center": (28.0575, -81.6501), "boundary": [[[ -83.0, 29.0 ], [ -80.0, 29.0 ], [ -80.0, 27.5 ], [ -83.0, 27.5 ], [ -83.0, 29.0 ]]]},
            "SC": {"name": "South Carolina Public Service Authority", "center": (33.0026, -80.0880), "boundary": [[[ -83.4, 35.2 ], [ -78.5, 35.2 ], [ -78.5, 32.0 ], [ -83.4, 32.0 ], [ -83.4, 35.2 ]]]},
            "SEPA": {"name": "Southeastern Power Administration", "center": (33.9770, -83.3770), "boundary": [[[ -90.0, 39.0 ], [ -75.0, 39.0 ], [ -75.0, 30.0 ], [ -90.0, 30.0 ], [ -90.0, 39.0 ]]]},
            "SOCO": {"name": "Southern Company Services", "center": (33.7490, -84.3880), "boundary": [[[ -91.6, 35.0 ], [ -80.7, 35.0 ], [ -80.7, 30.3 ], [ -91.6, 30.3 ], [ -91.6, 35.0 ]]]},
            "SPA": {"name": "Southwestern Power Administration", "center": (36.1627, -94.1700), "boundary": [[[ -103.0, 40.0 ], [ -89.0, 40.0 ], [ -89.0, 29.0 ], [ -103.0, 29.0 ], [ -103.0, 40.0 ]]]},
            "SWPP": {"name": "Southwest Power Pool", "center": (34.7465, -92.2896), "boundary": [[[ -104.0, 49.0 ], [ -89.0, 49.0 ], [ -89.0, 29.0 ], [ -104.0, 29.0 ], [ -104.0, 49.0 ]]]},
            "TEC": {"name": "Tampa Electric Company", "center": (27.9478, -82.4584), "boundary": [[[ -82.8, 28.2 ], [ -82.1, 28.2 ], [ -82.1, 27.7 ], [ -82.8, 27.7 ], [ -82.8, 28.2 ]]]},
            "TVA": {"name": "Tennessee Valley Authority", "center": (35.9606, -83.9207), "boundary": [[[ -90.3, 37.0 ], [ -81.6, 37.0 ], [ -81.6, 33.0 ], [ -90.3, 33.0 ], [ -90.3, 37.0 ]]]},
            "TEPC": {"name": "Tucson Electric Power", "center": (32.2226, -110.9747), "boundary": [[[ -112.0, 32.5 ], [ -110.0, 32.5 ], [ -110.0, 31.3 ], [ -112.0, 31.3 ], [ -112.0, 32.5 ]]]},
            "TIDC": {"name": "Turlock Irrigation District", "center": (37.4977, -120.8466), "boundary": [[[ -121.0, 37.8 ], [ -120.7, 37.8 ], [ -120.7, 37.3 ], [ -121.0, 37.3 ], [ -121.0, 37.8 ]]]},
            "US48": {"name": "United States Lower 48", "center": (39.8283, -98.5795), "boundary": [[[ -125.0, 49.0 ], [ -66.9, 49.0 ], [ -66.9, 25.0 ], [ -125.0, 25.0 ], [ -125.0, 49.0 ]]]},
            "WALC": {"name": "WAPA - Desert Southwest", "center": (33.4455, -112.0678), "boundary": [[[ -120.0, 37.0 ], [ -109.0, 37.0 ], [ -109.0, 31.3 ], [ -120.0, 31.3 ], [ -120.0, 37.0 ]]]},
            "WACM": {"name": "WAPA - Rocky Mountain", "center": (40.6331, -105.1433), "boundary": [[[ -109.0, 45.0 ], [ -96.0, 45.0 ], [ -96.0, 37.0 ], [ -109.0, 37.0 ], [ -109.0, 45.0 ]]]},
            "WAUW": {"name": "WAPA - Upper Great Plains West", "center": (46.8133, -92.1004), "boundary": [[[ -116.0, 49.0 ], [ -96.0, 49.0 ], [ -96.0, 41.0 ], [ -116.0, 41.0 ], [ -116.0, 49.0 ]]]},
            "ERCO": {"name": "ERCOT", "center": (30.2672, -97.7431), "boundary": [[[ -106.6, 36.5 ], [ -93.5, 36.5 ], [ -93.5, 25.8 ], [ -106.6, 25.8 ], [ -106.6, 36.5 ]]]},
            "EPE": {"name": "El Paso Electric", "center": (31.7619, -106.4850), "boundary": [[[ -108.0, 32.0 ], [ -106.0, 32.0 ], [ -106.0, 31.3 ], [ -108.0, 31.3 ], [ -108.0, 32.0 ]]]},
            "FMPP": {"name": "Florida Municipal Power Pool", "center": (30.4383, -84.2807), "boundary": [[[ -87.6, 31.0 ], [ -79.9, 31.0 ], [ -79.9, 24.5 ], [ -87.6, 24.5 ], [ -87.6, 31.0 ]]]},
            "GVL": {"name": "Gainesville Regional Utilities", "center": (29.6516, -82.3248), "boundary": [[[ -82.5, 29.8 ], [ -82.1, 29.8 ], [ -82.1, 29.5 ], [ -82.5, 29.5 ], [ -82.5, 29.8 ]]]},
            "GWA": {"name": "NaturEner Power Watch", "center": (45.6770, -108.5500), "boundary": [[[ -114.0, 49.0 ], [ -104.0, 49.0 ], [ -104.0, 45.0 ], [ -114.0, 45.0 ], [ -114.0, 49.0 ]]]},
            "NWMT": {"name": "NorthWestern Corporation", "center": (45.7833, -108.5007), "boundary": [[[ -116.0, 49.0 ], [ -104.0, 49.0 ], [ -104.0, 44.5 ], [ -116.0, 44.5 ], [ -116.0, 49.0 ]]]},
            "SIKE": {"name": "Sikeston Board of Municipal Utilities", "center": (36.9311, -89.5878), "boundary": [[[ -89.8, 37.0 ], [ -89.4, 37.0 ], [ -89.4, 36.8 ], [ -89.8, 36.8 ], [ -89.8, 37.0 ]]]},
            "TEX": {"name": "Tucson Electric Power","center": (32.2226, -110.9747),"boundary": [[
                [-111.5000, 32.8000],  # Noroeste (cerca de Marana)
                [-110.7000, 32.8000],  # Noreste (cerca de Catalina)
                [-110.5000, 32.4000],  # Este (cerca de Vail)
                [-110.5000, 31.8000],  # Sureste (cerca de Sahuarita)
                [-111.0000, 31.7000],  # Sur (cerca de Green Valley)
                [-111.3000, 31.7000],  # Suroeste (cerca de Three Points)
                [-111.5000, 32.0000],  # Oeste (cerca de Avra Valley)
                [-111.5000, 32.8000]   # Cierre del polígono
            ]]},
            "MIDW": {
                "name": "Midwest",
                "center": (41.8781, -87.6298),
                "boundary": [[[-97.0, 49.0], [-80.0, 49.0], [-80.0, 36.5], [-97.0, 36.5], [-97.0, 49.0]]]
            },
            "MIDA": {
                "name": "Mid-Atlantic",
                "center": (39.9526, -75.1652),
                "boundary": [[[-80.5, 42.5], [-74.0, 42.5], [-74.0, 36.5], [-80.5, 36.5], [-80.5, 42.5]]]
            },
            "TEN": {
                "name": "Tennessee Valley Authority",
                "center": (35.9606, -83.9207),
                "boundary": [
                    [
                        [-90.3100, 37.0000],
                        [-88.0000, 37.5000],
                        [-86.5000, 38.0000],
                        [-84.8000, 37.8000],
                        [-83.0000, 36.6000],
                        [-83.5000, 35.5000],
                        [-85.5000, 34.0000],
                        [-87.5000, 34.0000],
                        [-88.5000, 35.0000],
                        [-90.0000, 35.0000],
                        [-90.3100, 37.0000]
                    ]
                ]
            },
            "TEX": {"name": "Texas","center": (31.9686, -99.9018),"boundary": [[
                [-106.6456, 31.9999],  # Noroeste: El Paso (frontera NM)
                [-103.0000, 36.5000],  # Norte: Panhandle (frontera OK)
                [-100.0000, 36.5000],  # Noreste: Panhandle este
                [-94.0000, 33.0000],   # Este: Texarkana (frontera AR/LA)
                [-93.5000, 29.0000],   # Sureste: Beaumont (Golfo)
                [-97.0000, 26.0000],   # Sur: Brownsville (Río Grande)
                [-100.0000, 25.8372],  # Suroeste: Presidio (Río Grande)
                [-103.0000, 28.0000],  # Oeste: Big Bend
                [-106.5000, 31.7500],  # Frontera con México (Ciudad Juárez)
                [-106.6456, 31.9999]   # Cierre
                ]]
            }
        }
        
        # Agregar
        df_agg = aggregate_region_data(df_region, df_fuel, df_inter, coord_map)
        
        # Renombrar columnas para match con dashboard
        if not df_agg.empty:
            # Renombrar respondent -> region
            if 'respondent' in df_agg.columns:
                df_agg = df_agg.rename(columns={'respondent': 'region'})
            
            # Renombrar demand -> demand_mw, total_generation -> generation_mw
            rename_map = {
                'demand': 'demand_mw',
                'total_generation': 'generation_mw'
            }
            df_agg = df_agg.rename(columns=rename_map)
            
            # Prefijo fuel_ a columnas de combustibles
            # Identificar columnas que no son period, region, demand_mw, generation_mw, deficit, energy_*, lat, lon, anomaly*
            system_cols = ['period', 'region', 'demand_mw', 'generation_mw', 'deficit', 
                          'energy_sent', 'energy_received', 'lat', 'lon', 
                          'anomaly_score', 'is_anomaly', 'is_outlier']
            
            for col in df_agg.columns:
                if col not in system_cols and not col.startswith('fuel_'):
                    # Es una columna de combustible, agregar prefijo
                    df_agg = df_agg.rename(columns={col: f'fuel_{col.lower()}_mw'})
        
        # Detectar anomalías si hay datos
        if not df_agg.empty and 'deficit' in df_agg.columns:
            df_agg = detect_anomalies(df_agg, feature_col='deficit', contamination=0.02)
            # Renombrar is_anomaly -> anomaly (para compatibilidad con dashboard)
            if 'is_anomaly' in df_agg.columns:
                df_agg['anomaly'] = df_agg['is_anomaly'].astype(int)
        
        # Agregar columna updated_at con timestamp actual
        if not df_agg.empty:
            df_agg['updated_at'] = dt.datetime.now(dt.timezone.utc)
        
        # Guardar tabla agregada en PostgreSQL (acumulativo)
        if not df_agg.empty:
            try:
                # Intentar cargar datos existentes
                df_existing = pd.read_sql(
                    f'SELECT * FROM {SCHEMA}.eia_aggregated_realtime',
                    engine
                )
                # Convertir tipos
                if 'period' in df_existing.columns:
                    df_existing['period'] = pd.to_datetime(df_existing['period'], errors='coerce')
                if 'updated_at' in df_existing.columns:
                    df_existing['updated_at'] = pd.to_datetime(df_existing['updated_at'], errors='coerce')
                
                # ACUMULAR: agregar nuevos registros sin eliminar duplicados
                df_combined = pd.concat([df_existing, df_agg], ignore_index=True)
                
                # Filtrar por ventana de 24 horas basado en updated_at
                cutoff_time = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=24)
                df_combined = df_combined[df_combined['updated_at'] >= cutoff_time]
                
                # Guardar tabla acumulativa
                df_combined.to_sql(
                    'eia_aggregated_realtime',
                    engine,
                    schema=SCHEMA,
                    if_exists='replace',
                    index=False
                )
                
                registros_eliminados = len(df_existing) - len(df_combined) + len(df_agg)
                print(f"[{dt.datetime.now()}] [OK] Tabla agregada actualizada:")
                print(f"    • Registros nuevos agregados: {len(df_agg)}")
                print(f"    • Registros en tabla: {len(df_combined)}")
                print(f"    • Registros eliminados (>24h): {max(0, registros_eliminados)}")
                
            except Exception as e:
                # Si la tabla no existe, crear nueva
                print(f"  Tabla no existe, creando nueva...")
                df_agg.to_sql(
                    'eia_aggregated_realtime',
                    engine,
                    schema=SCHEMA,
                    if_exists='replace',
                    index=False
                )
                print(f"[{dt.datetime.now()}] [OK] Tabla agregada creada: {len(df_agg)} filas")
        else:
            print(f"[{dt.datetime.now()}] [WARN] No hay datos para agregar")
            
    except Exception as e:
        print(f"[{dt.datetime.now()}] [ERROR] Error en EDA: {e}")
        import traceback
        traceback.print_exc()


def run_ingest_every_minute():
    """Ejecuta ingestas y EDA cada minuto."""
    print(f"[{dt.datetime.now()}] 🚀 Iniciando ciclo de ingesta y EDA...")
    
    while True:
        try:
            print(f"\n{'='*60}")
            print(f"[{dt.datetime.now()}] Ciclo de actualización iniciado")
            print(f"{'='*60}")
            
            # Paso 1: Ingestar datos
            ingest_region()
            ingest_fueltype()
            ingest_interchange()
            
            # Paso 2: Ejecutar EDA y agregación
            run_eda_aggregation()
            
            print(f"[{dt.datetime.now()}] ✅ Ciclo completado. Esperando 60 segundos...\n")
            
        except Exception as e:
            print(f"[{dt.datetime.now()}] ❌ Error durante ciclo: {e}")
            import traceback
            traceback.print_exc()
        
        # Esperar 60 segundos
        time.sleep(60)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingesta incremental EIA + EDA en tiempo real")
    parser.add_argument("--historical", action="store_true", 
                        help="Cargar datos históricos desde 2022-01-01 (primera ejecución)")
    parser.add_argument("--once", action="store_true",
                        help="Ejecutar solo una vez y salir (sin bucle continuo)")
    args = parser.parse_args()
    
    print("="*70)
    print("  INGESTA INCREMENTAL EIA + EDA EN TIEMPO REAL")
    print("="*70)
    print(f"Schema: {SCHEMA}")
    print(f"Base de datos: {DB_HOST}/{DB_NAME}")
    if args.historical:
        print(f"Modo: HISTÓRICO (desde 2022-01-01)")
    else:
        print(f"Modo: INCREMENTAL (últimas 48 horas)")
    if args.once:
        print(f"Ejecución: UNA VEZ")
    else:
        print(f"Intervalo: 60 segundos (continuo)")
    print("="*70)
    print()
    
    # Validar conexión antes de continuar
    if not test_connection(max_retries=5, delay=5):
        print("\n❌ No se pudo establecer conexión con la base de datos.")
        print("El script se detendrá.\n")
        sys.exit(1)
    
    # Garantizar que exista el schema
    print("\nCreando schema si no existe...")
    try:
        with engine.begin() as conn:
            conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA};"))
        print(f"✅ Schema '{SCHEMA}' verificado\n")
    except Exception as e:
        print(f"❌ Error al crear schema: {e}")
        sys.exit(1)
    
    if not args.once:
        print("Presiona Ctrl+C para detener el proceso\n")
    print("="*70)
    
    try:
        # Ejecutar una primera vez inmediatamente
        print("\n[INICIO] Ejecutando ingesta...\n")
        historical = args.historical
        ingest_region(historical_mode=historical)
        ingest_fueltype(historical_mode=historical)
        ingest_interchange(historical_mode=historical)
        run_eda_aggregation()
        
        if args.once:
            print("\n✅ Ejecución única completada. Saliendo.")
            sys.exit(0)
        
        # Luego entrar al bucle
        run_ingest_every_minute()
        
    except KeyboardInterrupt:
        print("\n\n[DETENIDO] Proceso interrumpido por el usuario")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error fatal: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
