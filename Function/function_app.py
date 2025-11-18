"""Azure Function: Ingest EIA data (region, fueltype, interchange), run EDA + anomaly detection,
aggregate and persist snapshot + rolling 24h views, plus deficit rankings.

Refactor goals implemented:
- Removed duplicated legacy CLI/Spanish blocks.
- Added rolling 24h cumulative maintenance table.
- Unified anomaly columns: anomaly_score, is_anomaly, anomaly.
- Simplified ingestion & aggregation logic for clarity.
"""
from __future__ import annotations

import os
import math
import time
import logging
import json
from datetime import datetime, timedelta, timezone
from typing import Iterator, List

import pandas as pd
import requests
import azure.functions as func
from sqlalchemy import create_engine, text
from sklearn.ensemble import IsolationForest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("eia_function")

REQUIRED_ENV = ["API_KEY", "DATABASE_HOST", "DATABASE_PORT", "DATABASE_NAME", "DATABASE_USER", "DATABASE_PASSWORD"]
for var in REQUIRED_ENV:
    if not os.getenv(var):
        logger.warning("Environment variable %s is not set", var)

# Mapa de regiones con nombre, centro (lat, lon) y geocercas (lista de polígonos)
REGION_INFO = {
    "YAD": {"name": "Alcoa Power Generating, Inc. - Yadkin Division", "center": (35.7853, -81.3748), "boundary": [[[-81.5, 35.6], [-81.2, 35.6], [-81.2, 35.9], [-81.5, 35.9], [-81.5, 35.6]]]},
    "AZPS": {"name": "Arizona Public Service Company", "center": (33.4484, -112.0740), "boundary": [[[-114.8, 37.0], [-109.0, 37.0], [-109.0, 31.3], [-114.8, 31.3], [-114.8, 37.0]]]},
    "DEAA": {"name": "Arlington Valley, LLC", "center": (32.7700, -113.0600), "boundary": [[[-113.2, 32.6], [-112.9, 32.6], [-112.9, 32.9], [-113.2, 32.9], [-113.2, 32.6]]]},
    "AECI": {"name": "Associated Electric Cooperative, Inc.", "center": (37.2089, -93.2923), "boundary": [[[-95.8, 40.6], [-89.1, 40.6], [-89.1, 36.0], [-95.8, 36.0], [-95.8, 40.6]]]},
    "AVRN": {"name": "Avangrid Renewables, LLC", "center": (45.5231, -122.6765), "boundary": [[[-124.0, 49.0], [-109.0, 49.0], [-109.0, 31.0], [-124.0, 31.0], [-124.0, 49.0]]]},
    "AVA": {"name": "Avista Corporation", "center": (47.6588, -117.4260), "boundary": [[[-119.0, 49.0], [-116.0, 49.0], [-116.0, 46.5], [-119.0, 46.5], [-119.0, 49.0]]]},
    "BANC": {"name": "Balancing Authority of Northern California", "center": (38.5816, -121.4944), "boundary": [[[-124.4, 42.0], [-120.0, 42.0], [-120.0, 38.0], [-124.4, 38.0], [-124.4, 42.0]]]},
    "BPAT": {"name": "Bonneville Power Administration", "center": (45.6387, -122.7574), "boundary": [[[-124.8, 49.0], [-110.0, 49.0], [-110.0, 42.0], [-124.8, 42.0], [-124.8, 49.0]]]},
    "CAL": {"name": "California", "center": (36.7783, -119.4179), "boundary": [[[-124.4, 42.0], [-114.1, 42.0], [-114.1, 32.5], [-124.4, 32.5], [-124.4, 42.0]]]},
    "CISO": {"name": "California Independent System Operator", "center": (38.6686, -121.1023), "boundary": [[[-124.0, 41.0], [-114.5, 41.0], [-114.5, 33.0], [-124.0, 33.0], [-124.0, 41.0]]]},
    "CAR": {"name": "Carolinas", "center": (35.1958, -80.8930), "boundary": [[[-84.3, 36.6], [-75.4, 36.6], [-75.4, 33.8], [-84.3, 33.8], [-84.3, 36.6]]]},
    "CENT": {"name": "Central", "center": (39.7392, -104.9903), "boundary": [[[-109.0, 49.0], [-90.0, 49.0], [-90.0, 36.5], [-109.0, 36.5], [-109.0, 49.0]]]},
    "HST": {"name": "City of Homestead", "center": (25.4687, -80.4776), "boundary": [[[-80.6, 25.3], [-80.3, 25.3], [-80.3, 25.6], [-80.6, 25.6], [-80.6, 25.3]]]},
    "TPWR": {"name": "City of Tacoma, Department of Public Utilities, Light Division", "center": (47.2529, -122.4443), "boundary": [[[-122.6, 47.0], [-122.2, 47.0], [-122.2, 47.4], [-122.6, 47.4], [-122.6, 47.0]]]},
    "TAL": {"name": "City of Tallahassee", "center": (30.4383, -84.2807), "boundary": [[[-84.5, 30.3], [-84.1, 30.3], [-84.1, 30.6], [-84.5, 30.6], [-84.5, 30.3]]]},
    "SCEG": {"name": "Dominion Energy South Carolina, Inc.", "center": (33.9486, -81.9624), "boundary": [[[-83.4, 35.2], [-78.5, 35.2], [-78.5, 32.0], [-83.4, 32.0], [-83.4, 35.2]]]},
    "DUK": {"name": "Duke Energy Carolinas", "center": (35.2271, -80.8431), "boundary": [[[-84.3, 36.6], [-75.4, 36.6], [-75.4, 33.8], [-84.3, 33.8], [-84.3, 36.6]]]},
    "FPC": {"name": "Duke Energy Florida, Inc.", "center": (28.5383, -81.3792), "boundary": [[[-87.6, 31.0], [-79.9, 31.0], [-79.9, 24.5], [-87.6, 24.5], [-87.6, 31.0]]]},
    "CPLE": {"name": "Duke Energy Progress East", "center": (35.2271, -80.8431), "boundary": [[[-80.0, 36.6], [-75.4, 36.6], [-75.4, 33.8], [-80.0, 33.8], [-80.0, 36.6]]]},
    "CPLW": {"name": "Duke Energy Progress West", "center": (35.2271, -80.8431), "boundary": [[[-84.3, 36.6], [-80.0, 36.6], [-80.0, 33.8], [-84.3, 33.8], [-84.3, 36.6]]]},
    "FPL": {"name": "Florida Power & Light Co.", "center": (26.8892, -80.1108), "boundary": [[[-87.6, 30.7], [-79.9, 30.7], [-79.9, 25.0], [-82.0, 25.0], [-87.6, 30.7]]]},
    "GRID": {"name": "Gridforce Energy Management, LLC", "center": (45.5231, -122.6765), "boundary": [[[-107.0, 36.5], [-93.5, 36.5], [-93.5, 25.8], [-107.0, 25.8], [-107.0, 36.5]], [[-124.0, 46.0], [-121.0, 46.0], [-121.0, 45.5], [-124.0, 45.5], [-124.0, 46.0]]]},
    "IPCO": {"name": "Idaho Power Company", "center": (43.6150, -116.2023), "boundary": [[[-117.0, 49.0], [-111.0, 49.0], [-111.0, 42.0], [-117.0, 42.0], [-117.0, 49.0]]]},
    "IID": {"name": "Imperial Irrigation District", "center": (33.1130, -115.5711), "boundary": [[[-118.0, 34.0], [-114.0, 34.0], [-114.0, 32.5], [-118.0, 32.5], [-118.0, 34.0]]]},
    "ISNE": {"name": "ISO New England", "center": (42.3656, -71.2606), "boundary": [[[-73.7, 47.5], [-66.9, 47.5], [-66.9, 41.0], [-73.7, 41.0], [-73.7, 47.5]]]},
    "JEA": {"name": "JEA", "center": (30.3240, -81.6557), "boundary": [[[-82.0, 30.1], [-81.3, 30.1], [-81.3, 30.6], [-82.0, 30.6], [-82.0, 30.1]]]},
    "LGEE": {"name": "LG&E and KU Services Company", "center": (38.2527, -85.7585), "boundary": [[[-89.6, 39.2], [-81.9, 39.2], [-81.9, 36.5], [-89.6, 36.5], [-89.6, 39.2]]]},
    "LDWP": {"name": "Los Angeles Department of Water and Power", "center": (34.0522, -118.2437), "boundary": [[[-119.0, 34.8], [-117.8, 34.8], [-117.8, 33.7], [-119.0, 33.7], [-119.0, 34.8]]]},
    "MISO": {"name": "Midcontinent ISO", "center": (39.7635, -86.1576), "boundary": [[[-104.0, 49.0], [-80.0, 49.0], [-80.0, 29.0], [-104.0, 29.0], [-104.0, 49.0]]]},
    "NEVP": {"name": "Nevada Power Company", "center": (36.1699, -115.1398), "boundary": [[[-120.0, 39.0], [-114.0, 39.0], [-114.0, 35.0], [-120.0, 35.0], [-120.0, 39.0]]]},
    "NYIS": {"name": "New York ISO", "center": (42.7334, -73.8863), "boundary": [[[-79.8, 45.0], [-71.8, 45.0], [-71.8, 40.5], [-79.8, 40.5], [-79.8, 45.0]]]},
    "PACE": {"name": "PacifiCorp East", "center": (40.7608, -111.8910), "boundary": [[[-114.0, 45.0], [-104.0, 45.0], [-104.0, 37.0], [-114.0, 37.0], [-114.0, 45.0]]]},
    "PACW": {"name": "PacifiCorp West", "center": (40.7608, -111.8910), "boundary": [[[-124.0, 49.0], [-117.0, 49.0], [-117.0, 42.0], [-124.0, 42.0], [-124.0, 49.0]]]},
    "PJM": {"name": "PJM Interconnection", "center": (40.1194, -75.5253), "boundary": [[[-90.0, 42.5], [-74.5, 42.5], [-74.5, 36.5], [-90.0, 36.5], [-90.0, 42.5]]]},
    "PGE": {"name": "Portland General Electric", "center": (45.5231, -122.6765), "boundary": [[[-124.0, 46.3], [-121.5, 46.3], [-121.5, 45.5], [-124.0, 45.5], [-124.0, 46.3]]]},
    "PSCO": {"name": "Public Service Company of Colorado", "center": (39.7392, -104.9903), "boundary": [[[-109.0, 41.0], [-102.0, 41.0], [-102.0, 37.0], [-109.0, 37.0], [-109.0, 41.0]]]},
    "PNM": {"name": "Public Service Company of New Mexico", "center": (35.0853, -106.6056), "boundary": [[[-109.0, 37.0], [-103.0, 37.0], [-103.0, 31.3], [-109.0, 31.3], [-109.0, 37.0]]]},
    "CHPD": {"name": "PUD Chelan County", "center": (47.5390, -120.5012), "boundary": [[[-121.0, 48.0], [-120.0, 48.0], [-120.0, 47.3], [-121.0, 47.3], [-121.0, 48.0]]]},
    "GCPD": {"name": "PUD Grant County", "center": (47.2330, -119.5412), "boundary": [[[-120.0, 47.8], [-119.0, 47.8], [-119.0, 46.9], [-120.0, 46.9], [-120.0, 47.8]]]},
    "DOPD": {"name": "PUD Douglas County", "center": (47.9700, -119.5400), "boundary": [[[-120.0, 48.2], [-119.5, 48.2], [-119.5, 47.7], [-120.0, 47.7], [-120.0, 48.2]]]},
    "PSEI": {"name": "Puget Sound Energy", "center": (47.6062, -122.3321), "boundary": [[[-124.0, 49.0], [-120.5, 49.0], [-120.5, 47.0], [-124.0, 47.0], [-124.0, 49.0]]]},
    "SRP": {"name": "Salt River Project", "center": (33.4255, -111.9400), "boundary": [[[-113.0, 34.0], [-111.5, 34.0], [-111.5, 33.0], [-113.0, 33.0], [-113.0, 34.0]]]},
    "SCL": {"name": "Seattle City Light", "center": (47.6080, -122.3352), "boundary": [[[-122.5, 47.8], [-122.0, 47.8], [-122.0, 47.5], [-122.5, 47.5], [-122.5, 47.8]]]},
    "SEC": {"name": "Seminole Electric Cooperative", "center": (28.0575, -81.6501), "boundary": [[[-83.0, 29.0], [-80.0, 29.0], [-80.0, 27.5], [-83.0, 27.5], [-83.0, 29.0]]]},
    "SC": {"name": "South Carolina Public Service Authority", "center": (33.0026, -80.0880), "boundary": [[[-83.4, 35.2], [-78.5, 35.2], [-78.5, 32.0], [-83.4, 32.0], [-83.4, 35.2]]]},
    "SEPA": {"name": "Southeastern Power Administration", "center": (33.9770, -83.3770), "boundary": [[[-90.0, 39.0], [-75.0, 39.0], [-75.0, 30.0], [-90.0, 30.0], [-90.0, 39.0]]]},
    "SOCO": {"name": "Southern Company Services", "center": (33.7490, -84.3880), "boundary": [[[-91.6, 35.0], [-80.7, 35.0], [-80.7, 30.3], [-91.6, 30.3], [-91.6, 35.0]]]},
    "SPA": {"name": "Southwestern Power Administration", "center": (36.1627, -94.1700), "boundary": [[[-103.0, 40.0], [-89.0, 40.0], [-89.0, 29.0], [-103.0, 29.0], [-103.0, 40.0]]]},
    "SWPP": {"name": "Southwest Power Pool", "center": (34.7465, -92.2896), "boundary": [[[-104.0, 49.0], [-89.0, 49.0], [-89.0, 29.0], [-104.0, 29.0], [-104.0, 49.0]]]},
    "TEC": {"name": "Tampa Electric Company", "center": (27.9478, -82.4584), "boundary": [[[-82.8, 28.2], [-82.1, 28.2], [-82.1, 27.7], [-82.8, 27.7], [-82.8, 28.2]]]},
    "TVA": {"name": "Tennessee Valley Authority", "center": (35.9606, -83.9207), "boundary": [[[-90.3, 37.0], [-81.6, 37.0], [-81.6, 33.0], [-90.3, 33.0], [-90.3, 37.0]]]},
    "TEPC": {"name": "Tucson Electric Power", "center": (32.2226, -110.9747), "boundary": [[[-112.0, 32.5], [-110.0, 32.5], [-110.0, 31.3], [-112.0, 31.3], [-112.0, 32.5]]]},
    "TIDC": {"name": "Turlock Irrigation District", "center": (37.4977, -120.8466), "boundary": [[[-121.0, 37.8], [-120.7, 37.8], [-120.7, 37.3], [-121.0, 37.3], [-121.0, 37.8]]]},
    "US48": {"name": "United States Lower 48", "center": (39.8283, -98.5795), "boundary": [[[-125.0, 49.0], [-66.9, 49.0], [-66.9, 25.0], [-125.0, 25.0], [-125.0, 49.0]]]},
    "WALC": {"name": "WAPA - Desert Southwest", "center": (33.4455, -112.0678), "boundary": [[[-120.0, 37.0], [-109.0, 37.0], [-109.0, 31.3], [-120.0, 31.3], [-120.0, 37.0]]]},
    "WACM": {"name": "WAPA - Rocky Mountain", "center": (40.6331, -105.1433), "boundary": [[[-109.0, 45.0], [-96.0, 45.0], [-96.0, 37.0], [-109.0, 37.0], [-109.0, 45.0]]]},
    "WAUW": {"name": "WAPA - Upper Great Plains West", "center": (46.8133, -92.1004), "boundary": [[[-116.0, 49.0], [-96.0, 49.0], [-96.0, 41.0], [-116.0, 41.0], [-116.0, 49.0]]]},
    "ERCO": {"name": "ERCOT", "center": (30.2672, -97.7431), "boundary": [[[-106.6, 36.5], [-93.5, 36.5], [-93.5, 25.8], [-106.6, 25.8], [-106.6, 36.5]]]},
    "EPE": {"name": "El Paso Electric", "center": (31.7619, -106.4850), "boundary": [[[-108.0, 32.0], [-106.0, 32.0], [-106.0, 31.3], [-108.0, 31.3], [-108.0, 32.0]]]},
    "FMPP": {"name": "Florida Municipal Power Pool", "center": (30.4383, -84.2807), "boundary": [[[-87.6, 31.0], [-79.9, 31.0], [-79.9, 24.5], [-87.6, 24.5], [-87.6, 31.0]]]},
    "GVL": {"name": "Gainesville Regional Utilities", "center": (29.6516, -82.3248), "boundary": [[[-82.5, 29.8], [-82.1, 29.8], [-82.1, 29.5], [-82.5, 29.5], [-82.5, 29.8]]]},
    "GWA": {"name": "NaturEner Power Watch", "center": (45.6770, -108.5500), "boundary": [[[-114.0, 49.0], [-104.0, 49.0], [-104.0, 45.0], [-114.0, 45.0], [-114.0, 49.0]]]},
    "NWMT": {"name": "NorthWestern Corporation", "center": (45.7833, -108.5007), "boundary": [[[-116.0, 49.0], [-104.0, 49.0], [-104.0, 44.5], [-116.0, 44.5], [-116.0, 49.0]]]},
    "SIKE": {"name": "Sikeston Board of Municipal Utilities", "center": (36.9311, -89.5878), "boundary": [[[-89.8, 37.0], [-89.4, 37.0], [-89.4, 36.8], [-89.8, 36.8], [-89.8, 37.0]]]},
    "TEX": {"name": "Texas", "center": (31.9686, -99.9018), "boundary": [[[-106.6456, 31.9999], [-103.0000, 36.5000], [-100.0000, 36.5000], [-94.0000, 33.0000], [-93.5000, 29.0000], [-97.0000, 26.0000], [-100.0000, 25.8372], [-103.0000, 28.0000], [-106.5000, 31.7500], [-106.6456, 31.9999]]]},
}


def _build_engine():
    host = os.getenv("DATABASE_HOST", "localhost")
    port = os.getenv("DATABASE_PORT", "5432")
    db = os.getenv("DATABASE_NAME", "postgres")
    user = os.getenv("DATABASE_USER", "postgres")
    pwd = os.getenv("DATABASE_PASSWORD", "postgres")
    ssl_mode = os.getenv("DB_SSLMODE", "require")
    url = f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}?sslmode={ssl_mode}"
    return create_engine(
        url,
        pool_pre_ping=True,
        pool_recycle=3600,
        connect_args={
            "application_name": "eia_ingest",
            "keepalives": 1,
            "keepalives_idle": 300,
            "keepalives_interval": 30,
            "keepalives_count": 5,
        },
    )


ENGINE = _build_engine()

REGION_ENDPOINT = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
FUEL_ENDPOINT = "https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/"
INTERCHANGE_ENDPOINT = "https://api.eia.gov/v2/electricity/rto/interchange-data/data/"


def utc_hour_str_now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H")


def eia_paged_request(endpoint: str, params: dict, page_size: int = 5000, max_records: int = 5000) -> Iterator[dict]:
    offset = 0
    count = 0
    api_key = os.getenv("API_KEY")
    headers = {"X-Api-Key": api_key} if api_key else {}
    while count < max_records:
        q = params.copy(); q.update({"offset": offset, "length": min(page_size, max_records - count)})
        resp = requests.get(endpoint, params=q, headers=headers, timeout=60)
        if resp.status_code != 200:
            logger.error("EIA request failed %s %s", resp.status_code, resp.text)
            break
        data = resp.json().get("response", {})
        series = data.get("data", [])

        if not series:
            break
        for item in series:
            yield item
            count += 1
        offset += len(series)
        time.sleep(0.1)


def insert_from_df(df: pd.DataFrame, target_table: str, engine=ENGINE):
    """Pure INSERT (incremental append) without DELETE - for aggregate tables."""
    if df.empty:
        logger.info("Skipping insert for %s: empty dataframe", target_table)
        return
    # Add schema prefix if not present
    if '.' not in target_table:
        full_table = f"eia.{target_table}"
    else:
        full_table = target_table
    staging = f"eia.stg_{target_table.split('.')[-1]}_{int(time.time())}"
    cols_sql = ",".join([f'"{c}" TEXT' for c in df.columns])
    with engine.begin() as conn:
        conn.execute(text(f'DROP TABLE IF EXISTS {staging};'))
        conn.execute(text(f'CREATE TABLE {staging} ({cols_sql});'))
        rows = df.fillna("").values.tolist()
        placeholders = ",".join(["(" + ",".join(["%s"] * len(df.columns)) + ")" for _ in rows])
        flat = [str(x) for row in rows for x in row]
        insert_sql = f'INSERT INTO {staging} ("' + '","'.join(df.columns) + f'") VALUES {placeholders}'
        conn.connection.cursor().execute(insert_sql, flat)
        
        # Build SELECT with proper type casting
        numeric_cols = {"value", "generation_value", "demand_value", "interchange_value", "sent", "received"}
        timestamp_cols = {"period", "updated_at"}
        src_cols_list = []
        for c in df.columns:
            if c in timestamp_cols:
                src_cols_list.append(f'NULLIF({staging}."{c}", \'\')::timestamptz')
            elif c in numeric_cols or df[c].dtype.kind in "biufc":
                src_cols_list.append(f'NULLIF({staging}."{c}", \'\')::numeric')
            else:
                src_cols_list.append(f'NULLIF({staging}."{c}", \'\')') 
        
        tgt_cols = ",".join([f'"{c}"' for c in df.columns])
        src_cols = ",".join(src_cols_list)
        
        # Pure INSERT (no DELETE)
        insert_sql = f"""
        INSERT INTO {full_table} ({tgt_cols})
        SELECT {src_cols} FROM {staging};
        """
        conn.execute(text(insert_sql))
        conn.execute(text(f'DROP TABLE IF EXISTS {staging};'))
    logger.info("Inserted %d rows into %s (incremental append)", len(df), full_table)


def upsert_from_df(df: pd.DataFrame, target_table: str, conflict_cols: List[str], engine=ENGINE):
    """DELETE + INSERT strategy - for raw tables that need deduplication."""
    if df.empty:
        logger.info("Skipping upsert for %s: empty dataframe", target_table)
        return
    # Add schema prefix if not present
    if '.' not in target_table:
        full_table = f"eia.{target_table}"
    else:
        full_table = target_table
    staging = f"eia.stg_{target_table.split('.')[-1]}_{int(time.time())}"
    cols_sql = ",".join([f'"{c}" TEXT' for c in df.columns])
    with engine.begin() as conn:
        conn.execute(text(f'DROP TABLE IF EXISTS {staging};'))
        conn.execute(text(f'CREATE TABLE {staging} ({cols_sql});'))
        rows = df.fillna("").values.tolist()
        placeholders = ",".join(["(" + ",".join(["%s"] * len(df.columns)) + ")" for _ in rows])
        flat = [str(x) for row in rows for x in row]
        insert_sql = f'INSERT INTO {staging} ("' + '","'.join(df.columns) + f'") VALUES {placeholders}'
        conn.connection.cursor().execute(insert_sql, flat)
        
        # Build SELECT with proper type casting
        # Known numeric columns in EIA database schema
        numeric_cols = {"value", "generation_value", "demand_value", "interchange_value", "sent", "received"}
        # Known timestamp columns
        timestamp_cols = {"period", "updated_at"}
        src_cols_list = []
        for c in df.columns:
            if c in timestamp_cols:
                # Cast timestamp columns to timestamptz
                src_cols_list.append(f'NULLIF({staging}."{c}", \'\')::timestamptz')
            elif c in numeric_cols or df[c].dtype.kind in "biufc":  # numeric types
                # Cast numeric columns to numeric
                src_cols_list.append(f'NULLIF({staging}."{c}", \'\')::numeric')
            else:
                # Text columns - just use NULLIF
                src_cols_list.append(f'NULLIF({staging}."{c}", \'\')')
        
        tgt_cols = ",".join([f'"{c}"' for c in df.columns])
        src_cols = ",".join(src_cols_list)
        
        # DELETE + INSERT strategy (no unique constraint required)
        # Build WHERE clause for DELETE based on EXACT MATCH of ALL conflict_cols (not just IN for each individually)
        # This prevents deleting unrelated old data - only deletes exact duplicates
        delete_conditions = []
        for conflict_col in conflict_cols:
            col_type = "timestamptz" if conflict_col in timestamp_cols else "text"
            delete_conditions.append(f'"{conflict_col}"')
        
        # Use EXISTS subquery to only delete rows that have EXACT match on ALL conflict columns
        staging_cols = ", ".join([f'NULLIF("{c}", \'\')::{("timestamptz" if c in timestamp_cols else "text")}' for c in conflict_cols])
        target_cols = ", ".join([f't."{c}"' for c in conflict_cols])
        
        delete_sql = f"""
        DELETE FROM {full_table} t
        WHERE EXISTS (
            SELECT 1 FROM {staging} s
            WHERE ({target_cols}) = ({staging_cols})
        );
        """
        conn.execute(text(delete_sql))
        
        # Insert new records
        insert_sql = f"""
        INSERT INTO {full_table} ({tgt_cols})
        SELECT {src_cols} FROM {staging};
        """
        conn.execute(text(insert_sql))
        conn.execute(text(f'DROP TABLE IF EXISTS {staging};'))
    logger.info("Upserted %d rows into %s (DELETE + INSERT)", len(df), full_table)


def _utc_now_hour() -> datetime:
    return datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)


def _ensure_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _compute_time_window(historical_hours: int | None) -> tuple[datetime, datetime]:
    """Compute start/end (end inclusive adjustment) with optional fetch offset, UTC-aware."""
    now = _utc_now_hour()
    end_offset = int(os.getenv("API_END_OFFSET_HOURS", "1"))  # fetch slightly ahead to include latest posted hour
    end_inclusive = now + timedelta(hours=end_offset)
    start = now - timedelta(hours=historical_hours or 0)
    return start, end_inclusive


def _log_period_coverage(df: pd.DataFrame, label: str) -> None:
    """Log min/max period and detect staleness vs current hour."""
    if df.empty or "period" not in df.columns:
        logger.warning("%s ingestion returned empty or missing period column", label)
        return
    try:
        # periods should be UTC-aware
        min_p = df["period"].min()
        max_p = df["period"].max()
        now_hr = _utc_now_hour()
        lag_hours = (now_hr - max_p).total_seconds() / 3600.0 if max_p else None
        logger.info("%s period coverage: min=%s max=%s lag_hours=%.2f", label, min_p, max_p, lag_hours or -1)
        max_allowed_lag = int(os.getenv("MAX_ALLOWED_LAG_HOURS", "2"))
        if lag_hours is not None and lag_hours > max_allowed_lag:
            logger.warning(
                "%s latest period is stale (%.2f h > %d h). Consider increasing API_END_OFFSET_HOURS or checking API status.",
                label,
                lag_hours,
                max_allowed_lag,
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed logging period coverage for %s: %s", label, exc)


def ingest_region(historical_hours: int | None = None) -> pd.DataFrame:
    """Ingest region data starting from last DB period +1h up to now+2 days.
    Falls back to INITIAL_START_PERIOD if table empty. Ignores historical_hours when USE_DB_LAST_PERIOD=true."""
    use_db = os.getenv("USE_DB_LAST_PERIOD", "true").lower() == "true"
    initial_start_str = os.getenv("INITIAL_START_PERIOD", "2024-01-01T00")
    end_days_offset = int(os.getenv("FETCH_END_OFFSET_DAYS", "2"))
    max_records = int(os.getenv("MAX_RECORDS", "5000"))
    now_hr = _utc_now_hour()
    end_inclusive = now_hr + timedelta(days=end_days_offset)
    last_period = None
    if use_db:
        try:
            with ENGINE.connect() as conn:
                last_period = conn.execute(text("SELECT max(period) FROM eia.rto_region_data")).scalar()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to read last period region: %s", exc)
    last_period = _ensure_utc(last_period)
    if last_period:
        start = last_period + timedelta(hours=1)
    else:
        try:
            start = datetime.strptime(initial_start_str, "%Y-%m-%dT%H").replace(tzinfo=timezone.utc)
        except ValueError:
            logger.warning("Bad INITIAL_START_PERIOD '%s', fallback 24h", initial_start_str)
            start = now_hr - timedelta(hours=24)
    if start >= end_inclusive:
        start = end_inclusive - timedelta(hours=1)
    params = {
        "frequency": "hourly",
        "data[0]": "value",
        "facets[type][]": ["D", "NG", "TI"],
        "start": start.strftime("%Y-%m-%dT%H"),
        "end": end_inclusive.strftime("%Y-%m-%dT%H"),
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
    }
    records = list(eia_paged_request(REGION_ENDPOINT, params, page_size=5000, max_records=max_records))
    logger.info(
        "Region ingestion: %d records start=%s end=%s last_loaded=%s use_db=%s",
        len(records), start.strftime("%Y-%m-%dT%H"), end_inclusive.strftime("%Y-%m-%dT%H"),
        last_period.strftime("%Y-%m-%dT%H") if last_period else None, use_db
    )
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    if "period" in df.columns:
        df["period"] = pd.to_datetime(df["period"], errors="coerce", utc=True)
    _log_period_coverage(df, "region")
    return df


def ingest_fueltype(historical_hours: int | None = None) -> pd.DataFrame:
    use_db = os.getenv("USE_DB_LAST_PERIOD", "true").lower() == "true"
    initial_start_str = os.getenv("INITIAL_START_PERIOD", "2024-01-01T00")
    end_days_offset = int(os.getenv("FETCH_END_OFFSET_DAYS", "2"))
    max_records = int(os.getenv("MAX_RECORDS", "5000"))
    now_hr = _utc_now_hour()
    end_inclusive = now_hr + timedelta(days=end_days_offset)
    last_period = None
    if use_db:
        try:
            with ENGINE.connect() as conn:
                last_period = conn.execute(text("SELECT max(period) FROM eia.rto_fueltype_data")).scalar()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to read last period fueltype: %s", exc)
    last_period = _ensure_utc(last_period)
    if last_period:
        start = last_period + timedelta(hours=1)
    else:
        try:
            start = datetime.strptime(initial_start_str, "%Y-%m-%dT%H").replace(tzinfo=timezone.utc)
        except ValueError:
            logger.warning("Bad INITIAL_START_PERIOD '%s', fallback 24h", initial_start_str)
            start = now_hr - timedelta(hours=24)
    if start >= end_inclusive:
        start = end_inclusive - timedelta(hours=1)
    params = {
        "frequency": "hourly",
        "data[0]": "value",
        "start": start.strftime("%Y-%m-%dT%H"),
        "end": end_inclusive.strftime("%Y-%m-%dT%H"),
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
    }
    records = list(eia_paged_request(FUEL_ENDPOINT, params, page_size=5000, max_records=max_records))
    logger.info(
        "Fueltype ingestion: %d records start=%s end=%s last_loaded=%s use_db=%s",
        len(records), start.strftime("%Y-%m-%dT%H"), end_inclusive.strftime("%Y-%m-%dT%H"),
        last_period.strftime("%Y-%m-%dT%H") if last_period else None, use_db
    )
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df["period"] = pd.to_datetime(df["period"], errors="coerce", utc=True)
    _log_period_coverage(df, "fueltype")
    return df


def ingest_interchange(historical_hours: int | None = None) -> pd.DataFrame:
    use_db = os.getenv("USE_DB_LAST_PERIOD", "true").lower() == "true"
    initial_start_str = os.getenv("INITIAL_START_PERIOD", "2024-01-01T00")
    end_days_offset = int(os.getenv("FETCH_END_OFFSET_DAYS", "2"))
    max_records = int(os.getenv("MAX_RECORDS", "5000"))
    now_hr = _utc_now_hour()
    end_inclusive = now_hr + timedelta(days=end_days_offset)
    last_period = None
    if use_db:
        try:
            with ENGINE.connect() as conn:
                last_period = conn.execute(text("SELECT max(period) FROM eia.rto_interchange_data")).scalar()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to read last period interchange: %s", exc)
    last_period = _ensure_utc(last_period)
    if last_period:
        start = last_period + timedelta(hours=1)
    else:
        try:
            start = datetime.strptime(initial_start_str, "%Y-%m-%dT%H").replace(tzinfo=timezone.utc)
        except ValueError:
            logger.warning("Bad INITIAL_START_PERIOD '%s', fallback 24h", initial_start_str)
            start = now_hr - timedelta(hours=24)
    if start >= end_inclusive:
        start = end_inclusive - timedelta(hours=1)
    params = {
        "frequency": "hourly",
        "data[0]": "value",
        "start": start.strftime("%Y-%m-%dT%H"),
        "end": end_inclusive.strftime("%Y-%m-%dT%H"),
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
    }
    records = list(eia_paged_request(INTERCHANGE_ENDPOINT, params, page_size=5000, max_records=max_records))
    logger.info(
        "Interchange ingestion: %d records start=%s end=%s last_loaded=%s use_db=%s",
        len(records), start.strftime("%Y-%m-%dT%H"), end_inclusive.strftime("%Y-%m-%dT%H"),
        last_period.strftime("%Y-%m-%dT%H") if last_period else None, use_db
    )
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df["period"] = pd.to_datetime(df["period"], errors="coerce", utc=True)
    _log_period_coverage(df, "interchange")
    return df


def remove_high_missing(df: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    if df.empty:
        return df
    frac = df.isna().mean()
    keep = frac[frac <= threshold].index.tolist()
    return df[keep]


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    for c in df.columns:
        if df[c].dtype.kind in "biufc":
            df[c] = df[c].fillna(df[c].median())
        else:
            df[c] = df[c].fillna(df[c].mode().iloc[0])
    return df


def identify_outliers(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df and df[c].dtype.kind in "biufc":
            mu = df[c].mean(); sd = df[c].std() or 1
            df[f"z_{c}"] = (df[c] - mu) / sd
            df[f"is_outlier_{c}"] = (df[f"z_{c}"] > 3) | (df[f"z_{c}"] < -3)
    return df


def detect_anomalies(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    if target_col not in df or df.empty:
        df["anomaly_score"] = 0.0; df["anomaly"] = 0
        return df
    vals = df[target_col].values.reshape(-1, 1)
    try:
        iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        scores = -iso.fit(vals).score_samples(vals)
        df["anomaly_score"] = scores
        thresh = pd.Series(scores).quantile(0.95)
        is_anomaly_flag = scores >= thresh
    except Exception as exc:  # noqa: BLE001
        logger.warning("IsolationForest failed, fallback quantile logic: %s", exc)
        q95 = df[target_col].quantile(0.95)
        df["anomaly_score"] = (df[target_col] / q95).clip(upper=5.0)
        is_anomaly_flag = df[target_col] >= q95
    df["anomaly"] = is_anomaly_flag.astype(int)
    return df


def aggregate_region_data(region_df: pd.DataFrame, fuel_df: pd.DataFrame, interchange_df: pd.DataFrame) -> pd.DataFrame:
    # All datasets are required; API may not have published data yet
    if region_df.empty:
        logger.warning("Region data is empty - API may not have published data yet")
    if fuel_df.empty:
        logger.warning("Fuel data is empty - API may not have published data yet")
    if interchange_df.empty:
        logger.warning("Interchange data is empty - API may not have published data yet")
    if any(d.empty for d in [region_df, fuel_df, interchange_df]):
        return pd.DataFrame()
    region = region_df.copy(); fuel = fuel_df.copy(); inter = interchange_df.copy()
    fuel_pivot = fuel.pivot_table(index=["period", "respondent"], values="generation_value", columns="fueltype", aggfunc="sum").reset_index()
    sent = inter.groupby(["period", "source"], as_index=False)["interchange_value"].sum().rename(columns={"source": "respondent", "interchange_value": "sent"})
    received = inter.groupby(["period", "destination"], as_index=False)["interchange_value"].sum().rename(columns={"destination": "respondent", "interchange_value": "received"})
    merged = region.merge(fuel_pivot, on=["period", "respondent"], how="left")
    merged = merged.merge(sent, on=["period", "respondent"], how="left")
    merged = merged.merge(received, on=["period", "respondent"], how="left")
    merged["sent"] = pd.to_numeric(merged["sent"], errors="coerce").fillna(0)
    merged["received"] = pd.to_numeric(merged["received"], errors="coerce").fillna(0)
    merged["net_exchange"] = merged["sent"] - merged["received"]
    
    # Renombrar demand_value y generation_value a nombres de esquema DB
    merged.rename(columns={"demand_value": "demand_mw", "generation_value": "generation_mw"}, inplace=True)
    merged["demand_mw"] = pd.to_numeric(merged["demand_mw"], errors="coerce").fillna(0)
    merged["generation_mw"] = pd.to_numeric(merged["generation_mw"], errors="coerce").fillna(0)
    
    # Renombrar columnas fuel con sufijo _mw y convertir a numérico
    fuel_cols = [c for c in merged.columns if c not in ["period", "respondent", "demand_mw", "generation_mw", "sent", "received", "net_exchange"]]
    fuel_rename = {}
    for col in fuel_cols:
        if not col.endswith("_mw"):
            fuel_rename[col] = f"fuel_{col.lower()}_mw"
    if fuel_rename:
        merged.rename(columns=fuel_rename, inplace=True)
    
    # Convertir todas las columnas fuel_*_mw a numérico
    fuel_mw_cols = [c for c in merged.columns if c.startswith("fuel_") and c.endswith("_mw")]
    for col in fuel_mw_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0)
    
    merged["total_generation"] = merged[fuel_mw_cols].sum(axis=1) if fuel_mw_cols else 0
    merged["deficit"] = merged["demand_mw"] - merged["total_generation"] - merged["net_exchange"]
    merged["deficit_positive"] = merged["deficit"].clip(lower=0)
    merged["deficit_rank"] = merged.groupby("period")["deficit_positive"].rank(method="first", ascending=False)
    
    # Calcular deficit_pct_demand
    merged["deficit_pct_demand"] = (merged["deficit"] / merged["demand_mw"] * 100).replace([float('inf'), -float('inf')], 0).fillna(0)
    
    # Agregar columna TI (Total Interchange) - igual a net_exchange
    merged["TI"] = merged["net_exchange"]
    
    # Renombrar sent/received a energy_sent/energy_received
    merged.rename(columns={"sent": "energy_sent", "received": "energy_received"}, inplace=True)
    
    # Agregar coordenadas, region_name y boundary
    # Enriquecer con coordenadas y geocercas desde REGION_INFO
    lats = []
    lons = []
    names = []
    boundaries = []
    for r in merged["respondent"].astype(str):
        info = REGION_INFO.get(r)
        if info:
            lat, lon = info.get("center", (math.nan, math.nan))
            name = info.get("name", r)
            boundary = info.get("boundary")
        else:
            lat, lon = (math.nan, math.nan)
            name = r
            boundary = None
        lats.append(lat)
        lons.append(lon)
        names.append(name)
        # Serializar boundary como JSON si existe
        try:
            boundaries.append(json.dumps(boundary) if boundary else None)
        except Exception:
            boundaries.append(None)
    merged["lat"] = lats
    merged["lon"] = lons
    merged["region"] = merged["respondent"]  # Copiar respondent a region para esquema DB
    merged["region_name"] = names
    merged["boundary"] = boundaries
    
    # Agregar updated_at timestamp (UTC-aware)
    merged["updated_at"] = datetime.now(timezone.utc)
    
    return merged


def run_eda_aggregation(historical_hours: int | None = None):
    logger.info("Starting ingestion for aggregation")
    
    # Ingest each dataset independently from API
    region_df_api = ingest_region(historical_hours)
    fuel_df_api = ingest_fueltype(historical_hours)
    interchange_df_api = ingest_interchange(historical_hours)
    
    # Persist each dataset independently if has new data
    if not region_df_api.empty:
        # Keep original API data structure with all columns
        region_raw = region_df_api.copy()
        # Map back to raw format expected by database
        upsert_from_df(region_raw, "rto_region_data", ["period", "respondent", "type"])
        logger.info("Updated region data: %d records", len(region_raw))
    else:
        logger.info("No new region data from API")
    
    if not fuel_df_api.empty:
        # Keep original API data with all columns
        fuel_raw = fuel_df_api.copy()
        upsert_from_df(fuel_raw, "rto_fueltype_data", ["period", "respondent", "fueltype"])
        logger.info("Updated fueltype data: %d records", len(fuel_raw))
    else:
        logger.info("No new fueltype data from API")
    
    if not interchange_df_api.empty:
        # Keep original API data with all columns (fromba, toba, value)
        interchange_raw = interchange_df_api.copy()
        upsert_from_df(interchange_raw, "rto_interchange_data", ["period", "fromba", "toba"])
        logger.info("Updated interchange data: %d records", len(interchange_raw))
    else:
        logger.info("No new interchange data from API")
    
    # ALWAYS read latest data from tables for aggregation (regardless of new data)
    # Use aggregation window to only process recent data
    agg_window_hours = int(os.getenv("AGG_LOOKBACK_HOURS", "48"))
    logger.info("Reading latest %d hours from tables for aggregation", agg_window_hours)
    
    with ENGINE.connect() as conn:
        # Read region data and pivot it - filter by retention period to avoid processing stale data
        region_query = text("""
            SELECT period, respondent, type, value 
            FROM eia.rto_region_data 
            WHERE type IN ('D', 'NG')
              AND period >= (NOW() AT TIME ZONE 'UTC' - :agg_hours * INTERVAL '1 hour')
            ORDER BY period DESC
        """)
        region_raw_db = pd.read_sql(region_query, conn, params={"agg_hours": agg_window_hours})
        if not region_raw_db.empty:
            region_db = region_raw_db.pivot_table(index=["period", "respondent"], columns="type", values="value", aggfunc="sum").reset_index()
            if "D" in region_db.columns:
                region_db.rename(columns={"D": "demand_value"}, inplace=True)
            else:
                region_db["demand_value"] = 0
            if "NG" in region_db.columns:
                region_db.rename(columns={"NG": "generation_value"}, inplace=True)
            else:
                region_db["generation_value"] = 0
            logger.info("Read %d region records from database", len(region_db))
        else:
            region_db = pd.DataFrame(columns=["period", "respondent", "demand_value", "generation_value"])
        
        # Read fuel data - filter by retention period
        fuel_query = text("""
            SELECT period, respondent, fueltype, value as generation_value 
            FROM eia.rto_fueltype_data 
            WHERE period >= (NOW() AT TIME ZONE 'UTC' - :agg_hours * INTERVAL '1 hour')
            ORDER BY period DESC
        """)
        fuel_db = pd.read_sql(fuel_query, conn, params={"agg_hours": agg_window_hours})
        if not fuel_db.empty:
            logger.info("Read %d fuel records from database", len(fuel_db))
        
        # Read interchange data with correct column names (fromba, toba) - filter by retention period
        interchange_query = text("""
            SELECT period, fromba as source, toba as destination, value as interchange_value 
            FROM eia.rto_interchange_data 
            WHERE period >= (NOW() AT TIME ZONE 'UTC' - :agg_hours * INTERVAL '1 hour')
            ORDER BY period DESC
        """)
        interchange_db = pd.read_sql(interchange_query, conn, params={"agg_hours": agg_window_hours})
        if not interchange_db.empty:
            logger.info("Read %d interchange records from database", len(interchange_db))
    
    # region_db is already pivoted above
    # Perform EDA aggregation with latest available data from tables
    agg = aggregate_region_data(region_db, fuel_db, interchange_db)
    if agg.empty:
        logger.warning("No data available in tables for aggregation")
        return
    
    # EDA processing
    agg = remove_high_missing(agg); agg = impute_missing(agg)
    agg = identify_outliers(agg, ["demand_mw", "total_generation", "deficit"])
    agg = detect_anomalies(agg, "deficit")
    
    # Consolidar outlier flags en outlier_score y is_outlier
    outlier_cols = [c for c in agg.columns if c.startswith("is_outlier_")]
    if outlier_cols:
        agg["is_outlier"] = agg[outlier_cols].any(axis=1).astype(int)
    else:
        agg["is_outlier"] = 0
    
    z_cols = [c for c in agg.columns if c.startswith("z_")]
    if z_cols:
        agg["outlier_score"] = agg[z_cols].abs().max(axis=1)
    else:
        agg["outlier_score"] = 0.0
    
    # Asegurar que anomaly es int
    agg["anomaly"] = agg["anomaly"].astype(int)
    
    # Drop columns not in target table schema
    cols_to_drop = ["respondent"]  # respondent not in eia_aggregate_realtime, only region
    # Drop intermediate columns (z_*, is_outlier_*, deficit_positive, total_generation) not in target schema
    intermediate_cols = [c for c in agg.columns if c.startswith("z_") or c.startswith("is_outlier_") or c == "deficit_positive" or c == "total_generation"]
    cols_to_drop.extend(intermediate_cols)
    
    for col in cols_to_drop:
        if col in agg.columns:
            agg = agg.drop(columns=[col])
    
    # Persist aggregated results to single table: eia_aggregate_realtime (incremental INSERT)
    insert_from_df(agg, "eia_aggregate_realtime")
    logger.info("EDA aggregation + persistence complete: %d rows to eia_aggregate_realtime", len(agg))

    # Retention policies
    agg_retention_hours = int(os.getenv("AGG_RETENTION_HOURS", "48"))
    raw_retention_days = int(os.getenv("RAW_RETENTION_DAYS", "730"))  # ~2 years
    try:
        with ENGINE.begin() as conn:
            # Apply retention to aggregate table (hours)
            delete_sql = text("""
                DELETE FROM eia.eia_aggregate_realtime
                WHERE period < (NOW() AT TIME ZONE 'UTC' - (:rh * INTERVAL '1 hour'))
            """)
            conn.execute(delete_sql, {"rh": agg_retention_hours})
            
            # Apply retention to raw tables to keep only last N days
            for raw_table in ["eia.rto_region_data", "eia.rto_fueltype_data", "eia.rto_interchange_data"]:
                delete_raw_sql = text(f"""
                    DELETE FROM {raw_table}
                    WHERE period < (NOW() AT TIME ZONE 'UTC' - (:rd * INTERVAL '1 day'))
                """)
                conn.execute(delete_raw_sql, {"rd": raw_retention_days})
            
        logger.info("Retention applied: aggregate=%d hours, raw=%d days", agg_retention_hours, raw_retention_days)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Retention cleanup failed: %s", exc)


def ingest_cycle():  # Placeholder for Azure timer trigger
    try:
        logger.info("Timer trigger start")
        hist_hours = os.getenv("HISTORICAL_HOURS")
        historical = int(hist_hours) if hist_hours else 24  # Default: 24 hours for faster execution
        logger.info("Using historical hours: %d", historical)
        run_eda_aggregation(historical)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ingest cycle failed: %s", exc)
    finally:
        logger.info("Timer trigger end")



# Azure Functions timer trigger: ejecuta ingest_cycle cada minuto
app = func.FunctionApp()

@app.function_name(name="TimerIngestFunction")
@app.schedule(schedule="0 */1 * * * *", arg_name="mytimer", run_on_startup=True, use_monitor=False)
def timer_ingest_function(mytimer: func.TimerRequest) -> None:
    """Timer trigger que ejecuta el ciclo de ingesta y EDA cada minuto."""
    ingest_cycle()


