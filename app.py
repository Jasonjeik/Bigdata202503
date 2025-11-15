import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
import datetime as dt
import ast
from streamlit import runtime
import os


# =========================
# CONFIGURACIÓN DE BASE DE DATOS
# =========================
DB_USER = os.getenv("DB_USER", "prj1_admin")
DB_PASS = os.getenv("DB_PASSWORD", "Bigdataproyecto1")
DB_HOST = os.getenv("DB_HOST", "bigdataproyecto1.postgres.database.azure.com")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "proyecto1")
SCHEMA = os.getenv("DB_SCHEMA", "eia")

@st.cache_resource
def get_engine():
    """Crea conexión a PostgreSQL con cache."""
    return create_engine(
        f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
        connect_args={
            "sslmode": "require",
            "options": f"-csearch_path={SCHEMA}",
            "connect_timeout": 10,
            "keepalives": 1,
            "keepalives_idle": 30,
            "keepalives_interval": 10,
            "keepalives_count": 5
        },
        pool_pre_ping=True,
        pool_recycle=3600,
        pool_size=5,
        max_overflow=10
    )

def load_realtime_data():
    """Carga datos agregados en tiempo real desde PostgreSQL.

    Coloca el mensaje de error en st.session_state['db_error'] si falla.
    """
    engine = get_engine()
    query = f'SELECT * FROM {SCHEMA}.eia_aggregate_realtime ORDER BY period DESC;'
    try:
        # Ejecutar query directamente (pool_pre_ping valida conexión automáticamente)
        df = pd.read_sql(query, engine)
        if 'period' in df.columns:
            df['period'] = pd.to_datetime(df['period'], errors='coerce')
        if 'updated_at' in df.columns:
            df['updated_at'] = pd.to_datetime(df['updated_at'], errors='coerce')
        st.session_state.pop('db_error', None)
        return df
    except Exception as e:
        # Sanitizar mensaje (evitar mostrar contraseña si apareciera)
        msg = str(e).replace(DB_PASS, "***") if DB_PASS else str(e)
        st.session_state['db_error'] = msg
        return None

def get_last_update_time(df):
    """Obtiene el timestamp de la última actualización."""
    if df is None or df.empty or 'updated_at' not in df.columns:
        return None
    return df['updated_at'].max()

# =========================
# CONFIGURACIÓN DE PÁGINA
# =========================
st.set_page_config(
    page_title="⚡ EIA Grid Monitor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# TÍTULO Y HEADER
# =========================
st.title("⚡ Monitoreo en Tiempo Real - Red Eléctrica EIA")

# =========================
# CONTROL DE REFRESCO
# =========================
refresh_col1, refresh_col2, refresh_col3 = st.columns([1,1,2])
with refresh_col1:
    if st.button("🔄 Refrescar Ahora"):
        st.cache_data.clear()
        st.rerun()
with refresh_col2:
    auto_refresh = st.checkbox("Auto-refresco", value=False, help="Actualiza datos sin interrumpir navegación")
with refresh_col3:
    refresh_interval = st.selectbox("Intervalo (seg)", [30,60,120,300], index=1, help="Frecuencia de actualización automática")

if auto_refresh:
    # Intento nativo (Streamlit >=1.28) sino fallback JS
    try:
        if hasattr(st, 'autorefresh'):
            st.autorefresh(interval=refresh_interval * 1000, key="data_autorefresh")
        else:
            raise AttributeError("autorefresh not available")
    except Exception:
        st.markdown(
            f"<script>setTimeout(()=>window.parent.location.reload(), {refresh_interval*1000});</script>",
            unsafe_allow_html=True
        )

# =========================
# CARGA DE DATOS
# =========================
with st.spinner("Cargando datos..."):
    df = load_realtime_data()

# Mostrar última actualización
if df is not None and not df.empty:
    last_update = get_last_update_time(df)
    if last_update:
        st.markdown(f"**📅 Última actualización de datos:** {last_update.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    else:
        st.markdown(f"**📅 Datos cargados:** {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
else:
    st.markdown(f"**📅 Consultado:** {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if df is None:
    st.error("❌ Error al conectar con la base de datos o la tabla no existe.")
    if 'db_error' in st.session_state:
        st.code(st.session_state['db_error'])
    st.info("""
    **Pasos para resolver:**
    
    1. ⏰ Verifica que la Azure Function esté ejecutándose (timer cada minuto)
    
    2. ⏳ Espera al menos 1 minuto para que se cree la tabla `eia_aggregate_realtime`
    
    3. 🔄 Recarga esta página (Ctrl+R o F5)
    
    **Estado actual:** La tabla `eia.eia_aggregate_realtime` aún no existe en PostgreSQL o la Function no está activa.
    """)
    st.stop()

if df.empty:
    st.warning("⚠️ La tabla existe pero no tiene datos aún.")
    st.info("""
    **El pipeline se está ejecutando...**
    
    - ✅ Tabla creada
    - ⏳ Esperando primer ciclo de agregación (puede tardar 1-2 minutos)
    - 🔄 Esta página se actualizará automáticamente cada 60 segundos
    """)
    st.stop()

# Verificar que existan las columnas necesarias
required_cols = ['period', 'region', 'demand_mw', 'generation_mw', 'deficit']
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    st.error(f"❌ Faltan columnas en la tabla: {', '.join(missing_cols)}")
    st.info("""
    **La tabla tiene una estructura inesperada.**
    
    Esto puede ocurrir si:
    - El script `ingesta_eia_incremental.py` no ha completado el primer ciclo de EDA
    - Hay un error en la función `run_eda_aggregation()`
    
    Revisa los logs del script de ingesta para ver si hay errores.
    """)
    st.write("**Columnas disponibles:**", df.columns.tolist())
    st.stop()

# =========================
# SIDEBAR - FILTROS
# =========================
st.sidebar.header("🏛️ Filtros")

# Filtro de región
regions = ["Todas"] + sorted(df['region'].dropna().unique().tolist())
selected_region = st.sidebar.selectbox("Región", regions)

# Filtro de tiempo
hours_back = st.sidebar.slider("Horas hacia atrás", 1, 672, 24)

# Filtro de anomalías
show_anomalies = st.sidebar.checkbox("Mostrar solo anomalías", False)

# Información de debug
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Info de Datos")
st.sidebar.write(f"Total registros: {len(df):,}")

# Aplicar filtros
df_filtered = df.copy()

if selected_region != "Todas":
    df_filtered = df_filtered[df_filtered['region'] == selected_region]

# Filtro temporal - usar UTC para comparar correctamente
if 'period' in df_filtered.columns:
    # Asegurar que period es timezone-aware en UTC
    if df_filtered['period'].dt.tz is None:
        df_filtered['period'] = pd.to_datetime(df_filtered['period']).dt.tz_localize('UTC')
    elif df_filtered['period'].dt.tz != dt.timezone.utc:
        df_filtered['period'] = df_filtered['period'].dt.tz_convert('UTC')
    
    # Crear cutoff_time con timezone UTC
    cutoff_time = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=hours_back)
    df_filtered = df_filtered[df_filtered['period'] >= cutoff_time]

if show_anomalies and 'anomaly' in df_filtered.columns:
    df_filtered = df_filtered[df_filtered['anomaly'] == 1]

# Actualizar sidebar con info de filtros aplicados
st.sidebar.write(f"Registros filtrados: {len(df_filtered):,}")
if len(df_filtered) == 0:
    st.sidebar.warning("⚠️ No hay datos después de filtrar")

# =========================
# MÉTRICAS PRINCIPALES
# =========================
overview_tab, map_tab, fuel_tab, anomaly_tab, prediction_tab, data_tab = st.tabs([
    "📊 Overview", "🗺️ Mapa", "🔥 Fuel Mix", "🚨 Anomalías", "🔮 Predicción", "📁 Datos"
])

with overview_tab:
    st.subheader("Métricas Clave")
    if df_filtered.empty:
        st.warning(f"⚠️ No hay datos en el rango seleccionado ({hours_back}h)")
        st.stop()
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        total_demand = df_filtered.get('demand_mw', pd.Series(dtype=float)).sum()
        st.metric("Demanda Total", f"{total_demand:,.0f} MW")
    with m2:
        total_generation = df_filtered.get('generation_mw', pd.Series(dtype=float)).sum()
        st.metric("Generación Total", f"{total_generation:,.0f} MW")
    with m3:
        total_deficit = df_filtered.get('deficit', pd.Series(dtype=float)).sum()
        st.metric("Déficit Total", f"{total_deficit:,.0f} MW", delta=f"{total_deficit:,.0f} MW", delta_color="normal" if total_deficit >= 0 else "inverse")
    with m4:
        num_anomalies = df_filtered.get('anomaly', pd.Series(dtype=int)).sum()
        st.metric("Anomalías Detectadas", int(num_anomalies))

    st.markdown("---")
    st.subheader("Evolución Temporal del Déficit")
    if 'deficit' in df_filtered.columns and not df_filtered.empty:
        df_ts = df_filtered.groupby(['period', 'region'])['deficit'].sum().reset_index()
        fig_line = px.line(
            df_ts,
            x='period', y='deficit', color='region', markers=True,
            title='Déficit por Región en el Tiempo', labels={'deficit':'Déficit (MW)','period':'Periodo'}
        )
        fig_line.update_layout(
            hovermode='x unified',
            legend=dict(
                orientation='v',
                yanchor='top',
                y=1,
                xanchor='left',
                x=1.02,
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='rgba(0, 0, 0, 0.2)',
                borderwidth=1,
                font=dict(size=9)
            ),
            margin=dict(r=150)
        )
        st.plotly_chart(fig_line, width='stretch')
    else:
        st.info("No hay datos de déficit para mostrar.")

    st.markdown("---")
    st.subheader("⏰ Ventana de Datos")
    c1, c2, c3, c4 = st.columns(4)
    if not df.empty and 'period' in df.columns:
        min_period = df['period'].min(); max_period = df['period'].max()
        total_hours = (max_period - min_period).total_seconds() / 3600
        with c1: st.metric("Periodo más antiguo", min_period.strftime('%Y-%m-%d %H:%M'))
        with c2: st.metric("Periodo más reciente", max_period.strftime('%Y-%m-%d %H:%M'))
        with c3: st.metric("Ventana (period)", f"{total_hours:.1f} horas")
    if not df.empty and 'updated_at' in df.columns:
        total_updates = len(df); unique_periods = df['period'].nunique() if 'period' in df.columns else 0
        with c4: st.metric("Total actualizaciones", f"{total_updates:,}", help="Registros acumulados últimas 24h")
        st.info(f"ℹ️ Historial: {total_updates:,} updates de {unique_periods} periodos únicos (últimas 24h)")

with map_tab:
    st.subheader("Mapa de Regiones")
    if 'lat' in df_filtered.columns and 'lon' in df_filtered.columns and not df_filtered.empty:
        df_map = df_filtered.sort_values('period').groupby('region').last().reset_index()
        df_map['status'] = df_map.get('anomaly', pd.Series([0]*len(df_map))).apply(lambda x: '🚨 Anomalía' if x == 1 else '✅ Normal')
        color_map = {'✅ Normal': 'green', '🚨 Anomalía': 'red'}
        fig_map = go.Figure(); boundary_present = ('boundary' in df_map.columns) and df_map['boundary'].notna().any()
        if boundary_present:
            drawn = 0
            for _, row in df_map.iterrows():
                boundaries = []
                b = row.get('boundary')
                if b not in [None, '', '[]']:
                    try:
                        boundaries = ast.literal_eval(b) if isinstance(b, str) else b
                    except Exception:
                        boundaries = []
                status = row['status']; edge_color = 'red' if status.startswith('🚨') else 'green'
                fillcolor = 'rgba(255,0,0,0.2)' if status.startswith('🚨') else 'rgba(0,128,0,0.2)'
                ts = row.get('period'); ua = row.get('updated_at')
                def fmt(t):
                    try:
                        if pd.notnull(t):
                            t = pd.to_datetime(t)
                            t = t.tz_convert('UTC') if getattr(t,'tzinfo',None) else t.tz_localize('UTC')
                            return t.strftime('%Y-%m-%d %H:%M UTC')
                    except Exception: return ''
                    return ''
                ts_str = fmt(ts); ua_str = fmt(ua)
                for poly in boundaries or []:
                    try:
                        lons=[p[0] for p in poly]; lats=[p[1] for p in poly]
                        if not lons or not lats: continue
                        if lons[0]!=lons[-1] or lats[0]!=lats[-1]: lons.append(lons[0]); lats.append(lats[0])
                        hovertext=(f"Región: {row.get('region','')}<br>Nombre: {row.get('region_name','')}<br>Demanda: {row.get('demand_mw',0):,.0f} MW<br>Generación: {row.get('generation_mw',0):,.0f} MW<br>Déficit: {row.get('deficit',0):,.0f} MW<br>Hora: {ts_str}<br>Actualizado: {ua_str}<br>Estado: {status}")
                        fig_map.add_trace(go.Scattermap(lon=lons, lat=lats, mode='lines', fill='toself', fillcolor=fillcolor, line=dict(color=edge_color,width=2), name=f"{row.get('region','')} zona", hoverinfo='text', hovertext=hovertext, showlegend=False))
                        drawn += 1
                    except Exception: continue
            if drawn==0: st.info("No se detectaron geocercas.")
        marker_colors=[color_map.get(s,'green') for s in df_map['status']]
        def fmt_series(series):
            out=[]
            for v in series:
                try:
                    if pd.notnull(v):
                        v=pd.to_datetime(v); v=v.tz_convert('UTC') if getattr(v,'tzinfo',None) else v.tz_localize('UTC')
                        out.append(v.strftime('%Y-%m-%d %H:%M UTC'))
                    else: out.append('')
                except Exception: out.append('')
            return out
        time_texts=fmt_series(df_map.get('period', pd.Series([None]*len(df_map))))
        update_texts=fmt_series(df_map.get('updated_at', pd.Series([None]*len(df_map))))
        hover_texts=[(
            f"Región: {r}<br>Nombre: {n}<br>Demanda: {d:,.0f} MW<br>Generación: {g:,.0f} MW<br>Déficit: {dfv:,.0f} MW<br>Hora: {t}<br>Actualizado: {u}<br>Estado: {s}" )
            for r,n,d,g,dfv,s,t,u in zip(
                df_map['region'], df_map.get('region_name', pd.Series(['']*len(df_map))),
                df_map.get('demand_mw', pd.Series([0]*len(df_map))), df_map.get('generation_mw', pd.Series([0]*len(df_map))),
                df_map.get('deficit', pd.Series([0]*len(df_map))), df_map['status'], pd.Series(time_texts), pd.Series(update_texts))]
        fig_map.add_trace(go.Scattermap(lon=df_map['lon'], lat=df_map['lat'], mode='markers', marker=dict(size=10,color=marker_colors), text=hover_texts, hoverinfo='text', name='Regiones'))
        center_lon=float(pd.to_numeric(df_map['lon'], errors='coerce').dropna().mean()) if 'lon' in df_map else -96
        center_lat=float(pd.to_numeric(df_map['lat'], errors='coerce').dropna().mean()) if 'lat' in df_map else 38
        fig_map.update_layout(map_style='open-street-map', map=dict(zoom=3, center=dict(lon=center_lon, lat=center_lat)), margin={"r":0,"t":40,"l":0,"b":0}, legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
        st.plotly_chart(fig_map, width='stretch')
    else:
        st.info("No hay datos geográficos disponibles.")

with fuel_tab:
    st.subheader("Distribución por Combustible")
    fuel_cols=[c for c in df_filtered.columns if c.startswith('fuel_') and c.endswith('_mw')]
    if fuel_cols and not df_filtered.empty:
        fuels_numeric=df_filtered[fuel_cols].apply(pd.to_numeric, errors='coerce')
        fuel_totals=fuels_numeric.sum(min_count=1).fillna(0).sort_values(ascending=False)
        fuel_totals.index=fuel_totals.index.str.replace('fuel_','').str.replace('_mw','').str.upper()
        c1,c2=st.columns(2)
        with c1:
            fig_pie=px.pie(values=fuel_totals.values, names=fuel_totals.index, title='Proporción por Combustible', hole=0.3)
            st.plotly_chart(fig_pie, width='stretch')
        with c2:
            fig_bar=px.bar(x=fuel_totals.index, y=fuel_totals.values, labels={'x':'Tipo','y':'Generación (MW)'}, title='Generación Total', color=fuel_totals.values, color_continuous_scale='Viridis')
            st.plotly_chart(fig_bar, width='stretch')
    else:
        st.info("No hay datos de combustibles.")

with anomaly_tab:
    st.subheader("Alertas de Anomalías")
    if 'anomaly' in df_filtered.columns:
        df_anomalies=df_filtered[df_filtered['anomaly']==1].sort_values('period', ascending=False)
        if not df_anomalies.empty:
            display_cols=['period','region','demand_mw','generation_mw','deficit']
            display_cols=[c for c in display_cols if c in df_anomalies.columns]
            st.dataframe(df_anomalies[display_cols].head(50), width='stretch', hide_index=True)
        else:
            st.success("✅ No se detectaron anomalías.")
    else:
        st.info("La columna 'anomaly' no está disponible.")

with prediction_tab:
    st.subheader("Predicción de Demanda (Próximas 24h)")
    try:
        from prediction_model_v2 import forecast_demand_pipeline_v2, MODEL_DIR
        import os
        import json
        
        # Mostrar modelos disponibles
        if os.path.exists(MODEL_DIR):
            model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('_meta.json')]
            if model_files:
                st.info(f"🤖 **Modelos pre-entrenados disponibles:** {len(model_files)}")
                with st.expander("Ver modelos guardados"):
                    for meta_file in sorted(model_files):
                        try:
                            with open(os.path.join(MODEL_DIR, meta_file), 'r') as f:
                                meta = json.load(f)
                            region_name = meta.get('region', 'Desconocida')
                            model_type = meta.get('model_type', 'N/A')
                            saved_at = meta.get('saved_at', 'N/A')
                            trained_on = meta.get('trained_on', 0)
                            metrics = meta.get('metrics', {})
                            mae = metrics.get('MAE', 0)
                            rmse = metrics.get('RMSE', 0)
                            mape = metrics.get('MAPE', 0)
                            
                            st.markdown(f"""
                            **Región:** {region_name} | **Tipo:** {model_type} | **Guardado:** {saved_at}
                            - Datos de entrenamiento: {trained_on:,} registros
                            - MAE: {mae:.2f} MW | RMSE: {rmse:.2f} MW | MAPE: {mape:.2f}%
                            """)
                        except Exception as e:
                            st.warning(f"⚠️ Error leyendo {meta_file}: {e}")
            else:
                st.warning("⚠️ No hay modelos pre-entrenados. Se entrenará uno nuevo.")
        
        pred_regions=["TODAS"]+sorted([r for r in df['region'].dropna().unique() if r!=''])
        selected_pred_region=st.selectbox("Región", pred_regions, key='pred_region')
        cfg1,cfg2,cfg3,cfg4=st.columns(4)
        with cfg1: days_back=st.number_input("Días históricos", min_value=30, max_value=365, value=180, step=30)
        with cfg2: forecast_hours=st.number_input("Horas a predecir", min_value=6, max_value=72, value=24, step=6)
        with cfg3: use_lstm=st.checkbox("Usar LSTM", value=True)
        with cfg4: perform_cv=st.checkbox("Cross-Validation", value=True, help="TimeSeries K-Fold")
        n_splits=st.slider("Folds", min_value=2, max_value=5, value=3, disabled=not perform_cv)
        force_retrain=st.checkbox("Forzar reentrenamiento", value=False, help="Entrena y actualiza el mejor modelo guardado (ignora modelos pre-entrenados)")
        if st.button("🤖 Generar Predicción", key='generate_forecast_v2'):
            # Crear contenedor para mensajes de estado
            status_container = st.empty()
            progress_bar = st.progress(0)
            
            try:
                region_code=None if selected_pred_region=="TODAS" else selected_pred_region
                
                # Verificar si existe modelo pre-entrenado
                from prediction_model_v2 import _has_saved_artifacts, _region_key
                has_pretrained = _has_saved_artifacts(region_code) and not force_retrain
                
                if force_retrain:
                    status_container.info("🔄 **Modo:** Re-entrenamiento forzado - Entrenando nuevo modelo...")
                elif has_pretrained:
                    status_container.success("📦 **Modo:** Cargando modelo pre-entrenado existente...")
                else:
                    status_container.warning("🆕 **Modo:** No hay modelo guardado - Entrenando nuevo modelo...")
                
                progress_bar.progress(20)
                
                forecast_df, model_info, metrics=forecast_demand_pipeline_v2(
                    region_code=region_code, 
                    days_back=days_back, 
                    forecast_hours=forecast_hours, 
                    use_lstm=use_lstm, 
                    use_prophet_fallback=True, 
                    perform_cv=perform_cv, 
                    n_splits=n_splits, 
                    force_retrain=force_retrain
                )
                
                progress_bar.progress(100)
                
                if forecast_df is not None and not forecast_df.empty:
                    st.session_state['forecast']=forecast_df
                    st.session_state['forecast_region']=selected_pred_region
                    st.session_state['forecast_metrics']=metrics
                    st.session_state['forecast_model_info']=model_info
                    
                    # Determinar fuente del modelo
                    if force_retrain:
                        model_source = "re-entrenado"
                        icon = "🔄"
                    elif model_info.get('pretrained_used'):
                        model_source = "pre-entrenado (cargado desde disco)"
                        icon = "📦"
                    else:
                        model_source = "nuevo (recién entrenado)"
                        icon = "🆕"
                    
                    status_container.success(f"{icon} **Predicción completada** con modelo {model_source} ({metrics.get('model','?')})")
                    progress_bar.empty()
                else:
                    status_container.error("❌ Datos insuficientes para generar predicción.")
                    progress_bar.empty()
            except Exception as e:
                status_container.error(f"❌ Error en predicción: {e}")
                progress_bar.empty()
                import traceback
                st.code(traceback.format_exc())
        if 'forecast' in st.session_state and st.session_state['forecast'] is not None:
            forecast_df=st.session_state['forecast']; metrics=st.session_state.get('forecast_metrics', {}); region_name=st.session_state.get('forecast_region','TODAS')
            model_info = st.session_state.get('forecast_model_info', {})
            is_pretrained = model_info.get('pretrained_used', False)
            trained_records = model_info.get('trained_on', 0)
            
            model_badge = "🔄 Modelo pre-entrenado" if is_pretrained else "🆕 Modelo nuevo"
            st.markdown(f"**Modelo:** {metrics.get('model','?')} | {model_badge} | Registros de entrenamiento: {trained_records:,}")
            mc1,mc2,mc3=st.columns(3)
            with mc1: st.metric("MAE", f"{metrics.get('MAE',0):.2f} MW")
            with mc2: st.metric("RMSE", f"{metrics.get('RMSE',0):.2f} MW")
            with mc3: st.metric("MAPE", f"{metrics.get('MAPE',0):.2f}%")
            if 'CV_MAE' in metrics:
                st.markdown("**Cross-Validation**")
                cv1,cv2,cv3=cv_cols=st.columns(3)
                with cv1: st.metric("CV MAE", f"{metrics.get('CV_MAE',0):.2f} MW")
                with cv2: st.metric("CV RMSE", f"{metrics.get('CV_RMSE',0):.2f} MW")
                with cv3: st.metric("CV MAPE", f"{metrics.get('CV_MAPE',0)*100:.2f}%")
            fg=go.Figure(); fg.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'], mode='lines+markers', name='Predicción', line=dict(color='#1f77b4', width=3)))
            fg.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_upper'], mode='lines', name='Lim sup', line=dict(color='rgba(31,119,180,0.3)', width=0), hoverinfo='skip'))
            fg.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_lower'], mode='lines', name='Lim inf', line=dict(color='rgba(31,119,180,0.3)', width=0), fill='tonexty', fillcolor='rgba(31,119,180,0.1)', hoverinfo='skip'))
            fg.update_layout(title=f'Predicción {region_name} ({forecast_hours}h)', xaxis_title='Fecha/Hora', yaxis_title='Demanda (MW)', hovermode='x unified', height=420)
            st.plotly_chart(fg, width='stretch')
            with st.expander("Tabla de predicciones"):
                tdf = forecast_df.copy()
                tdf['ds'] = tdf['ds'].dt.strftime('%Y-%m-%d %H:%M')
                tdf.columns = ['Fecha/Hora','Predicción (MW)','Límite Inferior (MW)','Límite Superior (MW)']
                tdf_styled = tdf.style.format({
                    'Predicción (MW)': '{:.2f}',
                    'Límite Inferior (MW)': '{:.2f}',
                    'Límite Superior (MW)': '{:.2f}'
                })
                st.dataframe(tdf_styled, width='stretch', hide_index=True)
                csv = tdf.to_csv(index=False)
                st.download_button(
                    "Descargar CSV",
                    data=csv,
                    file_name=f"prediccion_{region_name}_{dt.datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime='text/csv'
                )
    except ImportError:
        st.warning("Instala dependencias: tensorflow prophet scikit-learn")

with data_tab:
    st.subheader("Datos Completos")
    with st.expander("Ver Datos"):
        display_df=df_filtered.drop(columns=['updated_at'], errors='ignore')
        st.dataframe(display_df, width='stretch', hide_index=True)
    with st.expander("Estadísticas Descriptivas"):
        numeric_cols=df_filtered.select_dtypes(include=['float64','int64']).columns
        if len(numeric_cols)>0:
            st.dataframe(df_filtered[numeric_cols].describe(), width='stretch')
        else:
            st.info("No hay columnas numéricas.")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>📡 Dashboard actualizado automáticamente cada 60 segundos</p>
    <p>Fuente de datos: EIA Form 930 API | Procesamiento: Pipeline EDA con IsolationForest</p>
    <p>🕐 Historial acumulativo: Últimas 24 horas de actualizaciones (basado en updated_at)</p>
</div>
""", unsafe_allow_html=True)
