# ⚡ PROYECTO 1 - GUÍA COMPLETA DE DESPLIEGUE

## Monitoreo en Tiempo Real de la Red Eléctrica EIA con Machine Learning

Este documento describe paso a paso cómo ejecutar el proyecto completo desde cero.

---

## 📋 TABLA DE CONTENIDOS

1. [Requisitos Previos](#requisitos)
2. [Instalación de Dependencias](#instalacion)
3. [Configuración de la Base de Datos](#configuracion-db)
4. [Ingesta Histórica de Datos (2022-presente)](#ingesta-historica)
5. [Ingesta Incremental en Tiempo Real](#ingesta-incremental)
6. [Entrenamiento del Modelo de Predicción](#ml-training)
7. [Despliegue del Dashboard Streamlit](#streamlit-deploy)
8. [Solución de Problemas](#troubleshooting)

---

<a id="requisitos"></a>
## 1️⃣ REQUISITOS PREVIOS

### Software Requerido
- Python 3.10 o superior
- Acceso a internet (para API de EIA)
- PostgreSQL en Azure (ya configurado)

### Credenciales
- API Key de EIA: `YOAXAO8j6vBYbOPycDJH0yCfwWgzpK94LQLaZ1hT`
- PostgreSQL: `bigdataproyecto1.postgres.database.azure.com`
- Usuario: `prj1_admin`
- Base de datos: `proyecto1`

---

<a id="instalacion"></a>
## 2️⃣ INSTALACIÓN DE DEPENDENCIAS

Abre una terminal en la carpeta del proyecto y ejecuta:

```bash
# Librerías básicas
pip install pandas numpy requests sqlalchemy psycopg2-binary

# Visualización
pip install streamlit plotly

# Machine Learning
pip install scikit-learn prophet

# Opcional (para notebooks)
pip install jupyter notebook
```

**Verificar instalación:**
```bash
python -c "import pandas, streamlit, prophet; print('✅ Todas las librerías instaladas correctamente')"
```

---

<a id="configuracion-db"></a>
## 3️⃣ CONFIGURACIÓN DE LA BASE DE DATOS

### 3.1 Verificar Conexión

Ejecuta el siguiente script para verificar conectividad:

```python
import psycopg2

conn_params = {
    'host': 'bigdataproyecto1.postgres.database.azure.com',
    'port': 5432,
    'database': 'proyecto1',
    'user': 'prj1_admin',
    'password': 'Bigdataproyecto1',
    'sslmode': 'require'
}

try:
    conn = psycopg2.connect(**conn_params)
    print("✅ Conexión exitosa a PostgreSQL")
    conn.close()
except Exception as e:
    print(f"❌ Error de conexión: {e}")
```

### 3.2 Crear Schema

El script de ingesta creará automáticamente el schema `eia` si no existe. Pero puedes verificarlo ejecutando:

```python
from sqlalchemy import create_engine, text

engine = create_engine(
    "postgresql+psycopg2://prj1_admin:Bigdataproyecto1@bigdataproyecto1.postgres.database.azure.com:5432/proyecto1",
    connect_args={"sslmode": "require"}
)

with engine.begin() as conn:
    conn.execute(text("CREATE SCHEMA IF NOT EXISTS eia;"))
    print("✅ Schema 'eia' creado/verificado")
```

---

<a id="ingesta-historica"></a>
## 4️⃣ INGESTA HISTÓRICA DE DATOS (PRIMERA EJECUCIÓN)

Esta carga inicial descargará todos los datos desde **2022-01-01** hasta el presente. Puede tardar **30-60 minutos** dependiendo de tu conexión.

### 4.1 Ejecutar Carga Histórica

```bash
python ingesta_eia_incremental.py --historical --once
```

**Parámetros:**
- `--historical`: Carga desde 2022-01-01 (en lugar de solo últimas 48 horas)
- `--once`: Ejecuta una sola vez y sale (sin bucle continuo)

### 4.2 Qué Hace Este Proceso

1. **Descarga de 3 endpoints EIA:**
   - `region-data`: Demanda, generación neta, intercambio de transmisión
   - `fuel-type-data`: Generación por tipo de combustible (gas, carbón, nuclear, renovables, etc.)
   - `interchange-data`: Intercambios de energía entre regiones

2. **Almacenamiento en tablas raw:**
   - `eia.rto_region_data`
   - `eia.rto_fueltype_data`
   - `eia.rto_interchange_data`

3. **Procesamiento EDA:**
   - Limpieza de datos
   - Detección de anomalías con Isolation Forest
   - Agregación por región y minuto
   - Almacenamiento en `eia.eia_aggregated_realtime`

### 4.3 Verificar Datos Cargados

```python
from sqlalchemy import create_engine
import pandas as pd

engine = create_engine(
    "postgresql+psycopg2://prj1_admin:Bigdataproyecto1@bigdataproyecto1.postgres.database.azure.com:5432/proyecto1",
    connect_args={"sslmode": "require", "options": "-csearch_path=eia"}
)

# Verificar cantidad de registros
tables = ['rto_region_data', 'rto_fueltype_data', 'rto_interchange_data', 'eia_aggregated_realtime']

for table in tables:
    df = pd.read_sql(f"SELECT COUNT(*) as total, MIN(period) as min_date, MAX(period) as max_date FROM {table}", engine)
    print(f"\n📊 {table}:")
    print(f"   Total registros: {df['total'][0]:,}")
    print(f"   Período: {df['min_date'][0]} → {df['max_date'][0]}")
```

**Salida esperada:**
```
📊 rto_region_data:
   Total registros: ~3,500,000
   Período: 2022-01-01 00:00:00+00 → 2025-11-14 17:00:00+00

📊 rto_fueltype_data:
   Total registros: ~5,000,000
   Período: 2022-01-01 00:00:00+00 → 2025-11-14 17:00:00+00

... (etc)
```

---

<a id="ingesta-incremental"></a>
## 5️⃣ INGESTA INCREMENTAL EN TIEMPO REAL

Una vez cargados los datos históricos, ejecuta el script en modo continuo para mantener los datos actualizados cada 60 segundos.

### 5.1 Iniciar Proceso de Streaming

```bash
python ingesta_eia_incremental.py
```

**Sin parámetros**, el script:
- Carga solo las **últimas 48 horas** (modo incremental)
- Se ejecuta en **bucle continuo** cada 60 segundos
- Actualiza automáticamente la tabla agregada

### 5.2 Dejar Ejecutándose en Segundo Plano

**Opción 1: Terminal persistente (recomendado para desarrollo)**
- Mantén la terminal abierta
- Para detener: `Ctrl+C`

**Opción 2: Windows Task Scheduler (producción)**
1. Abre Task Scheduler
2. Crear tarea básica
3. Trigger: Al iniciar sesión (o diariamente)
4. Acción: Ejecutar `python ingesta_eia_incremental.py`
5. Configurar reinicio automático en caso de error

**Opción 3: Usar `nohup` (Linux/Mac) o `start` (Windows)**
```bash
# Windows
start /B python ingesta_eia_incremental.py

# Linux/Mac
nohup python ingesta_eia_incremental.py &
```

### 5.3 Monitorear el Proceso

Abre otra terminal y verifica que los datos se están actualizando:

```bash
python -c "from sqlalchemy import create_engine; import pandas as pd; engine = create_engine('postgresql+psycopg2://prj1_admin:Bigdataproyecto1@bigdataproyecto1.postgres.database.azure.com:5432/proyecto1', connect_args={'sslmode': 'require', 'options': '-csearch_path=eia'}); df = pd.read_sql('SELECT MAX(updated_at) as last_update FROM eia_aggregated_realtime', engine); print(f'🕐 Última actualización: {df[\"last_update\"][0]}')"
```

---

<a id="ml-training"></a>
## 6️⃣ ENTRENAMIENTO DEL MODELO DE PREDICCIÓN

El modelo de Machine Learning se entrena **bajo demanda** desde la interfaz de Streamlit. Pero puedes probarlo independientemente:

### 6.1 Entrenar y Generar Predicción

```python
from prediction_model import forecast_demand_pipeline

# Predicción para California (CISO)
forecast, model, metrics = forecast_demand_pipeline(
    region_code='CISO',
    days_back=180,       # Usar últimos 180 días para entrenamiento
    forecast_hours=24,   # Predecir próximas 24 horas
    use_raw_data=True    # Usar tabla rto_region_data (más datos históricos)
)

print("\n📈 Métricas del modelo:")
print(f"   MAE: {metrics['MAE']:.2f} MW")
print(f"   RMSE: {metrics['RMSE']:.2f} MW")
print(f"   MAPE: {metrics['MAPE']:.2f}%")

print("\n🔮 Primeras 5 predicciones:")
print(forecast.head())

# Guardar predicciones
forecast.to_csv("prediccion_ciso_24h.csv", index=False)
print("\n💾 Predicciones guardadas en: prediccion_ciso_24h.csv")
```

### 6.2 Probar con Todas las Regiones Agregadas

```python
forecast_all, model_all, metrics_all = forecast_demand_pipeline(
    region_code=None,     # None = agregar todas las regiones
    days_back=180,
    forecast_hours=24,
    use_raw_data=True
)
```

### 6.3 Interpretar las Métricas

- **MAE (Mean Absolute Error)**: Error promedio en MW. Mientras más bajo, mejor.
- **RMSE (Root Mean Squared Error)**: Penaliza errores grandes. Más bajo es mejor.
- **MAPE (Mean Absolute Percentage Error)**: Error porcentual. Ideal < 10%.

**Ejemplo de métricas aceptables:**
```
MAE:  1,234.56 MW      (< 5% de demanda promedio)
RMSE: 1,890.12 MW      
MAPE: 4.32%            (< 10% = bueno)
```

---

<a id="streamlit-deploy"></a>
## 7️⃣ DESPLIEGUE DEL DASHBOARD STREAMLIT

### 7.1 Iniciar Streamlit

Abre una **nueva terminal** (la otra debe tener el script de ingesta ejecutándose) y ejecuta:

```bash
streamlit run app.py
```

**Salida esperada:**
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.X:8501
```

### 7.2 Acceder al Dashboard

1. Abre tu navegador en: `http://localhost:8501`
2. Deberías ver el dashboard con:
   - ✅ Métricas en tiempo real (demanda, generación, déficit, anomalías)
   - 📈 Gráfico temporal de evolución del déficit
   - 🗺️ Mapa interactivo con polígonos de regiones
   - 🔮 Sección de predicción con ML
   - 📊 Visualización de fuel mix

### 7.3 Usar la Sección de Predicción

1. Scroll down hasta **"🔮 Predicción de Demanda - Próximas 24 Horas"**
2. Selecciona una región (ej: `CISO`, `MISO`, `NYIS`, etc.) o `TODAS`
3. Haz clic en **"🤖 Generar Predicción"**
4. Espera 1-2 minutos mientras:
   - Se cargan datos históricos de 180 días
   - Se entrena el modelo Prophet
   - Se generan predicciones para las próximas 24 horas
5. Visualiza:
   - Curva de predicción con intervalos de confianza
   - Métricas de precisión del modelo
   - Tabla detallada de predicciones horarias
   - Botón para descargar predicciones en CSV

### 7.4 Funcionalidades del Dashboard

**Filtros en Sidebar:**
- **Región**: Filtrar por región específica o ver todas
- **Horas hacia atrás**: Ventana temporal (1-672 horas)
- **Mostrar solo anomalías**: Ver únicamente eventos anómalos

**Mapa Interactivo:**
- Polígonos coloreados:
  - 🟢 Verde = Estado normal
  - 🔴 Rojo = Anomalía detectada
- Hover muestra:
  - Nombre de región
  - Demanda, generación, déficit
  - Hora del evento
  - Última actualización

**Auto-refresh:**
- La página se actualiza automáticamente cada 60 segundos
- O puedes usar el botón "🔄 Actualizar Datos"

---

<a id="troubleshooting"></a>
## 8️⃣ SOLUCIÓN DE PROBLEMAS

### ❌ "No se pudo establecer conexión con la base de datos"

**Causa:** Tu IP no está permitida en el firewall de Azure PostgreSQL.

**Solución:**
1. Ve al Azure Portal → PostgreSQL Server → Connection Security
2. Agrega tu IP pública actual
3. Guarda cambios
4. Reinicia el script

---

### ❌ "Error al conectar con la base de datos o la tabla no existe"

**Causa:** El script de ingesta no ha completado el primer ciclo.

**Solución:**
1. Verifica que `ingesta_eia_incremental.py` esté ejecutándose
2. Espera 1-2 minutos para que se cree `eia_aggregated_realtime`
3. Recarga el dashboard (F5)

---

### ❌ "Datos insuficientes para entrenar modelo (mínimo 48 observaciones)"

**Causa:** No hay suficientes datos históricos en la base de datos.

**Solución:**
1. Ejecuta la carga histórica: `python ingesta_eia_incremental.py --historical --once`
2. Espera a que complete (puede tardar 30-60 minutos)
3. Intenta generar predicción nuevamente

---

### ❌ "Prophet no está instalado"

**Causa:** Falta la librería Prophet.

**Solución:**
```bash
pip install prophet
```

Si falla la instalación (especialmente en Windows):
```bash
# Instalar dependencias C++ necesarias
pip install pystan
pip install prophet
```

---

### ❌ Dashboard muestra "No hay datos en el rango de tiempo seleccionado"

**Causa:** Filtro temporal demasiado estrecho o datos no llegaron aún.

**Solución:**
1. Aumenta "Horas hacia atrás" en el sidebar (ej: 168 horas = 1 semana)
2. Verifica que el script de ingesta esté ejecutándose
3. Consulta la base directamente para ver qué datos hay disponibles

---

### ❌ Proceso de ingesta se detiene con errores de timeout

**Causa:** Conexión interrumpida o API de EIA sobrecargada.

**Solución:**
- El script tiene reintentos automáticos, pero si falla constantemente:
1. Verifica tu conexión a internet
2. Espera unos minutos (la API puede estar temporalmente no disponible)
3. Reinicia el script

---

## 📚 ARCHIVOS DEL PROYECTO

| Archivo | Descripción |
|---------|-------------|
| `ingesta_eia_incremental.py` | Script principal de ingesta y EDA |
| `eda_utils.py` | Funciones de limpieza, agregación y anomalías |
| `prediction_model.py` | Modelo Prophet para forecasting |
| `app.py` | Dashboard Streamlit |
| `Proyecto1_Stream.ipynb` | Notebook con la historia completa |
| `README.md` | Esta guía |

---

## 🎓 CONCEPTOS CLAVE

### Tablas en PostgreSQL

1. **rto_region_data** (raw)
   - Datos crudos de demanda (D), generación neta (NG), intercambio de transmisión (TI)
   - Granularidad: horaria
   - Índice: (period, respondent, type)

2. **rto_fueltype_data** (raw)
   - Generación por tipo de combustible
   - Tipos: COL, NG, NUC, SUN, WND, WAT, etc.
   - Índice: (period, respondent, fueltype)

3. **rto_interchange_data** (raw)
   - Intercambios de energía entre regiones
   - Índice: (period, fromba, toba)

4. **eia_aggregated_realtime** (procesada)
   - Datos agregados con EDA aplicado
   - Incluye: demand_mw, generation_mw, deficit, anomaly score
   - Ventana: últimas 24 horas (basado en updated_at)
   - Acumulativo: conserva múltiples actualizaciones por periodo

### Pipeline EDA

1. **Limpieza**: Elimina filas con >80% missing values, imputa medianas
2. **Detección de outliers**: Z-score y IQR
3. **Agregación**: Pivotea por región, calcula totales y déficit
4. **Anomalías**: Isolation Forest sobre columna `deficit`
5. **Enriquecimiento**: Agrega coordenadas geográficas y geocercas

### Modelo de Predicción

- **Algoritmo**: Prophet (Facebook)
- **Features**: Histórico de demanda + estacionalidad (anual, semanal, diaria)
- **Output**: Predicción puntual + intervalos de confianza (95%)
- **Reentrenamiento**: On-demand desde la UI

---

## 🚀 PRÓXIMOS PASOS

Una vez que todo esté funcionando:

1. **Automatizar** el script de ingesta con Task Scheduler o cron
2. **Optimizar** consultas SQL con índices adicionales si el dashboard es lento
3. **Expandir** el modelo de ML para incluir variables exógenas (clima, eventos, etc.)
4. **Escalar** usando contenedores Docker y deploy en cloud (Azure App Service, AWS ECS, etc.)
5. **Agregar** alertas automáticas cuando se detecten anomalías críticas

---

## 📞 SOPORTE

Si encuentras problemas no cubiertos en esta guía:

1. Revisa los logs del script de ingesta
2. Verifica conectividad con `psql` o pgAdmin
3. Consulta la documentación de la API de EIA: https://www.eia.gov/opendata/

---

**✅ ¡Proyecto completamente funcional y documentado!**
