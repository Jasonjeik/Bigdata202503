from sqlalchemy import create_engine, text
import os

os.environ['DATABASE_USER'] = 'Proyecto1'
os.environ['DATABASE_PASSWORD'] = 'Proyecto2024@'
os.environ['DATABASE_HOST'] = 'bigdataproyecto1.postgres.database.azure.com'
os.environ['DATABASE_PORT'] = '5432'
os.environ['DATABASE_NAME'] = 'proyecto1'

engine = create_engine(
    f'postgresql+psycopg2://{os.environ["DATABASE_USER"]}:{os.environ["DATABASE_PASSWORD"]}@{os.environ["DATABASE_HOST"]}:{os.environ["DATABASE_PORT"]}/{os.environ["DATABASE_NAME"]}?sslmode=require'
)

conn = engine.connect()

# Check rto_region_data
print("\nColumns in eia.rto_region_data:")
result = conn.execute(text("SELECT column_name, data_type FROM information_schema.columns WHERE table_schema = 'eia' AND table_name = 'rto_region_data' ORDER BY ordinal_position"))
for row in result:
    print(f"  {row[0]}: {row[1]}")

# Check rto_fueltype_data
print("\nColumns in eia.rto_fueltype_data:")
result = conn.execute(text("SELECT column_name, data_type FROM information_schema.columns WHERE table_schema = 'eia' AND table_name = 'rto_fueltype_data' ORDER BY ordinal_position"))
for row in result:
    print(f"  {row[0]}: {row[1]}")

# Check rto_interchange_data
print("\nColumns in eia.rto_interchange_data:")
result = conn.execute(text("SELECT column_name, data_type FROM information_schema.columns WHERE table_schema = 'eia' AND table_name = 'rto_interchange_data' ORDER BY ordinal_position"))
for row in result:
    print(f"  {row[0]}: {row[1]}")

conn.close()
