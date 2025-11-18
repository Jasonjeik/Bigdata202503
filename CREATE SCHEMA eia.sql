CREATE SCHEMA eia
    AUTHORIZATION prj1_admin;


CREATE TABLE IF NOT EXISTS eia.eia_aggregate_realtime
(
    period timestamp with time zone,
    region text COLLATE pg_catalog."default",
    demand_mw double precision,
    generation_mw double precision,
    "TI" double precision,
    deficit double precision,
    fuel_bat_mw double precision,
    fuel_col_mw double precision,
    fuel_geo_mw double precision,
    fuel_ng_mw double precision,
    fuel_nuc_mw double precision,
    fuel_oes_mw double precision,
    fuel_oil_mw double precision,
    fuel_oth_mw double precision,
    fuel_ps_mw double precision,
    fuel_snb_mw double precision,
    fuel_sun_mw double precision,
    fuel_ues_mw double precision,
    fuel_unk_mw double precision,
    fuel_wat_mw double precision,
    fuel_wnb_mw double precision,
    fuel_wnd_mw double precision,
    energy_sent text COLLATE pg_catalog."default",
    energy_received text COLLATE pg_catalog."default",
    net_exchange double precision,
    deficit_pct_demand double precision,
    lat double precision,
    lon double precision,
    boundary text COLLATE pg_catalog."default",
    region_name text COLLATE pg_catalog."default",
    outlier_score double precision,
    is_outlier bigint,
    anomaly bigint,
    anomaly_score double precision,
    deficit_rank bigint,
    updated_at timestamp with time zone
)

TABLESPACE pg_default;

ALTER TABLE eia.eia_aggregate_realtime
    OWNER to prj1_admin;

CREATE TABLE IF NOT EXISTS eia.rto_fueltype_data
(
    period timestamp with time zone,
    respondent text COLLATE pg_catalog."default",
    "respondent-name" text COLLATE pg_catalog."default",
    fueltype text COLLATE pg_catalog."default",
    "type-name" text COLLATE pg_catalog."default",
    value numeric,
    "value-units" text COLLATE pg_catalog."default"
)

TABLESPACE pg_default;

ALTER TABLE eia.rto_fueltype_data
    OWNER to prj1_admin;

-- Index: eia.idx_rto_fueltype_data_period
CREATE INDEX IF NOT EXISTS idx_rto_fueltype_data_period
    ON eia.rto_fueltype_data USING btree
    (period ASC NULLS LAST)
    TABLESPACE pg_default;
-- Index: eia.idx_rto_fueltype_data_value
CREATE INDEX IF NOT EXISTS idx_rto_fueltype_data_value
    ON eia.rto_fueltype_data USING btree
    (value ASC NULLS LAST)
    TABLESPACE pg_default;
-- Index: eia.ux_rto_fueltype_data
CREATE UNIQUE INDEX IF NOT EXISTS ux_rto_fueltype_data
    ON eia.rto_fueltype_data USING btree
    (period ASC NULLS LAST, respondent COLLATE pg_catalog."default" ASC NULLS LAST, fueltype COLLATE pg_catalog."default" ASC NULLS LAST)
    TABLESPACE pg_default;


    CREATE TABLE IF NOT EXISTS eia.rto_interchange_data
(
    period timestamp with time zone,
    fromba text COLLATE pg_catalog."default",
    "fromba-name" text COLLATE pg_catalog."default",
    toba text COLLATE pg_catalog."default",
    "toba-name" text COLLATE pg_catalog."default",
    value numeric,
    "value-units" text COLLATE pg_catalog."default"
)

TABLESPACE pg_default;

ALTER TABLE eia.rto_interchange_data
    OWNER to prj1_admin;

-- Index: eia.idx_rto_interchange_data_period
CREATE INDEX IF NOT EXISTS idx_rto_interchange_data_period
    ON eia.rto_interchange_data USING btree
    (period ASC NULLS LAST)
    TABLESPACE pg_default;
-- Index: eia.idx_rto_interchange_data_value
CREATE INDEX IF NOT EXISTS idx_rto_interchange_data_value
    ON eia.rto_interchange_data USING btree
    (value ASC NULLS LAST)
    TABLESPACE pg_default;
-- Index: eia.ux_rto_interchange_data
CREATE UNIQUE INDEX IF NOT EXISTS ux_rto_interchange_data
    ON eia.rto_interchange_data USING btree
    (period ASC NULLS LAST, fromba COLLATE pg_catalog."default" ASC NULLS LAST, toba COLLATE pg_catalog."default" ASC NULLS LAST)
    TABLESPACE pg_default;

    CREATE TABLE IF NOT EXISTS eia.rto_region_data
(
    period timestamp with time zone,
    respondent text COLLATE pg_catalog."default",
    "respondent-name" text COLLATE pg_catalog."default",
    type text COLLATE pg_catalog."default",
    "type-name" text COLLATE pg_catalog."default",
    value numeric,
    "value-units" text COLLATE pg_catalog."default"
)

TABLESPACE pg_default;

ALTER TABLE eia.rto_region_data
    OWNER to prj1_admin;

-- Index: eia.idx_rto_region_data_period
CREATE INDEX IF NOT EXISTS idx_rto_region_data_period
    ON eia.rto_region_data USING btree
    (period ASC NULLS LAST)
    TABLESPACE pg_default;
-- Index: eia.idx_rto_region_data_value
CREATE INDEX IF NOT EXISTS idx_rto_region_data_value
    ON eia.rto_region_data USING btree
    (value ASC NULLS LAST)
    TABLESPACE pg_default;
-- Index: eia.ux_rto_region_data
CREATE UNIQUE INDEX IF NOT EXISTS ux_rto_region_data
    ON eia.rto_region_data USING btree
    (period ASC NULLS LAST, respondent COLLATE pg_catalog."default" ASC NULLS LAST, type COLLATE pg_catalog."default" ASC NULLS LAST)
    TABLESPACE pg_default;