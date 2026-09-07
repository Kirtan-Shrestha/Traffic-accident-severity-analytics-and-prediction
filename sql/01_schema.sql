-- =========================================================
-- Traffic Accident Analytics - Star Schema
-- Dimensions: date, road, weather, location, severity
-- Fact: fact_accident
-- =========================================================

CREATE EXTENSION IF NOT EXISTS postgis;

-- ---------------------------------------------------------
-- DIM_DATE
-- ---------------------------------------------------------
CREATE TABLE IF NOT EXISTS dim_date (
    date_key        INTEGER PRIMARY KEY,        -- YYYYMMDD
    full_date       DATE NOT NULL,
    year            SMALLINT NOT NULL,
    month           SMALLINT NOT NULL,
    day             SMALLINT NOT NULL,
    hour            SMALLINT NOT NULL,
    weekday_name    VARCHAR(10) NOT NULL,
    is_weekend      BOOLEAN NOT NULL
);

-- ---------------------------------------------------------
-- DIM_ROAD
-- ---------------------------------------------------------
CREATE TABLE IF NOT EXISTS dim_road (
    road_key        SERIAL PRIMARY KEY,
    road_type_code  SMALLINT,
    road_type_label VARCHAR(50),
    speed_limit     SMALLINT,
    first_road_class SMALLINT,
    UNIQUE (road_type_code, speed_limit, first_road_class)
);

-- ---------------------------------------------------------
-- DIM_WEATHER
-- ---------------------------------------------------------
CREATE TABLE IF NOT EXISTS dim_weather (
    weather_key             SERIAL PRIMARY KEY,
    weather_conditions_code SMALLINT,
    weather_conditions_label VARCHAR(50),
    temperature_2m          NUMERIC(5,2),
    precipitation           NUMERIC(6,2),
    rain                    NUMERIC(6,2),
    snowfall                NUMERIC(6,2),
    cloud_cover             NUMERIC(5,2),
    wind_speed_10m          NUMERIC(6,2),
    relative_humidity_2m    NUMERIC(5,2)
);

-- ---------------------------------------------------------
-- DIM_LOCATION
-- ---------------------------------------------------------
CREATE TABLE IF NOT EXISTS dim_location (
    location_key        SERIAL PRIMARY KEY,
    latitude             NUMERIC(9,6) NOT NULL,
    longitude            NUMERIC(9,6) NOT NULL,
    lat_grid             NUMERIC(4,1),
    lon_grid             NUMERIC(4,1),
    local_authority_district VARCHAR(100),
    police_force         SMALLINT,
    urban_or_rural_label VARCHAR(20),
    geom                 GEOMETRY(Point, 4326),
    UNIQUE (latitude, longitude)
);

-- ---------------------------------------------------------
-- DIM_SEVERITY
-- ---------------------------------------------------------
CREATE TABLE IF NOT EXISTS dim_severity (
    severity_key    SERIAL PRIMARY KEY,
    severity_code   SMALLINT UNIQUE NOT NULL,
    severity_label  VARCHAR(20) NOT NULL
);

-- ---------------------------------------------------------
-- FACT_ACCIDENT
-- ---------------------------------------------------------
CREATE TABLE IF NOT EXISTS fact_accident (
    accident_key        SERIAL PRIMARY KEY,
    collision_index      VARCHAR(20) UNIQUE NOT NULL,
    date_key             INTEGER REFERENCES dim_date(date_key),
    road_key             INTEGER REFERENCES dim_road(road_key),
    weather_key          INTEGER REFERENCES dim_weather(weather_key),
    location_key         INTEGER REFERENCES dim_location(location_key),
    severity_key         INTEGER REFERENCES dim_severity(severity_key),
    number_of_vehicles   SMALLINT,
    number_of_casualties SMALLINT
);

-- Indexes for common analytical query patterns
CREATE INDEX IF NOT EXISTS idx_fact_date ON fact_accident(date_key);
CREATE INDEX IF NOT EXISTS idx_fact_road ON fact_accident(road_key);
CREATE INDEX IF NOT EXISTS idx_fact_weather ON fact_accident(weather_key);
CREATE INDEX IF NOT EXISTS idx_fact_location ON fact_accident(location_key);
CREATE INDEX IF NOT EXISTS idx_fact_severity ON fact_accident(severity_key);
CREATE INDEX IF NOT EXISTS idx_location_geom ON dim_location USING GIST(geom);