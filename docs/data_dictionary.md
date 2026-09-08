# Data Dictionary — Traffic Accident Severity Analytics

## fact_accident

| Field | Type | Description | Source | Example | Nullable | Validation | Layer |
|---|---|---|---|---|---|---|---|
| accident_key | SERIAL | Surrogate primary key | Generated | 1 | No | Auto-increment | fact |
| collision_index | VARCHAR(20) | Natural key from STATS19 | STATS19 collisions | 2023170L30113 | No | Unique, non-null | fact |
| date_key | INTEGER (FK) | Links to dim_date | Derived from collision_datetime | 2023011808 | No | Must exist in dim_date | fact |
| road_key | INTEGER (FK) | Links to dim_road | Derived from road fields | 3 | Yes | Must exist in dim_road | fact |
| weather_key | INTEGER (FK) | Links to dim_weather | Derived from weather fields | 12 | Yes | Must exist in dim_weather | fact |
| location_key | INTEGER (FK) | Links to dim_location | Derived from lat/lon | 45 | Yes | Must exist in dim_location | fact |
| severity_key | INTEGER (FK) | Links to dim_severity | STATS19 collision_severity | 3 | No | Must exist in dim_severity | fact |
| number_of_vehicles | SMALLINT | Vehicles involved | STATS19 collisions | 2 | No | > 0 | fact |
| number_of_casualties | SMALLINT | Casualties resulting | STATS19 collisions | 1 | No | > 0 | fact |
| temperature_2m | NUMERIC(5,2) | Air temp at collision time (°C) | Open-Meteo | 12.6 | Yes | Numeric | fact |
| precipitation | NUMERIC(6,2) | Precipitation (mm) | Open-Meteo | 0.0 | Yes | ≥ 0 | fact |
| wind_speed_10m | NUMERIC(6,2) | Wind speed at 10m (km/h) | Open-Meteo | 15.3 | Yes | ≥ 0 | fact |
| relative_humidity_2m | NUMERIC(5,2) | Relative humidity (%) | Open-Meteo | 72 | Yes | 0-100 | fact |

## dim_date

| Field | Type | Description | Source | Example | Nullable | Validation | Layer |
|---|---|---|---|---|---|---|---|
| date_key | INTEGER (PK) | Surrogate key, format YYYYMMDDHH | Derived | 2023011808 | No | Unique | dimension |
| full_date | DATE | Calendar date | Derived from STATS19 date | 2023-01-18 | No | Valid date | dimension |
| year | SMALLINT | Year | Derived | 2023 | No | - | dimension |
| month | SMALLINT | Month (1-12) | Derived | 1 | No | 1-12 | dimension |
| day | SMALLINT | Day of month | Derived | 18 | No | 1-31 | dimension |
| hour | SMALLINT | Hour of day (0-23) | Derived from STATS19 time | 8 | No | 0-23 | dimension |
| weekday_name | VARCHAR(10) | Day name | Derived | Wednesday | No | - | dimension |
| is_weekend | BOOLEAN | Saturday/Sunday flag | Derived | false | No | - | dimension |

## dim_road

| Field | Type | Description | Source | Example | Nullable | Validation | Layer |
|---|---|---|---|---|---|---|---|
| road_key | SERIAL (PK) | Surrogate key | Generated | 1 | No | Auto-increment | dimension |
| road_type_code | SMALLINT | STATS19 road type code | STATS19 collisions | 6 | Yes | See data guide | dimension |
| road_type_label | VARCHAR(50) | Decoded road type | Decoded via code_lookups.json | Single carriageway | Yes | - | dimension |
| speed_limit | SMALLINT | Posted speed limit (mph) | STATS19 collisions | 30 | Yes | ≥ 0 | dimension |
| first_road_class | SMALLINT | Road classification code | STATS19 collisions | 3 | Yes | See data guide | dimension |

## dim_weather (categorical)

| Field | Type | Description | Source | Example | Nullable | Validation | Layer |
|---|---|---|---|---|---|---|---|
| weather_key | SERIAL (PK) | Surrogate key | Generated | 1 | No | Auto-increment | dimension |
| weather_conditions_code | SMALLINT | STATS19 weather code | STATS19 collisions | 1 | Yes | See data guide | dimension |
| weather_conditions_label | VARCHAR(50) | Decoded weather condition | Decoded via code_lookups.json | Fine no high winds | Yes | - | dimension |
| temperature_band | VARCHAR(20) | Bucketed temperature range | Derived from Open-Meteo temp | 10-20 | Yes | One of: Below 0, 0-10, 10-20, Above 20 | dimension |
| precipitation_level | VARCHAR(20) | Bucketed precipitation level | Derived from Open-Meteo precip | None | Yes | One of: None, Light, Moderate, Heavy | dimension |

*Note: dim_weather intentionally uses banded/categorical values, not exact continuous readings, to remain a genuine low-cardinality dimension (89 rows). Exact temperature/precipitation/wind/humidity values are stored as measures on fact_accident.*

## dim_location

| Field | Type | Description | Source | Example | Nullable | Validation | Layer |
|---|---|---|---|---|---|---|---|
| location_key | SERIAL (PK) | Surrogate key | Generated | 1 | No | Auto-increment | dimension |
| latitude | NUMERIC(9,6) | WGS84 latitude | STATS19 collisions | 54.611942 | No | 49.5-61.0 (UK bounds) | dimension |
| longitude | NUMERIC(9,6) | WGS84 longitude | STATS19 collisions | -1.077081 | No | -8.5-2.0 (UK bounds) | dimension |
| lat_grid | NUMERIC(4,1) | Latitude rounded to 0.3° grid | Derived, used for weather join | 54.6 | Yes | - | dimension |
| lon_grid | NUMERIC(4,1) | Longitude rounded to 0.3° grid | Derived, used for weather join | -1.1 | Yes | - | dimension |
| local_authority_district | VARCHAR(100) | Local authority name | Decoded from local_authority_ons_district via code_lookups.json | Birmingham | Yes | - | dimension |
| police_force | SMALLINT | Police force code | STATS19 collisions | 1 | Yes | See data guide | dimension |
| urban_or_rural_label | VARCHAR(20) | Urban/rural classification | Decoded via code_lookups.json | Urban | Yes | - | dimension |
| geom | GEOMETRY(Point, 4326) | PostGIS point geometry | Derived from lat/lon | POINT(-1.077081 54.611942) | Yes | Valid WGS84 point | dimension |

## dim_severity

| Field | Type | Description | Source | Example | Nullable | Validation | Layer |
|---|---|---|---|---|---|---|---|
| severity_key | SERIAL (PK) | Surrogate key | Generated | 1 | No | Auto-increment | dimension |
| severity_code | SMALLINT | STATS19 severity code | STATS19 collisions | 3 | No | 1, 2, or 3 | dimension |
| severity_label | VARCHAR(20) | Decoded severity | Decoded via code_lookups.json | Slight | No | Fatal, Serious, or Slight | dimension |

## Rejected records log (data/rejected/collisions_rejected_2023.csv)

| Field | Type | Description | Source | Example | Nullable | Validation | Layer |
|---|---|---|---|---|---|---|---|
| record_id | VARCHAR(20) | collision_index of the failing record | Pipeline validation | 2023430039147 | No | - | rejected |
| error_type | VARCHAR(50) | Category of validation failure | Pipeline validation | invalid_latitude | No | One of 7 defined error types | rejected |
| error_message | TEXT | Human-readable failure reason | Pipeline validation | latitude outside expected UK range | No | - | rejected |
| pipeline_timestamp | TIMESTAMP | When the check ran | Pipeline validation | 2026-09-07T11:54:06Z | No | ISO 8601 | rejected |
| source | VARCHAR(255) | Source file that produced the record | Pipeline validation | data/staging/collisions_staged_2023.csv | No | - | rejected |
