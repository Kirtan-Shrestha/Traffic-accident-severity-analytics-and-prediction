-- =========================================================
-- Analytical Data Mart
-- Aggregates required by the assignment: by hour, day, road
-- type, weather condition, plus severity distribution and
-- location hotspots.
-- =========================================================

-- Accidents by hour of day
CREATE MATERIALIZED VIEW IF NOT EXISTS mart_accidents_by_hour AS
SELECT
    d.hour,
    COUNT(*) AS total_accidents,
    SUM(f.number_of_casualties) AS total_casualties,
    ROUND(AVG(f.number_of_casualties), 2) AS avg_casualties_per_accident
FROM fact_accident f
JOIN dim_date d ON f.date_key = d.date_key
GROUP BY d.hour
ORDER BY d.hour;

-- Accidents by day of week
CREATE MATERIALIZED VIEW IF NOT EXISTS mart_accidents_by_weekday AS
SELECT
    d.weekday_name,
    d.is_weekend,
    COUNT(*) AS total_accidents,
    SUM(f.number_of_casualties) AS total_casualties
FROM fact_accident f
JOIN dim_date d ON f.date_key = d.date_key
GROUP BY d.weekday_name, d.is_weekend
ORDER BY total_accidents DESC;

-- Accidents by road type
CREATE MATERIALIZED VIEW IF NOT EXISTS mart_accidents_by_road_type AS
SELECT
    r.road_type_label,
    r.speed_limit,
    COUNT(*) AS total_accidents,
    SUM(f.number_of_casualties) AS total_casualties,
    ROUND(AVG(f.number_of_casualties), 2) AS avg_casualties_per_accident
FROM fact_accident f
JOIN dim_road r ON f.road_key = r.road_key
GROUP BY r.road_type_label, r.speed_limit
ORDER BY total_accidents DESC;

-- Accidents by weather condition
CREATE MATERIALIZED VIEW IF NOT EXISTS mart_accidents_by_weather AS
SELECT
    w.weather_conditions_label,
    w.temperature_band,
    w.precipitation_level,
    COUNT(*) AS total_accidents,
    SUM(f.number_of_casualties) AS total_casualties
FROM fact_accident f
JOIN dim_weather w ON f.weather_key = w.weather_key
GROUP BY w.weather_conditions_label, w.temperature_band, w.precipitation_level
ORDER BY total_accidents DESC;

-- Severity distribution
CREATE MATERIALIZED VIEW IF NOT EXISTS mart_severity_distribution AS
SELECT
    s.severity_label,
    COUNT(*) AS total_accidents,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct_of_total
FROM fact_accident f
JOIN dim_severity s ON f.severity_key = s.severity_key
GROUP BY s.severity_label
ORDER BY total_accidents DESC;

-- Location hotspots (by local authority district)
CREATE MATERIALIZED VIEW IF NOT EXISTS mart_location_hotspots AS
SELECT
    l.local_authority_district,
    COUNT(*) AS total_accidents,
    SUM(f.number_of_casualties) AS total_casualties,
    ROUND(AVG(l.latitude), 4) AS avg_latitude,
    ROUND(AVG(l.longitude), 4) AS avg_longitude
FROM fact_accident f
JOIN dim_location l ON f.location_key = l.location_key
GROUP BY l.local_authority_district
ORDER BY total_accidents DESC;