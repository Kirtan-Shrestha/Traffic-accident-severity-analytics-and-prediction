CREATE TABLE IF NOT EXISTS ml_accident_features AS
SELECT
    f.accident_key, f.collision_index,
    d.year, d.month, d.hour, d.weekday_name, d.is_weekend,
    r.road_type_label, r.speed_limit,
    w.weather_conditions_label, w.temperature_band, w.precipitation_level,
    l.urban_or_rural_label, l.latitude, l.longitude,
    f.number_of_vehicles, f.temperature_2m, f.precipitation, f.wind_speed_10m,
    s.severity_label AS target_severity,
    CASE
        WHEN d.month <= 8 THEN 'train'
        WHEN d.month IN (9,10) THEN 'validation'
        ELSE 'test'
    END AS split
FROM fact_accident f
JOIN dim_date d ON f.date_key = d.date_key
JOIN dim_road r ON f.road_key = r.road_key
JOIN dim_weather w ON f.weather_key = w.weather_key
JOIN dim_location l ON f.location_key = l.location_key
JOIN dim_severity s ON f.severity_key = s.severity_key;

CREATE INDEX IF NOT EXISTS idx_ml_split ON ml_accident_features(split);