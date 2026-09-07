# Dataset Information

## Source
UK Road Safety Open Data (STATS19), published by the Department for Transport (DfT).

- Official page: https://www.gov.uk/government/statistical-data-sets/road-safety-open-data
- License: Open Government Licence v3.0 (free to use, requires attribution)
- Data guide (field code lookups): `docs/data_guide.xlsx`

## Year selected
2023 (full year, final validated data)

Rationale: 2023 is a complete, validated year not affected by the junction_detail (Nov 2025)
or vehicle_location_restricted_lane (Jul 2026) data quality issues that affected 2024/2025
files. A single year (~100k+ records) is sufficient to demonstrate the full pipeline while
keeping processing time and project scope manageable within the assignment timeline.

## Files acquired

| File | Rows (approx.) | Size | Description |
|---|---|---|---|
| collisions_2023.csv | ~100,000+ | 19.8 MB | One row per accident/collision |
| vehicles_2023.csv | ~180,000+ | 20.9 MB | One row per vehicle involved in a collision |
| casualties_2023.csv | ~130,000+ | 10.7 MB | One row per casualty from a collision |

All files stored in `data/raw/`, unmodified from their original download. These files are
never manually edited — all transformations happen through the pipeline scripts.

## Download date
7 September 2026

## Download method
Direct HTTPS download from data.dft.gov.uk (see `src/ingestion/` for the reproducible
download script, added in Phase 2).

## Key fields (to be expanded into full data dictionary later)
- **collisions_2023.csv**: accident_index, accident_severity, date, time, longitude,
  latitude, road_type, weather_conditions, light_conditions, number_of_vehicles,
  number_of_casualties
- **vehicles_2023.csv**: accident_index, vehicle_type, vehicle_manoeuvre
- **casualties_2023.csv**: accident_index, casualty_class, casualty_severity, age_of_casualty

Field codes are integers; see `docs/data_guide.xlsx` for the full lookup tables
(e.g. accident_severity: 1=Fatal, 2=Serious, 3=Slight).

## Second data source (Phase 3)
Weather enrichment will use the Open-Meteo Historical Weather API
(https://open-meteo.com/en/docs/historical-weather-api), joined to collision records by
date/time and latitude/longitude. No API key required; free for non-commercial use.