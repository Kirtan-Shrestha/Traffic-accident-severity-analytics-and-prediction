"""
Traffic Accident Severity Analytics Dashboard.
Reads from the PostgreSQL analytical mart and fact/dimension tables -
never from manually edited CSV files.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(page_title="UK Traffic Accident Analytics 2023", layout="wide")


@st.cache_resource
def get_engine():
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB")
    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"
    return create_engine(url)


@st.cache_data(ttl=300)
def load_fact_data():
    engine = get_engine()
    query = """
        SELECT
            f.collision_index, f.number_of_vehicles, f.number_of_casualties,
            f.temperature_2m, f.precipitation, f.wind_speed_10m,
            d.full_date, d.year, d.month, d.hour, d.weekday_name, d.is_weekend,
            r.road_type_label, r.speed_limit,
            w.weather_conditions_label, w.temperature_band, w.precipitation_level,
            l.latitude, l.longitude, l.local_authority_district, l.urban_or_rural_label,
            s.severity_label
        FROM fact_accident f
        JOIN dim_date d ON f.date_key = d.date_key
        JOIN dim_road r ON f.road_key = r.road_key
        JOIN dim_weather w ON f.weather_key = w.weather_key
        JOIN dim_location l ON f.location_key = l.location_key
        JOIN dim_severity s ON f.severity_key = s.severity_key
    """
    return pd.read_sql(query, engine)


def main():
    st.title("UK Traffic Accident Severity Analytics — 2023")
    st.caption("Data: UK Road Safety Open Data (STATS19) + Open-Meteo Historical Weather")

    df = load_fact_data()

    # --- Sidebar filters ---
    st.sidebar.header("Filters")

    years = sorted(df["year"].unique())
    selected_years = st.sidebar.multiselect("Year", years, default=years)

    severities = sorted(df["severity_label"].unique())
    selected_severities = st.sidebar.multiselect("Severity", severities, default=severities)

    road_types = sorted(df["road_type_label"].dropna().unique())
    selected_road_types = st.sidebar.multiselect("Road Type", road_types, default=road_types)

    weather_conditions = sorted(df["weather_conditions_label"].dropna().unique())
    selected_weather = st.sidebar.multiselect("Weather", weather_conditions, default=weather_conditions)

    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    selected_weekdays = st.sidebar.multiselect("Weekday", weekdays, default=weekdays)

    filtered = df[
        df["year"].isin(selected_years)
        & df["severity_label"].isin(selected_severities)
        & df["road_type_label"].isin(selected_road_types)
        & df["weather_conditions_label"].isin(selected_weather)
        & df["weekday_name"].isin(selected_weekdays)
    ]

    st.markdown(f"**{len(filtered):,} accidents** match current filters (of {len(df):,} total)")

    # --- KPI row ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Accidents", f"{len(filtered):,}")
    col2.metric("Total Casualties", f"{int(filtered['number_of_casualties'].sum()):,}")
    fatal_count = (filtered["severity_label"] == "Fatal").sum()
    col3.metric("Fatal Accidents", f"{fatal_count:,}")
    avg_casualties = filtered["number_of_casualties"].mean() if len(filtered) else 0
    col4.metric("Avg Casualties/Accident", f"{avg_casualties:.2f}")

    st.divider()

    # --- View 1: Severity distribution ---
    st.subheader("1. Severity Distribution")
    severity_counts = filtered["severity_label"].value_counts().reset_index()
    severity_counts.columns = ["Severity", "Count"]
    fig1 = px.pie(severity_counts, names="Severity", values="Count", hole=0.4)
    st.plotly_chart(fig1, use_container_width=True)

    col_a, col_b = st.columns(2)

    # --- View 2: Accidents by hour ---
    with col_a:
        st.subheader("2. Accidents by Hour of Day")
        by_hour = filtered.groupby("hour").size().reset_index(name="count")
        fig2 = px.bar(by_hour, x="hour", y="count", labels={"hour": "Hour", "count": "Accidents"})
        st.plotly_chart(fig2, use_container_width=True)

    # --- View 3: Accidents by weekday ---
    with col_b:
        st.subheader("3. Accidents by Weekday")
        by_weekday = filtered.groupby("weekday_name").size().reindex(weekdays).reset_index(name="count")
        fig3 = px.bar(by_weekday, x="weekday_name", y="count", labels={"weekday_name": "Day", "count": "Accidents"})
        st.plotly_chart(fig3, use_container_width=True)

    col_c, col_d = st.columns(2)

    # --- View 4: Weather impact ---
    with col_c:
        st.subheader("4. Weather Impact")
        by_weather = filtered.groupby("weather_conditions_label").size().reset_index(name="count")
        by_weather = by_weather.sort_values("count", ascending=False).head(8)
        fig4 = px.bar(by_weather, x="count", y="weather_conditions_label", orientation="h",
                      labels={"weather_conditions_label": "Weather", "count": "Accidents"})
        st.plotly_chart(fig4, use_container_width=True)

    # --- View 5: Road type analysis ---
    with col_d:
        st.subheader("5. Road Type Analysis")
        by_road = filtered.groupby("road_type_label").size().reset_index(name="count")
        by_road = by_road.sort_values("count", ascending=False)
        fig5 = px.bar(by_road, x="count", y="road_type_label", orientation="h",
                      labels={"road_type_label": "Road Type", "count": "Accidents"})
        st.plotly_chart(fig5, use_container_width=True)

    st.divider()

    # --- View 6: Location hotspot map ---
    st.subheader("6. Location Hotspot Map")
    map_sample = filtered.sample(min(5000, len(filtered))) if len(filtered) > 0 else filtered
    fig6 = px.scatter_map(
        map_sample, lat="latitude", lon="longitude",
        color="severity_label", hover_data=["local_authority_district", "road_type_label"],
        zoom=4.5, height=600, map_style="open-street-map",
        center={"lat": 54.5, "lon": -3.0},
    )
    st.plotly_chart(fig6, use_container_width=True)

    st.divider()

    # --- View 7: Top hotspot districts table ---
    st.subheader("7. Top Accident Hotspot Districts")
    hotspots = filtered.groupby("local_authority_district").agg(
        total_accidents=("collision_index", "count"),
        total_casualties=("number_of_casualties", "sum"),
    ).reset_index().sort_values("total_accidents", ascending=False).head(15)
    st.dataframe(hotspots, use_container_width=True)


if __name__ == "__main__":
    main()