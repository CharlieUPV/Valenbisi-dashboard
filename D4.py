import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import openrouteservice as ors
from sklearn.linear_model import LinearRegression
import numpy as np
import time
import json
from shapely.geometry import LineString

st.set_page_config(page_title="Valenbisi + Traffic Routes", layout="wide")

@st.cache_data
def load_valenbisi():
    df = pd.read_csv("valenbisi.csv", sep=";")
    gdf = gpd.read_file("valenbisi.geojson")
    return df, gdf

@st.cache_data(ttl=180)
def load_traffic():
    url = "https://valencia.opendatasoft.com/explore/dataset/estat-transit-temps-real-estado-trafico-tiempo-real/download/?format=csv"
    return pd.read_csv(url, sep=";")

@st.cache_data
def load_traffic_from_csv():
    df_traffic = pd.read_csv("trafico8.csv", sep=";")
    df_traffic = df_traffic.dropna(subset=['geo_shape'])
    geometries = []
    for raw in df_traffic['geo_shape']:
        try:
            coords = json.loads(raw)['coordinates']
            geom = LineString(coords)
            geometries.append(geom)
        except:
            geometries.append(None)
    gdf_traffic = gpd.GeoDataFrame(df_traffic, geometry=geometries, crs="EPSG:4326")
    return gdf_traffic.dropna(subset=["geometry"])

@st.cache_resource
def train_model():
    np.random.seed(42)
    data = pd.DataFrame({
        'available_bikes': np.random.randint(0, 20, 200),
        'free_slots': np.random.randint(0, 20, 200),
        'hour': np.random.randint(0, 24, 200),
    })
    data['target'] = data['available_bikes'] + np.random.normal(0, 2, 200)
    model = LinearRegression()
    model.fit(data[['available_bikes', 'free_slots', 'hour']], data['target'])
    return model

@st.cache_data(ttl=3600)
def get_route(_client, coords, profile):
    return _client.directions(coords, profile=profile, format='geojson')

df, gdf = load_valenbisi()
df.columns = df.columns.str.lower()
gdf.columns = gdf.columns.str.lower()
df['numero'] = df['numero'].astype(str)
gdf['number'] = gdf['number'].astype(str)
merged = pd.merge(gdf, df, left_on='number', right_on='numero', how='inner')
traffic = load_traffic()
model = train_model()

tab1, tab2, tab3 = st.tabs(["🚚 Stations", "📍 Routes + Traffic", "🤖 Prediction"])

with tab1:
    st.sidebar.header("Filters and search")
    search_text = st.sidebar.text_input("Search station by name")
    min_bikes = st.sidebar.slider("🚲 Min available bikes", 0, int(merged['bicis_disponibles'].max()), 0)
    min_slots = st.sidebar.slider("🄹 Min available slots", 0, int(merged['espacios_libres'].max()), 0)
    status_filter = st.sidebar.selectbox("🔌 Active stations", ["All", "Yes", "No"])

    filtered = merged[
        (merged['bicis_disponibles'] >= min_bikes) &
        (merged['espacios_libres'] >= min_slots)
    ]
    if status_filter == "Yes":
        filtered = filtered[filtered['activo'] == "T"]
    elif status_filter == "No":
        filtered = filtered[filtered['activo'] == "F"]
    if search_text:
        filtered = filtered[filtered['direccion'].str.contains(search_text, case=False, na=False)]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Stations displayed", filtered.shape[0])
    col2.metric("Total available bikes", int(filtered['bicis_disponibles'].sum()))
    col3.metric("Total free slots", int(filtered['espacios_libres'].sum()))
    occupancy = 1 - (filtered['espacios_libres'].sum() / filtered['espacios_totales'].sum()) if filtered.shape[0] > 0 else 0
    col4.metric("Average occupancy", f"{occupancy:.1%}")

    m = folium.Map(location=[39.4699, -0.3763], zoom_start=13)
    marker_cluster = MarkerCluster().add_to(m)

    for _, row in filtered.iterrows():
        lon, lat = row['geometry'].coords[0]
        popup = f"""
        <b>{row['direccion']}</b><br>
        🚲 Bikes: {row['bicis_disponibles']}<br>
        🄹 Free slots: {row['espacios_libres']}<br>
        ♻️ Total: {row['espacios_totales']}<br>
        ⏱️ Last update: {row['fecha_actualizacion']}
        """
        folium.Marker(
            location=[lat, lon],
            popup=popup,
            icon=folium.Icon(color="green" if row['bicis_disponibles'] > 0 else "red", icon="bicycle", prefix="fa")
        ).add_to(marker_cluster)

    st.subheader("📹 Map of Valenbisi stations with clusters")
    st_folium(m, width=1000, height=600)

    with st.expander("📋 View filtered stations table"):
        display_cols = ['direccion', 'numero', 'bicis_disponibles', 'espacios_libres', 'activo', 'fecha_actualizacion']
        st.dataframe(filtered[display_cols].reset_index(drop=True))

    with st.sidebar:
        if 'refresh' not in st.session_state:
            st.session_state.refresh = False
        refresh = st.checkbox("🔄 Auto-refresh every 60 seconds", value=st.session_state.refresh, key="auto_refresh_checkbox")
        if refresh:
            stop_button = st.button("⏹️ Stop Auto-refresh")
            if stop_button:
                st.session_state.refresh = False
                st.rerun()
        else:
            st.session_state.refresh = False

    if st.session_state.refresh:
        time_left = 60
        placeholder = st.empty()
        while st.session_state.refresh and time_left > 0:
            placeholder.write(f"🔄 Refreshing in {time_left} seconds...")
            time.sleep(1)
            time_left -= 1
        placeholder.empty()
        if st.session_state.refresh:
            st.rerun()

# (TAB 2 y TAB 3 se continúan con la misma lógica de traducción)
# Si quieres, te completo el resto también con las pestañas de Rutas y Predicción.
