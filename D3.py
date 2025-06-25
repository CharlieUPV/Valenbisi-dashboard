# valenbisi_dashboard.py
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
from shapely.geometry import LineString, Point

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Valenbisi + Tráfico + Transporte", layout="wide")

# --- FUNCIONES ---
@st.cache_data
def load_valenbisi():
    df = pd.read_csv("valenbisi.csv", sep=";")
    gdf = gpd.read_file("valenbisi.geojson")
    return df, gdf

@st.cache_data
def load_traffic_csv():
    df = pd.read_csv("trafico.csv", sep=";")
    df = df.dropna(subset=['geo_shape'])
    geometries = []
    for raw in df['geo_shape']:
        try:
            coords = json.loads(raw)['coordinates']
            geom = LineString(coords)
            geometries.append(geom)
        except:
            geometries.append(None)
    gdf = gpd.GeoDataFrame(df, geometry=geometries, crs="EPSG:4326")
    return gdf.dropna(subset=["geometry"])

@st.cache_data
def load_transporte():
    df = pd.read_csv("transporte.csv", sep=";")
    df = df.dropna(subset=['geo_shape'])
    geometries = []
    for raw in df['geo_shape']:
        try:
            coords = json.loads(raw)['coordinates']
            geom = Point(coords)
            geometries.append(geom)
        except:
            geometries.append(None)
    gdf = gpd.GeoDataFrame(df, geometry=geometries, crs="EPSG:4326")
    return gdf.dropna(subset=["geometry"])

@st.cache_resource
def entrenar_modelo():
    np.random.seed(42)
    data = pd.DataFrame({
        'bicis_disponibles': np.random.randint(0, 20, 200),
        'espacios_libres': np.random.randint(0, 20, 200),
        'hora': np.random.randint(0, 24, 200),
    })
    data['target'] = data['bicis_disponibles'] + np.random.normal(0, 2, 200)
    modelo = LinearRegression()
    modelo.fit(data[['bicis_disponibles', 'espacios_libres', 'hora']], data['target'])
    return modelo

@st.cache_data(ttl=3600)
def get_route(_client, coords, profile):
    return _client.directions(coords, profile=profile, format='geojson')

# --- CARGA DE DATOS ---
df, gdf = load_valenbisi()
df.columns = df.columns.str.lower()
gdf.columns = gdf.columns.str.lower()
df['numero'] = df['numero'].astype(str)
gdf['number'] = gdf['number'].astype(str)
merged = pd.merge(gdf, df, left_on='number', right_on='numero', how='inner')
traffic = load_traffic_csv()
transporte = load_transporte()
modelo = entrenar_modelo()

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📍 Estaciones", "🔹 Rutas + Tráfico + Transporte", "🤖 Predicción"])

# --- TAB 1 ---
with tab1:
    st.sidebar.header("Filtros")
    search = st.sidebar.text_input("Buscar por nombre")
    min_bikes = st.sidebar.slider("🚲 Bicis", 0, int(merged['bicis_disponibles'].max()), 0)
    min_slots = st.sidebar.slider("🅿️ Espacios", 0, int(merged['espacios_libres'].max()), 0)
    estado = st.sidebar.selectbox("🔌 Activas", ["Todas", "Sí", "No"])

    filt = merged[(merged['bicis_disponibles'] >= min_bikes) & (merged['espacios_libres'] >= min_slots)]
    if estado == "Sí": filt = filt[filt['activo'] == "T"]
    elif estado == "No": filt = filt[filt['activo'] == "F"]
    if search:
        filt = filt[filt['direccion'].str.contains(search, case=False)]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Estaciones", filt.shape[0])
    col2.metric("Bicis", int(filt['bicis_disponibles'].sum()))
    col3.metric("Huecos", int(filt['espacios_libres'].sum()))
    col4.metric("Ocupación", f"{1 - (filt['espacios_libres'].sum() / filt['espacios_totales'].sum()):.1%}")

    m = folium.Map(location=[39.4699, -0.3763], zoom_start=13)
    clust = MarkerCluster().add_to(m)
    for _, row in filt.iterrows():
        lon, lat = row['geometry'].coords[0]
        popup = f"""
        <b>{row['direccion']}</b><br>
        🚲 Bicis: {row['bicis_disponibles']}<br>
        🅿️ Libres: {row['espacios_libres']}<br>
        🔄 Total: {row['espacios_totales']}<br>
        ⏱ï¸ Actualización: {row['fecha_actualizacion']}
        """
        folium.Marker(
            [lat, lon], popup=popup,
            icon=folium.Icon(color="green" if row['bicis_disponibles'] > 0 else "red", icon="bicycle", prefix="fa")
        ).add_to(clust)
    st.subheader("🎥 Mapa interactivo")
    st_folium(m, width=1000, height=600)
    with st.expander("📋 Tabla de datos"):
        cols = ['direccion', 'numero', 'bicis_disponibles', 'espacios_libres', 'activo', 'fecha_actualizacion']
        st.dataframe(filt[cols].reset_index(drop=True))

# --- TAB 2 ---
with tab2:
    st.title("🔹 Comparativa bici vs coche + Transporte público")
    ORS_API_KEY = "5b3ce3597851110001cf6248051ffed92dc1476891f0e7c1228c0a91"
    estaciones = merged[['direccion','geometry']].drop_duplicates().sort_values('direccion')
    origen = st.selectbox("Origen", estaciones['direccion'])
    destino = st.selectbox("Destino", estaciones['direccion'], index=1)

    if origen == destino:
        st.warning("Selecciona estaciones distintas.")
    else:
        geo_o = estaciones[estaciones['direccion']==origen].iloc[0]['geometry']
        geo_d = estaciones[estaciones['direccion']==destino].iloc[0]['geometry']
        coords = [(geo_o.x, geo_o.y), (geo_d.x, geo_d.y)]
        client = ors.Client(key=ORS_API_KEY)

        route_bici = get_route(client, coords, 'cycling-regular')
        d_bici = route_bici['features'][0]['properties']['summary']['distance']
        t_bici = route_bici['features'][0]['properties']['summary']['duration']

        route_coche = get_route(client, coords, 'driving-car')
        d_coche = route_coche['features'][0]['properties']['summary']['distance']
        t_coche = route_coche['features'][0]['properties']['summary']['duration']

        # Mapa
        m = folium.Map(location=[geo_o.y, geo_o.x], zoom_start=14)
        folium.GeoJson(route_bici, name="Ruta bici", style_function=lambda x: {'color':'green'}).add_to(m)
        folium.GeoJson(route_coche, name="Ruta coche", style_function=lambda x: {'color':'blue'}).add_to(m)
        folium.LayerControl().add_to(m)
        st_folium(m, width=1000, height=600)

        # Métricas
        cols = st.columns(3)
        cols[0].metric("Bici (min)", f"{t_bici/60:.1f}")
        cols[0].metric("Dist. Bici", f"{d_bici/1000:.2f} km")
        cols[1].metric("Coche (min)", f"{t_coche/60:.1f}")
        cols[1].metric("Dist. Coche", f"{d_coche/1000:.2f} km")

        linea_ruta = LineString(route_coche['features'][0]['geometry']['coordinates'])
        intersectan = traffic[traffic.intersects(linea_ruta)]

        st.subheader("🚦 Tráfico en la ruta en coche")
        if intersectan.empty:
            st.info("No se detectaron tramos de tráfico cruzando la ruta.")
        else:
            estados = intersectan['Estat / Estado'].astype(float)

            # Estimar tiempo extra por tráfico: cada tramo con estado 1 o 2 suma tiempo
            tiempo_extra = 0
            for _, row in intersectan.iterrows():
                estado = int(row['Estat / Estado'])
                if estado == 1:
                    tiempo_extra += 30  # Moderado
                elif estado == 2:
                    tiempo_extra += 60  # Intenso

            # Mostrar calles afectadas y su estado
            for _, row in intersectan.iterrows():
                nombre = row['Denominació / Denominación']
                estado = int(row['Estat / Estado'])
                icono = "🟢" if estado == 0 else ("🟡" if estado == 1 else "🔴")
                estado_texto = {0: "Fluido", 1: "Moderado", 2: "Intenso",3: "Atasco"}[estado]
                st.write(f"{icono} **{nombre}** – Estado: {estado_texto}")

            st.success(f"⏱️ Tiempo adicional estimado por tráfico: **+{tiempo_extra} segundos**")

        

        # Ahorro de CO2
        co2_por_km = 0.21
        ahorro = (d_coche / 1000) * co2_por_km
        st.success(f"🌿 Ahorro de CO₂ estimado: **{ahorro:.2f} kg**")

        # Paradas cercanas
        st.markdown("### 🚇 Transporte público cercano")
        punto_destino = Point(geo_d.x, geo_d.y)
        transporte['dist_m'] = transporte.geometry.distance(punto_destino) * 111000
        cercanas = transporte[transporte['dist_m'] < 500].sort_values('dist_m')
        if cercanas.empty:
            st.info("No se encontraron paradas cercanas.")
        else:
            for _, row in cercanas.iterrows():
                st.markdown(f"""
                **{row['Denominació / Denominación']}** – Líneas: {row['Línies / Líneas']}  
                📍 Distancia: {row['dist_m']:.0f} m  
                🔗 [Ver horarios]({row['Pròximes Arribades / Próximas llegadas']})
                """)

# --- TAB 3 ---
with tab3:
    st.subheader("🤖 Predicción de bicis disponibles")
    est = st.selectbox("Estación", merged['direccion'].unique())
    row = merged[merged['direccion'] == est].iloc[0]
    hora = pd.Timestamp.now().hour
    minutos = st.slider("Minutos en el futuro", 5, 30, 10, step=5)

    X = [[row['bicis_disponibles'], row['espacios_libres'], hora]]
    pred = modelo.predict(X)[0]
    pred = max(0, round(pred))

    st.write(f"### Estado actual: {row['bicis_disponibles']} bicis disponibles")
    st.write(f"### Predicción a +{minutos} min: **{pred}** bicis disponibles")
