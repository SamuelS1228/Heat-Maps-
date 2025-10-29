import math
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk

@st.cache_data(show_spinner=False)
def haversine_miles(lat1, lon1, lat2, lon2):
    R = 3958.7613
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

@st.cache_data(show_spinner=False)
def compute_nearest_wh(customers: pd.DataFrame, warehouses: pd.DataFrame) -> pd.DataFrame:
    cust = customers.copy().reset_index(drop=True)
    if warehouses.empty:
        cust["nearest_wh"] = None
        cust["distance_miles"] = np.nan
        return cust

    c_lat = cust["lat"].to_numpy()
    c_lon = cust["lon"].to_numpy()
    w_lat = warehouses["lat"].to_numpy()
    w_lon = warehouses["lon"].to_numpy()

    dists = []
    for i in range(len(warehouses)):
        d = haversine_miles(c_lat, c_lon, w_lat[i], w_lon[i])
        dists.append(d)
    D = np.vstack(dists).T

    nearest_idx = np.argmin(D, axis=1)
    cust["nearest_wh"] = warehouses["name"].iloc[nearest_idx].to_numpy()
    cust["distance_miles"] = D[np.arange(D.shape[0]), nearest_idx]
    return cust

@st.cache_data(show_spinner=False)
def classify_service_levels(df: pd.DataFrame, nd940: float, ndeod: float) -> pd.DataFrame:
    out = df.copy()
    def labeler(miles: float) -> str:
        if pd.isna(miles):
            return "Unassigned"
        if miles <= nd940:
            return "Next‑Day 9:40am"
        elif miles <= ndeod:
            return "Next‑Day EOD"
        else:
            return "2‑Day+"
    out["service_level"] = out["distance_miles"].apply(labeler)
    return out

st.set_page_config(page_title="Service‑Level Visualizer", layout="wide")
st.title("Service‑Level Visualizer (distance‑based)")
st.caption("Upload customers, input warehouses, set mileage bands, and visualize/measure coverage.")

st.sidebar.header("Inputs")

uploaded = st.sidebar.file_uploader("Upload customers CSV", type=["csv"])

if uploaded:
    raw = pd.read_csv(uploaded)
else:
    raw = pd.DataFrame({
        "latitude": [33.748995, 34.052235, 40.712776, 41.878113],
        "longitude": [-84.387982, -118.243683, -74.005974, -87.629799],
    })

lat_col = st.sidebar.selectbox("Latitude column", options=list(raw.columns))
lon_col = st.sidebar.selectbox("Longitude column", options=list(raw.columns))
customers = raw.rename(columns={lat_col: "lat", lon_col: "lon"})[["lat", "lon"]].copy()

st.sidebar.subheader("Warehouse locations")
default_wh = pd.DataFrame({
    "name": ["RNO", "DFW", "CHI", "ATL", "YOR"],
    "lat": [39.5296, 32.7767, 41.8781, 33.7490, 39.9626],
    "lon": [-119.8138, -96.7970, -87.6298, -84.3880, -76.7277],
})
wh_editor = st.sidebar.data_editor(default_wh, use_container_width=True, num_rows="dynamic", key="wh_editor")

st.sidebar.subheader("Service bands (miles)")
nd940 = st.sidebar.number_input("Next‑Day 9:40am (miles)", min_value=0, max_value=3000, value=350, step=10)
ndeod = st.sidebar.number_input("Next‑Day EOD (miles)", min_value=0, max_value=3000, value=500, step=10)
if ndeod < nd940:
    ndeod = nd940

road_factor = st.sidebar.slider("Road factor", 0.8, 1.6, 1.15, 0.01)

warehouses = wh_editor.copy()
for c in ["lat", "lon"]:
    warehouses[c] = pd.to_numeric(warehouses[c], errors="coerce")
warehouses = warehouses.dropna(subset=["name", "lat", "lon"]).reset_index(drop=True)

base = compute_nearest_wh(customers, warehouses)
base["distance_miles"] = base["distance_miles"] * road_factor
labeled = classify_service_levels(base, nd940, ndeod)

total = len(labeled)
by_band = labeled["service_level"].value_counts().reindex(["Next‑Day 9:40am", "Next‑Day EOD", "2‑Day+", "Unassigned"], fill_value=0)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Next‑Day 9:40am", f"{by_band['Next‑Day 9:40am']:,}", f"{(by_band['Next‑Day 9:40am']/total*100 if total else 0):.1f}%")
col2.metric("Next‑Day EOD", f"{by_band['Next‑Day EOD']:,}", f"{(by_band['Next‑Day EOD']/total*100 if total else 0):.1f}%")
col3.metric("2‑Day+", f"{by_band['2‑Day+']:,}", f"{(by_band['2‑Day+']/total*100 if total else 0):.1f}%")
col4.metric("Total customers", f"{total:,}")

color_map = {
    "Next‑Day 9:40am": [0, 128, 0],
    "Next‑Day EOD": [255, 165, 0],
    "2‑Day+": [220, 20, 60],
    "Unassigned": [128, 128, 128],
}
plot_df = labeled.copy()
plot_df["color"] = plot_df["service_level"].map(color_map)

mean_lat = float(plot_df["lat"].mean())
mean_lon = float(plot_df["lon"].mean())

cust_layer = pdk.Layer("ScatterplotLayer", data=plot_df, get_position="[lon, lat]", get_fill_color="color", get_radius=3500, pickable=True, auto_highlight=True)
wh_layer = pdk.Layer("ScatterplotLayer", data=warehouses, get_position="[lon, lat]", get_fill_color=[0, 0, 0], get_radius=6000)

r = pdk.Deck(map_style="mapbox://styles/mapbox/light-v9", initial_view_state=pdk.ViewState(latitude=mean_lat, longitude=mean_lon, zoom=4), layers=[cust_layer, wh_layer])
st.pydeck_chart(r)

st.download_button("Download results CSV", labeled.to_csv(index=False), file_name="service_levels_by_customer.csv")
