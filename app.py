import math
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk

# -------------------------------
# Helpers
# -------------------------------

@st.cache_data(show_spinner=False)
def haversine_miles(lat1, lon1, lat2, lon2):
    R = 3958.7613  # miles
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
    if warehouses.empty or cust.empty:
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
    D = np.vstack(dists).T  # [n_customers, n_warehouses]

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
            return "Next-Day 9:40am"
        elif miles <= ndeod:
            return "Next-Day EOD"
        else:
            return "2-Day+"
    out["service_level"] = out["distance_miles"].apply(labeler)
    return out

# -------------------------------
# UI
# -------------------------------

st.set_page_config(page_title="Service-Level Visualizer", layout="wide")
st.title("Service-Level Visualizer (distance-based)")
st.caption("Upload customers, input warehouses, set mileage bands, and visualize/measure coverage.")

with st.expander("Tips", expanded=False):
    st.markdown(
        "- Use the **Road factor** to approximate driving miles (e.g., 1.10–1.20).\n"
        "- Adjust **point sizes** below if customers are hard to see.\n"
        "- You can paste warehouses from Excel into the editor."
    )

# Sidebar
st.sidebar.header("Inputs")

# Upload customers
uploaded = st.sidebar.file_uploader("Upload customers CSV", type=["csv"])
if uploaded:
    raw = pd.read_csv(uploaded)
else:
    raw = pd.DataFrame({
        "latitude": [33.748995, 34.052235, 40.712776, 41.878113, 29.760427],
        "longitude": [-84.387982, -118.243683, -74.005974, -87.629799, -95.369804],
    })

st.sidebar.markdown("**Map latitude/longitude columns**")
lat_col = st.sidebar.selectbox("Latitude column", options=list(raw.columns), index=0)
lon_col = st.sidebar.selectbox("Longitude column", options=list(raw.columns), index=min(1, len(raw.columns)-1))
customers = raw.rename(columns={lat_col: "lat", lon_col: "lon"})[["lat", "lon"]].copy()

# Warehouses
st.sidebar.markdown("---")
st.sidebar.subheader("Warehouse locations")
default_wh = pd.DataFrame({
    "name": ["RNO", "DFW", "CHI", "ATL", "YOR"],
    "lat": [39.5296, 32.7767, 41.8781, 33.7490, 39.9626],
    "lon": [-119.8138, -96.7970, -87.6298, -84.3880, -76.7277],
})
wh_editor = st.sidebar.data_editor(default_wh, use_container_width=True, num_rows="dynamic", key="wh_editor")

# Bands
st.sidebar.markdown("---")
st.sidebar.subheader("Service bands (miles)")
nd940 = st.sidebar.number_input("Next-Day 9:40am (miles)", min_value=0, max_value=3000, value=350, step=10)
ndeod = st.sidebar.number_input("Next-Day EOD (miles)", min_value=0, max_value=3000, value=500, step=10)
if ndeod < nd940:
    st.sidebar.warning("Next-Day EOD must be ≥ Next-Day 9:40am. Adjusting to match.")
    ndeod = nd940

st.sidebar.markdown("---")
road_factor = st.sidebar.slider("Road factor", 0.8, 1.6, 1.15, 0.01, help="Multiply great-circle miles by this factor")

# Point size controls
st.sidebar.markdown("---")
st.sidebar.subheader("Map point sizes")
cust_radius_m = st.sidebar.slider("Customer point radius (meters)", 1000, 20000, 8000, 500)
wh_radius_m = st.sidebar.slider("Warehouse point radius (meters)", 2000, 40000, 16000, 1000)

# Clean WH
warehouses = wh_editor.copy()
for c in ["lat", "lon"]:
    warehouses[c] = pd.to_numeric(warehouses[c], errors="coerce")
warehouses = warehouses.dropna(subset=["name", "lat", "lon"]).reset_index(drop=True)

# Compute
base = compute_nearest_wh(customers, warehouses)
base["distance_miles"] = base["distance_miles"] * road_factor
labeled = classify_service_levels(base, nd940, ndeod)

# KPIs
total = len(labeled)
by_band = labeled["service_level"].value_counts().reindex(["Next-Day 9:40am", "Next-Day EOD", "2-Day+", "Unassigned"], fill_value=0)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Next-Day 9:40am", f"{by_band['Next-Day 9:40am']:,}", f"{(by_band['Next-Day 9:40am']/total*100 if total else 0):.1f}%")
col2.metric("Next-Day EOD", f"{by_band['Next-Day EOD']:,}", f"{(by_band['Next-Day EOD']/total*100 if total else 0):.1f}%")
col3.metric("2-Day+", f"{by_band['2-Day+']:,}", f"{(by_band['2-Day+']/total*100 if total else 0):.1f}%")
col4.metric("Total customers", f"{total:,}")

# -------------------------------
# Map
# -------------------------------
st.subheader("Map")

# Color map for bands (RGBA for visibility)
color_map = {
    "Next-Day 9:40am": [0, 128, 0, 200],     # green
    "Next-Day EOD": [255, 165, 0, 220],      # orange
    "2-Day+": [220, 20, 60, 240],            # crimson
    "Unassigned": [128, 128, 128, 220],
}
plot_df = labeled.copy()
plot_df["fill_color"] = plot_df["service_level"].map(color_map)
plot_df["line_color"] = [0, 0, 0, 230]

if plot_df.empty:
    plot_df = pd.DataFrame({"lat": [39.5], "lon": [-98.35], "fill_color": [[128,128,128,200]], "line_color": [[0,0,0,230]], "service_level": ["Unassigned"], "nearest_wh": [None], "distance_miles": [np.nan]})

mean_lat = float(plot_df["lat"].mean())
mean_lon = float(plot_df["lon"].mean())

# Customer layer with stroke outline for visibility
cust_layer = pdk.Layer(
    "ScatterplotLayer",
    data=plot_df,
    get_position="[lon, lat]",
    get_fill_color="fill_color",
    get_line_color="line_color",
    get_radius=cust_radius_m,
    stroked=True,
    lineWidthMinPixels=1.5,
    pickable=True,
    auto_highlight=True,
)

# Warehouse layer (solid black points with white outline)
wh_layer = pdk.Layer(
    "ScatterplotLayer",
    data=warehouses.assign(fill_color=[[0,0,0,255]]*len(warehouses)),
    get_position="[lon, lat]",
    get_fill_color="fill_color",
    get_radius=wh_radius_m,
    stroked=True,
    get_line_color=[255,255,255,255],
    lineWidthMinPixels=2.5,
    pickable=True,
)

# Warehouse labels
label_layer = pdk.Layer(
    "TextLayer",
    data=warehouses.assign(text=warehouses["name"]),
    get_position="[lon, lat]",
    get_text="text",
    get_size=14,
    get_color=[0,0,0,255],
    get_angle=0,
    get_alignment_baseline="'top'",
    get_pixel_offset=[0, -12],
)

deck = pdk.Deck(
    map_style="mapbox://styles/mapbox/light-v9",
    initial_view_state=pdk.ViewState(latitude=mean_lat, longitude=mean_lon, zoom=4),
    layers=[cust_layer, wh_layer, label_layer],
    tooltip={
        "html": "<b>Band:</b> {service_level}<br/><b>Nearest WH:</b> {nearest_wh}<br/><b>Distance:</b> {distance_miles} mi",
        "style": {"backgroundColor": "#ffffff", "color": "#000000"},
    },
)
st.pydeck_chart(deck)

# Legend
st.markdown(
    "**Legend**  
"
    "🟢 Next-Day 9:40am &nbsp;&nbsp; 🟠 Next-Day EOD &nbsp;&nbsp; 🔴 2-Day+ &nbsp;&nbsp; ⬤ Black = Warehouse"
)

# Data & download
with st.expander("Preview data & download results", expanded=False):
    st.dataframe(labeled.assign(distance_miles=labeled["distance_miles"].round(1)))
    st.download_button("Download results CSV", labeled.to_csv(index=False), file_name="service_levels_by_customer.csv", mime="text/csv")
