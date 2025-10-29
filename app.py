import math
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk

@st.cache_data(show_spinner=False)
def haversine_miles(lat1, lon1, lat2, lon2):
    R = 3958.7613
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

@st.cache_data(show_spinner=False)
def compute_nearest_wh(customers, warehouses):
    if customers.empty or warehouses.empty:
        return pd.DataFrame()
    c_lat, c_lon = customers['lat'].to_numpy(), customers['lon'].to_numpy()
    w_lat, w_lon = warehouses['lat'].to_numpy(), warehouses['lon'].to_numpy()
    dists = [haversine_miles(c_lat, c_lon, w_lat[i], w_lon[i]) for i in range(len(warehouses))]
    D = np.vstack(dists).T
    nearest_idx = np.argmin(D, axis=1)
    customers['nearest_wh'] = warehouses['name'].iloc[nearest_idx].to_numpy()
    customers['distance_miles'] = D[np.arange(D.shape[0]), nearest_idx]
    return customers

@st.cache_data(show_spinner=False)
def classify_service_levels(df, nd940, ndeod):
    def labeler(m):
        if pd.isna(m): return 'Unassigned'
        if m <= nd940: return 'Next-Day 9:40am'
        elif m <= ndeod: return 'Next-Day EOD'
        else: return '2-Day+'
    df['service_level'] = df['distance_miles'].apply(labeler)
    return df

st.set_page_config(page_title='Service-Level Visualizer', layout='wide')
st.title('Service-Level Visualizer (distance-based)')
st.caption('Upload customers, input warehouses, set mileage bands, and visualize/measure coverage.')

st.sidebar.header('Inputs')

uploaded = st.sidebar.file_uploader('Upload customers CSV', type=['csv'])
if uploaded:
    raw = pd.read_csv(uploaded)
else:
    raw = pd.DataFrame({'latitude':[33.7,34.0,40.7,41.8],'longitude':[-84.3,-118.2,-74.0,-87.6]})

lat_col = st.sidebar.selectbox('Latitude column', raw.columns, index=0)
lon_col = st.sidebar.selectbox('Longitude column', raw.columns, index=1)
customers = raw.rename(columns={lat_col:'lat',lon_col:'lon'})[['lat','lon']]

default_wh = pd.DataFrame({
    'name':['RNO','DFW','CHI','ATL','YOR'],
    'lat':[39.5,32.7,41.8,33.7,39.9],
    'lon':[-119.8,-96.7,-87.6,-84.3,-76.7]
})
warehouses = st.sidebar.data_editor(default_wh, use_container_width=True, num_rows='dynamic')

nd940 = st.sidebar.number_input('Next-Day 9:40am (miles)',0,3000,350,10)
ndeod = st.sidebar.number_input('Next-Day EOD (miles)',0,3000,500,10)
if ndeod < nd940: ndeod = nd940
road_factor = st.sidebar.slider('Road factor',0.8,1.6,1.15,0.01)

cust_radius_m = st.sidebar.slider('Customer point radius (meters)', 500, 80000, 12000, 1000)
wh_radius_m = st.sidebar.slider('Warehouse point radius (meters)', 2000, 100000, 25000, 2000)

warehouses = warehouses.dropna(subset=['lat','lon']).reset_index(drop=True)
base = compute_nearest_wh(customers, warehouses)
base['distance_miles'] *= road_factor
labeled = classify_service_levels(base, nd940, ndeod)

color_map = {
    'Next-Day 9:40am':[0,128,0,220],
    'Next-Day EOD':[255,165,0,220],
    '2-Day+':[220,20,60,240],
    'Unassigned':[128,128,128,220]
}
labeled['fill_color'] = labeled['service_level'].map(color_map)

mean_lat, mean_lon = labeled['lat'].mean(), labeled['lon'].mean()

cust_layer = pdk.Layer('ScatterplotLayer', data=labeled, get_position='[lon, lat]',
    get_fill_color='fill_color', get_radius=cust_radius_m, stroked=True,
    get_line_color=[0,0,0], lineWidthMinPixels=1.5, pickable=True)

wh_layer = pdk.Layer('ScatterplotLayer', data=warehouses,
    get_position='[lon, lat]', get_fill_color=[0,0,0,255], get_radius=wh_radius_m,
    stroked=True, get_line_color=[255,255,255], lineWidthMinPixels=2.5, pickable=True)

label_layer = pdk.Layer('TextLayer', data=warehouses.assign(text=warehouses['name']),
    get_position='[lon, lat]', get_text='text', get_size=16, get_color=[0,0,0,255],
    get_alignment_baseline="'top'", get_pixel_offset=[0,-12])

deck = pdk.Deck(map_style='mapbox://styles/mapbox/light-v9',
    initial_view_state=pdk.ViewState(latitude=mean_lat, longitude=mean_lon, zoom=4),
    layers=[cust_layer, wh_layer, label_layer],
    tooltip={'html':'<b>Band:</b> {service_level}<br/><b>Nearest WH:</b> {nearest_wh}<br/><b>Distance:</b> {distance_miles} mi'})

st.pydeck_chart(deck)

st.markdown('**Legend**  \n🟢 Next-Day 9:40am | 🟠 Next-Day EOD | 🔴 2-Day+ | ⚫ Warehouse')
