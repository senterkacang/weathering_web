# =========================================================
# WEB DASHBOARD PREDIKSI CUACA (FINAL EDITION)
# File: app.py (Main Controller)
# =========================================================

# ---------------------------------------------------------
# 1. IMPORT LIBRARIES
# ---------------------------------------------------------

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from datetime import datetime, timedelta
import pytz
import plotly.express as px
from ui_components import render_navbar, render_hero_card, render_hourly_item

# Library Peta Interaktif
import folium
from streamlit_folium import st_folium

# Mengimpor komponen UI dari file terpisah
from ui_components import render_hero_card, render_hourly_item


# ---------------------------------------------------------
# 2. SYSTEM & PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Weathering Waves AI", 
    page_icon="🌊", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Inisialisasi Session State untuk Lokasi Default (Sleman, Yogyakarta)
if 'lat' not in st.session_state:
    st.session_state['lat'] = -7.7956

if 'lon' not in st.session_state:
    st.session_state['lon'] = 110.3695

TIMEZONE = "Asia/Jakarta"
LOOKBACK_HOURS = 72
FORECAST_HOURS = 24

MODEL_PATH = 'weathering_waves.keras'
SCALER_PATH = 'weathering_waves_scaler.pkl'

INPUT_FEATURES = [
    'suhu', 'kelembaban', 'tekanan_udara', 'tutupan_awan', 
    'kecepatan_angin', 'jam_sin', 'jam_cos'
]

TARGET_FEATURE = ['suhu', 'kelembaban', 'tekanan_udara']

NUM_FEATURES_INPUT = len(INPUT_FEATURES)
NUM_FEATURES_OUTPUT = len(TARGET_FEATURE)


# ---------------------------------------------------------
# 3. FUZZY LOGIC ENGINE (SUGENO)
# ---------------------------------------------------------

# Kurva Dasar
def trimf(x, a, b, c):
    if x <= a or x >= c: return 0.0
    elif x == b: return 1.0
    elif x < b: return (x - a) / (b - a)
    else: return (c - x) / (c - b)

def left_shoulder(x, a, b):
    if x <= a: return 1.0
    elif x >= b: return 0.0
    else: return (b - x) / (b - a)

def right_shoulder(x, a, b):
    if x <= a: return 0.0
    elif x >= b: return 1.0
    else: return (x - a) / (b - a)

# Fungsi Keanggotaan
def suhu_dingin(x): return left_shoulder(x, 24.0, 26.0)
def suhu_normal(x): return trimf(x, 24.0, 26.5, 29.0)
def suhu_panas(x): return right_shoulder(x, 27.0, 30.0)

def hum_rendah(x): return left_shoulder(x, 70, 80)
def hum_sedang(x): return trimf(x, 75, 85, 95)
def hum_tinggi(x): return right_shoulder(x, 90, 100)

def press_rendah(x): return left_shoulder(x, 996.5, 998.0)
def press_normal(x): return trimf(x, 997.0, 998.5, 1000.0)
def press_tinggi(x): return right_shoulder(x, 999.0, 1001.0)

# Aturan (Rule Base)
RULES = {
    ("dingin", "rendah", "rendah"): 40, ("dingin", "rendah", "normal"): 25, ("dingin", "rendah", "tinggi"): 10,
    ("normal", "rendah", "rendah"): 35, ("normal", "rendah", "normal"): 20, ("normal", "rendah", "tinggi"): 10,
    ("panas", "rendah", "rendah"): 30, ("panas", "rendah", "normal"): 15, ("panas", "rendah", "tinggi"): 10,
    ("dingin", "sedang", "rendah"): 70, ("dingin", "sedang", "normal"): 50, ("dingin", "sedang", "tinggi"): 35,
    ("normal", "sedang", "rendah"): 65, ("normal", "sedang", "normal"): 45, ("normal", "sedang", "tinggi"): 30,
    ("panas", "sedang", "rendah"): 55, ("panas", "sedang", "normal"): 40, ("panas", "sedang", "tinggi"): 25,
    ("dingin", "tinggi", "rendah"): 100, ("dingin", "tinggi", "normal"): 90, ("dingin", "tinggi", "tinggi"): 70,
    ("normal", "tinggi", "rendah"): 95, ("normal", "tinggi", "normal"): 85, ("normal", "tinggi", "tinggi"): 65,
    ("panas", "tinggi", "rendah"): 85, ("panas", "tinggi", "normal"): 75, ("panas", "tinggi", "tinggi"): 60,
}

# Defuzzifikasi
def fuzzy_weather_sugeno(suhu, kelembaban, tekanan):
    suhu_set = {"dingin": suhu_dingin(suhu), "normal": suhu_normal(suhu), "panas": suhu_panas(suhu)}
    hum_set = {"rendah": hum_rendah(kelembaban), "sedang": hum_sedang(kelembaban), "tinggi": hum_tinggi(kelembaban)}
    press_set = {"rendah": press_rendah(tekanan), "normal": press_normal(tekanan), "tinggi": press_tinggi(tekanan)}

    numerator = 0
    denominator = 0
    
    for (s, h, p), z in RULES.items():
        w = min(suhu_set[s], hum_set[h], press_set[p])
        numerator += w * z
        denominator += w
        
    if denominator == 0: return 0
    return numerator / denominator

def weather_label(index):
    if index < 20: return "Clear ☀️"
    elif index < 40: return "Partly Cloudy ⛅"
    elif index < 60: return "Cloudy ☁️"
    elif index < 80: return "Light Rain 🌦️"
    elif index < 90: return "Moderate Rain 🌧️"
    else: return "Heavy Rain ⛈️"


# ---------------------------------------------------------
# 4. DEEP LEARNING ENGINE & DATA FETCHING
# ---------------------------------------------------------


@st.cache_resource
def load_ml_assets():
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

def fetch_exact_72h_history(waktu_sekarang, lat, lon):
    import requests
    
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&"
        f"past_days=4&forecast_days=0&"
        f"hourly=temperature_2m,relative_humidity_2m,"
        f"surface_pressure,cloud_cover,wind_speed_10m&"
        f"timezone=Asia%2FJakarta"
    )
    
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame({
            'suhu': data['hourly']['temperature_2m'],
            'kelembaban': data['hourly']['relative_humidity_2m'],
            'tekanan_udara': data['hourly']['surface_pressure'],
            'tutupan_awan': data['hourly']['cloud_cover'],
            'kecepatan_angin': data['hourly']['wind_speed_10m']
        })
        
        df['waktu'] = pd.to_datetime(data['hourly']['time'])
        df = df[df['waktu'] <= waktu_sekarang]
        
        df_72h = df.tail(LOOKBACK_HOURS).copy()
        df_72h['jam'] = df_72h['waktu'].dt.hour
        df_72h['jam_sin'] = np.sin(df_72h['jam'] * (2. * np.pi / 24))
        df_72h['jam_cos'] = np.cos(df_72h['jam'] * (2. * np.pi / 24))
        
        return df_72h[INPUT_FEATURES]
    else:
        st.error(f"Failed to connect to Open-Meteo API: {response.status_code}")
        return None

def run_realtime_prediction(df_input, model, scaler):
    input_scaled = scaler.transform(df_input)
    input_tensor = np.expand_dims(input_scaled, axis=0)
    
    predicted_scaled = model.predict(input_tensor, verbose=0)[0]
    
    dummy_pred = np.zeros((FORECAST_HOURS, NUM_FEATURES_INPUT))
    dummy_pred[:, :NUM_FEATURES_OUTPUT] = predicted_scaled
    
    pred_asli_lengkap = scaler.inverse_transform(dummy_pred)
    
    return pred_asli_lengkap[:, :NUM_FEATURES_OUTPUT]


# ---------------------------------------------------------
# 5. UTILITY FUNCTIONS
# ---------------------------------------------------------

def load_css(file_name):
    """Membaca file CSS dan menampilkan error jika gagal ditemukan"""
    import os
    if os.path.exists(file_name):
        with open(file_name, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.error(f"⚠️ Gagal memuat desain! File '{file_name}' tidak ditemukan di folder: {os.getcwd()}")


# ---------------------------------------------------------
# 6. MAIN USER INTERFACE
# ---------------------------------------------------------

def main():
    
    # 6.1. Eksekusi CSS Global
    load_css("style.css")
    
    # 6.2. Header Dashboard
    st.markdown(render_navbar(), unsafe_allow_html=True)

    # 6.3. Load Model & Setup Waktu
    with st.spinner('Loading AI Model & Initializing System...'):
        model, scaler = load_ml_assets()

    tz = pytz.timezone(TIMEZONE)
    waktu_raw = datetime.now(tz)
    waktu_sekarang = waktu_raw.replace(minute=0, second=0, microsecond=0, tzinfo=None)

    # 6.4. Eksekusi Engine Berdasarkan Pilihan Lokasi
    with st.spinner('Fetching localized satellite data & generating forecast...'):
        current_lat = st.session_state['lat']
        current_lon = st.session_state['lon']
        
        df_input = fetch_exact_72h_history(waktu_sekarang, current_lat, current_lon)
        
        if df_input is not None:
            hasil_prediksi = run_realtime_prediction(df_input, model, scaler)
            
            # Kompilasi Hasil Prediksi 24 Jam
            hasil_list = []
            for i in range(FORECAST_HOURS):
                waktu_prediksi = waktu_sekarang + timedelta(hours=i+1)
                suhu = round(hasil_prediksi[i, 0], 1)
                kel = round(hasil_prediksi[i, 1], 1)
                tek = round(hasil_prediksi[i, 2], 1)
                
                idx_fuzzy = fuzzy_weather_sugeno(suhu, kel, tek)
                status = weather_label(idx_fuzzy)
                
                hasil_list.append({
                    "Time": waktu_prediksi,
                    "Hour": waktu_prediksi.strftime('%I %p'),
                    "Temperature (°C)": suhu,
                    "Humidity (%)": kel,
                    "Pressure (hPa)": tek,
                    "Fuzzy Index": round(idx_fuzzy, 1),
                    "Status": status
                })
            
            df_hasil = pd.DataFrame(hasil_list)

            # Ekstraksi Kondisi Saat Ini (Current Weather)
            current_data = df_input.iloc[-1]
            suhu_now = round(current_data['suhu'], 1)
            kel_now = round(current_data['kelembaban'], 1)
            tek_now = round(current_data['tekanan_udara'], 1)
            
            idx_now = fuzzy_weather_sugeno(suhu_now, kel_now, tek_now)
            status_now = weather_label(idx_now)

            icon_map = {
                "Clear ☀️": "☀️",
                "Partly Cloudy ⛅": "⛅",
                "Cloudy ☁️": "☁️",
                "Light Rain 🌦️": "🌦️",
                "Moderate Rain 🌧️": "🌧️",
                "Heavy Rain ⛈️": "⛈️"
            }


            # ---------------------------------------------
            # 7. TATA LETAK SPLIT-SCREEN (WIDGET & MAP)
            # ---------------------------------------------
            
            col_widget, col_map = st.columns([1.8, 1])
            
            # 7.1. Hero Section: Current Weather Widget
            with col_widget:
                st.markdown("### 🎯 Current Weather")
                
                ikon_hero = icon_map.get(status_now, "❓")
                status_teks_hero = status_now.split(" ")[0] + (" " + status_now.split(" ")[1] if len(status_now.split(" ")) > 1 else "")
                waktu_header = waktu_raw.strftime('%I:%M %p') 

                # Render komponen HTML dengan argumen yang diekstrak
                hero_html = render_hero_card(
                    waktu_header=waktu_header, 
                    ikon_hero=ikon_hero, 
                    suhu_now=int(round(suhu_now)), 
                    status_teks_hero=status_teks_hero, 
                    kel_now=kel_now, 
                    tek_now=tek_now, 
                    idx_now=round(idx_now, 1)
                )
                
                st.markdown(hero_html, unsafe_allow_html=True)
                st.info(f"📍 Target Coordinates: {current_lat:.4f}, {current_lon:.4f}")


            # 7.2. Interactive Map Selection Widget
            with col_map:
                # 1. JUDUL DIPINDAHKAN KE LUAR agar sejajar persis dengan judul di kiri
                st.markdown("### 🗺️ Select Location")
                
                # Membungkus peta ke dalam card bawaan Streamlit
                with st.container(border=True):
                    st.caption("Click anywhere on the map to forecast.")
                    
                    m = folium.Map(
                        location=[current_lat, current_lon], 
                        zoom_start=10, 
                        control_scale=True,
                        tiles=None 
                    )
                    
                    folium.TileLayer(
                        tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
                        attr='Google',
                        name='Google Maps Hybrid',
                        overlay=False,
                        control=True
                    ).add_to(m)
                    
                    folium.Marker(
                        [current_lat, current_lon], 
                        tooltip="Current Prediction Area", 
                        icon=folium.Icon(color="red", icon="info-sign")
                    ).add_to(m)
                    
                    # 2. TINGGI PETA DITAMBAH (315) agar garis bawahnya rata dengan info box di kiri
                    map_data = st_folium(m, width=None, height=315, returned_objects=["last_clicked"])
                    
                    if map_data["last_clicked"]:
                        new_lat = map_data["last_clicked"]["lat"]
                        new_lon = map_data["last_clicked"]["lng"]
                        
                        if new_lat != st.session_state['lat'] or new_lon != st.session_state['lon']:
                            st.session_state['lat'] = new_lat
                            st.session_state['lon'] = new_lon
                            st.rerun()

            st.write("---")


            # ---------------------------------------------
            # 8. VISUALISASI LANJUTAN (LIST & CHART)
            # ---------------------------------------------

            # 8.1. Hourly Forecast List (Horizontal Scroll)
            st.markdown("### 🕒 Hourly Forecast")

            # Membuka pembungkus container horizontal
            hourly_html = '<div class="hourly-wrapper">'

            for index, row in df_hasil.iterrows():
                h_jam = row["Hour"]
                if h_jam.startswith("0"): h_jam = h_jam[1:]
                
                h_suhu = int(round(row['Temperature (°C)']))
                h_kel = int(round(row['Humidity (%)']))
                h_idx = row['Fuzzy Index']
                h_ikon = icon_map.get(row['Status'], "❓")
                
                # Render baris item hourly
                hourly_html += render_hourly_item(
                    jam=h_jam, 
                    ikon=h_ikon, 
                    suhu=h_suhu, 
                    kel=h_kel, 
                    idx=h_idx, 
                    status=row['Status']
                )

            # Menutup pembungkus container
            hourly_html += '</div>'
            
            st.markdown(hourly_html, unsafe_allow_html=True)
            
            st.write("---")


            # 8.2. Interactive Plotly Area Charts
            with st.container(border=True):
                st.markdown("### 📈 24-Hour Trend")
                
                tab1, tab2, tab3 = st.tabs(["🌡️ Temperature", "💧 Humidity", "🌪️ Air Pressure"])
                
                def create_area_chart(df, y_col, color, title):
                    
                    fig = px.area(
                        df, x="Hour", y=y_col, 
                        markers=True, 
                        color_discrete_sequence=[color],
                        title=f"{title} Forecast"
                    )
                    
                    fig.update_layout(
                        xaxis_title="Time (Hour)",
                        yaxis_title=y_col,
                        hovermode="x unified",
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=0, r=0, t=40, b=0)
                    )
                    
                    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
                    
                    # Konfigurasi Auto Scale untuk sumbu Y
                    fig.update_yaxes(
                        showgrid=True, 
                        gridwidth=1, 
                        gridcolor='rgba(128, 128, 128, 0.2)', 
                        rangemode="normal"
                    )
                    
                    return fig

                with tab1:
                    st.plotly_chart(create_area_chart(df_hasil, "Temperature (°C)", "#FF4B4B", "Temperature"), use_container_width=True)
                with tab2:
                    st.plotly_chart(create_area_chart(df_hasil, "Humidity (%)", "#0068C9", "Relative Humidity"), use_container_width=True)
                with tab3:
                    st.plotly_chart(create_area_chart(df_hasil, "Pressure (hPa)", "#29B09D", "Atmospheric Pressure"), use_container_width=True)


# ---------------------------------------------------------
# 9. EXECUTION POINT
# ---------------------------------------------------------
if __name__ == "__main__":
    main()