# =========================================================
# FILE: ui_components.py
# FUNGSI: Merender blok HTML antarmuka pengguna
# =========================================================

# 1. Komponen Top Navigation Bar (Header)
def render_navbar():
    
    html = """
<div class="top-navbar">
    <div class="nav-brand">
        <div class="nav-logo">🌊</div>
        <div class="nav-title-group">
            <div class="nav-title">Weathering Waves AI</div>
            <div class="nav-subtitle">Real-Time Deep Learning & Fuzzy Logic</div>
        </div>
    </div>
</div>
"""
    
    return html.replace('\n', '')


# 2. Komponen Kartu Hero (Cuaca Saat Ini)
def render_hero_card(waktu_header, ikon_hero, suhu_now, status_teks_hero, kel_now, tek_now, idx_now):
    
    html = f"""
<div class="hero-card">
    <div class="hero-header">
        <span>CURRENT WEATHER</span>
        <span>{waktu_header}</span>
    </div>
    
    <div class="hero-body">
        <div class="hero-left">
            <div class="hero-temp-row">
                <div class="hero-icon">{ikon_hero}</div>
                <div class="hero-temp-wrapper">
                    <div class="hero-temp">{suhu_now}</div>
                    <div class="hero-unit">&deg;C</div>
                </div>
            </div>
            <div class="hero-desc">{status_teks_hero}</div>
        </div>
        
        <div class="hero-right">
            <div class="hero-detail-row">
                <span class="hero-label">Relative Humidity</span>
                <span class="hero-val">{kel_now} %</span>
            </div>
            <div class="hero-detail-row">
                <span class="hero-label">Atmospheric Pressure</span>
                <span class="hero-val">{tek_now} hPa</span>
            </div>
            <div class="hero-detail-row">
                <span class="hero-label">Precision Temp (Sensor)</span>
                <span class="hero-val">{suhu_now} &deg;C</span>
            </div>
            <div class="hero-detail-row">
                <span class="hero-label">Current Fuzzy Index</span>
                <span class="hero-val val-highlight">{idx_now}</span>
            </div>
        </div>
    </div>
</div>
"""
    
    return html.replace('\n', '')


# 3. Komponen Kartu Jam (Scroll Horizontal)
def render_hourly_item(jam, ikon, suhu, kel, idx, status):
    
    html = f"""
<div class="hourly-item" title="{status}">
    <div class="h-time">{jam}</div>
    <div class="h-icon">{ikon}</div>
    <div class="h-temp">{suhu}&deg;</div>
    <div class="h-drop">💧 {kel}%</div>
    <div class="h-idx">Idx: {round(idx, 1)}</div>
</div>
"""

    return html.replace('\n', '')