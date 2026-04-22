import streamlit as st
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor

CITIES = {
    "Lahore":          (31.5204, 74.3587),
    "Faisalabad":      (31.4180, 73.0790),
    "Hyderabad":       (25.3960, 68.3578),
    "Sukkur":          (27.7244, 68.8572),
    "Abbottabad":      (34.1688, 73.2215),
    "Dera Ismail Khan":(31.8314, 70.9017),
    "Chaman":          (30.9218, 66.4490),
    "Gilgit":          (35.9220, 74.3087),
    "Skardu":          (35.2971, 75.6333),
    "Muzaffarabad":    (34.3700, 73.4710),
    "Quetta":          (30.1798, 66.9750),
    "Bhimber":         (32.9726, 74.0694),
}

COLORS = {
    "Normal":  "#2ECC71",
    "Warning": "#F39C12",
    "Danger":  "#E67E22",
    "Extreme": "#E74C3C"
}

def get_alert(temp):
    if temp >= 47:
        return "Extreme", "Ghar se bahar mat niklein!"
    elif temp >= 44:
        return "Danger", "Sirf zaroorat pe bahar jayen!"
    elif temp >= 40:
        return "Warning", "Paani zyada piyen, dhoop se bachein!"
    else:
        return "Normal", "Koi risk nahi"

@st.cache_data(show_spinner=False)
def download_data(city, lat, lon):
    url = (
        f"https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters=T2M&community=RE"
        f"&longitude={lon}&latitude={lat}"
        f"&start=19810101&end=20250422&format=JSON"
    )
    r = requests.get(url, timeout=60)
    temp_data = r.json()["properties"]["parameter"]["T2M"]
    df = pd.DataFrame(list(temp_data.items()), columns=["date","temperature"])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df[df["temperature"] > -90].reset_index(drop=True)
    return df

@st.cache_resource(show_spinner=False)
def train_model(city):
    lat, lon = CITIES[city]
    df = download_data(city, lat, lon)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[["temperature"]]).flatten()
    X, y = [], []
    for i in range(len(scaled) - 30 - 7 + 1):
        X.append(scaled[i:i+30])
        y.append(scaled[i+30:i+30+7])
    X, y = np.array(X), np.array(y)
    split = int(0.8 * len(X))
    model = MultiOutputRegressor(
        GradientBoostingRegressor(n_estimators=100, max_depth=4)
    )
    model.fit(X[:split], y[:split])
    return model, scaler, df

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(page_title="Pakistan Heat Wave Warning", page_icon="🌡️", layout="wide")
st.title("🌡️ Pakistan Heat Wave Early Warning System")
st.markdown("**AI-Powered 7-Day Forecast — NASA Data & Machine Learning**")
st.divider()

# ── PAKISTAN MAP (hamesha dikhe) ──────────────────────────────
st.subheader("🗺️ Pakistan Heat Wave Map — Live Status")
st.markdown("Saari cities ka current alert — circle pe hover karo!")

with st.spinner("Saari cities ka data load ho raha hai..."):
    map_data = []
    for map_city, (lat, lon) in CITIES.items():
        try:
            mdl, scl, df_c = train_model(map_city)
            last_30 = scl.transform(df_c[["temperature"]].tail(30)).flatten()
            pred = mdl.predict([last_30])[0]
            temps = scl.inverse_transform(pred.reshape(-1,1)).flatten()
            max_temp = max(temps)
            level, msg = get_alert(max_temp)
            color_map = {
                "Normal":  "#2ECC71",
                "Warning": "#F39C12",
                "Danger":  "#E67E22",
                "Extreme": "#E74C3C"
            }
            map_data.append({
                "city": map_city,
                "lat": lat,
                "lon": lon,
                "temp": max_temp,
                "level": level,
                "color": color_map[level],
                "msg": msg
            })
        except:
            pass

fig_map = go.Figure()
for d in map_data:
    fig_map.add_trace(go.Scattergeo(
        lat=[d["lat"]],
        lon=[d["lon"]],
        mode="markers+text",
        marker=dict(size=20, color=d["color"], opacity=0.8,
                    line=dict(width=2, color="white")),
        text=d["city"],
        textposition="top center",
        hovertemplate=(
            f"<b>{d['city']}</b><br>"
            f"Max Temp: {d['temp']:.1f}°C<br>"
            f"Alert: {d['level']}<br>"
            f"{d['msg']}<extra></extra>"
        ),
        name=d["city"]
    ))

fig_map.update_layout(
    geo=dict(
        scope="asia",
        center=dict(lat=30, lon=69),
        projection_scale=4,
        showland=True, landcolor="#F5F5F5",
        showocean=True, oceancolor="#AED6F1",
        showcoastlines=True, coastlinecolor="#BDC3C7",
        showcountries=True, countrycolor="#BDC3C7",
    ),
    height=500,
    margin=dict(l=0, r=0, t=0, b=0),
    showlegend=False,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)"
)

st.plotly_chart(fig_map, use_container_width=True)

col1, col2, col3, col4 = st.columns(4)
col1.markdown("🟢 **Normal** — <40°C", unsafe_allow_html=True)
col2.markdown("🟡 **Warning** — 40-44°C", unsafe_allow_html=True)
col3.markdown("🟠 **Danger** — 44-47°C", unsafe_allow_html=True)
col4.markdown("🔴 **Extreme** — >47°C", unsafe_allow_html=True)

# ── CITY FORECAST ─────────────────────────────────────────────
st.divider()
st.subheader("🔍 City Ka Detailed Forecast")

city = st.selectbox("City select karo:", list(CITIES.keys()))
predict_btn = st.button("🔍 Forecast Dekho!")

if predict_btn:
    with st.spinner(f"{city} ka forecast ban raha hai..."):
        model, scaler, df = train_model(city)
        last_30 = scaler.transform(df[["temperature"]].tail(30)).flatten()
        pred_scaled = model.predict([last_30])[0]
        pred_temps = scaler.inverse_transform(pred_scaled.reshape(-1,1)).flatten()

    st.subheader(f"📊 {city} — Aglay 7 Din Ka Forecast")
    cols = st.columns(7)
    for i, (col, temp) in enumerate(zip(cols, pred_temps)):
        level, msg = get_alert(temp)
        color = COLORS[level]
        with col:
            st.markdown(f"""
            <div style="background:{color}22;border:2px solid {color};
                border-radius:10px;padding:10px;text-align:center;">
                <div style="font-size:13px;font-weight:bold;">Din {i+1}</div>
                <div style="font-size:22px;font-weight:bold;color:{color};">{temp:.1f}°C</div>
                <div style="font-size:11px;">{level}</div>
            </div>""", unsafe_allow_html=True)

    st.divider()
    fig, ax = plt.subplots(figsize=(10, 4))
    days = [f"Din {i+1}" for i in range(7)]
    bar_colors = [COLORS[get_alert(t)[0]] for t in pred_temps]
    bars = ax.bar(days, pred_temps, color=bar_colors, edgecolor="white", linewidth=1.5)
    ax.axhline(y=40, color="#F39C12", linestyle="--", linewidth=1.5, label="Warning 40C")
    ax.axhline(y=44, color="#E67E22", linestyle="--", linewidth=1.5, label="Danger 44C")
    ax.axhline(y=47, color="#E74C3C", linestyle="--", linewidth=1.5, label="Extreme 47C")
    for bar, temp in zip(bars, pred_temps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.3,
                f"{temp:.1f}C", ha="center", fontsize=10, fontweight="bold")
    ax.set_title(f"{city} — 7-Day Forecast", fontsize=13, fontweight="bold")
    ax.set_ylabel("Temperature (C)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    st.pyplot(fig)

    st.divider()
    alert_rows = []
    for i, temp in enumerate(pred_temps):
        level, msg = get_alert(temp)
        alert_rows.append({"Din": f"Din {i+1}", "Temp": f"{temp:.1f}C", "Alert": level, "Message": msg})
    st.dataframe(pd.DataFrame(alert_rows), use_container_width=True, hide_index=True)

    max_temp = max(pred_temps)
    max_level, max_msg = get_alert(max_temp)
    max_color = COLORS[max_level]
    st.markdown(f"""
    <div style="background:{max_color}22;border:2px solid {max_color};
        border-radius:12px;padding:15px;text-align:center;margin-top:10px;">
        <h3 style="color:{max_color};">Overall Risk: {max_level}</h3>
        <p>{max_msg}</p><p>Max: <b>{max_temp:.1f}C</b></p>
    </div>""", unsafe_allow_html=True)
