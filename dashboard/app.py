from pathlib import Path
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


st.set_page_config(page_title="Smart Waste Dashboard", layout="wide")
st.title("Smart Waste Dashboard")

bins_path = Path("data/bins_with_status.csv")
route_path = Path("data/optimized_route.csv")
pred_path = Path("data/fill_predictions.csv")
map_path = Path("maps/smart_waste_map.html")

if not bins_path.exists():
    st.error("Le fichier data/bins_with_status.csv est introuvable. Lance d'abord `python main.py`.")
    st.stop()

bins = pd.read_csv(bins_path)
route = pd.read_csv(route_path) if route_path.exists() else pd.DataFrame()
pred = pd.read_csv(pred_path) if pred_path.exists() else pd.DataFrame()

critical_count = int((bins["status"] == "critique").sum())
monitor_count = int((bins["status"] == "a_surveille").sum())
normal_count = int((bins["status"] == "normal").sum())

if "distance_from_previous_m" in route.columns:
    route_distance = float(route["distance_from_previous_m"].sum())
elif "distance_from_previous" in route.columns:
    route_distance = float(route["distance_from_previous"].sum())
else:
    route_distance = 0.0

high_risk_j3 = int((pred["overflow_risk_j3"] == "eleve").sum()) if not pred.empty else 0

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Poubelles totales", len(bins))
col2.metric("Critiques", critical_count)
col3.metric("A surveiller", monitor_count)
col4.metric("Distance estimee (sans retour)", f"{route_distance:.0f} m")
col5.metric("Risque eleve J+3", high_risk_j3)

st.subheader("Etat des poubelles")
st.dataframe(bins, use_container_width=True)

st.subheader("Ordre de collecte")
if route.empty:
    st.info("Aucun trajet disponible.")
else:
    st.dataframe(route, use_container_width=True)

st.subheader("Predictions IA remplissage")
if pred.empty:
    st.info("Aucune prediction disponible. Lance `python main.py`.")
else:
    st.dataframe(pred, use_container_width=True)

st.subheader("Carte")
if map_path.exists():
    html_content = map_path.read_text(encoding="utf-8")
    components.html(html_content, height=600, scrolling=True)
else:
    st.warning("Carte non générée. Lance `python main.py`.")
