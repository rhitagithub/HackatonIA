from pathlib import Path
import sys
import time
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from maps.visualize_map import create_map
from routing.optimize_route import optimize_route
from simulation.simulate_iot import apply_collection_reset


st.set_page_config(page_title="Smart Waste Dashboard", layout="wide")
st.title("Smart Waste Dashboard")

with st.sidebar:
    st.header("Temps reel")
    auto_refresh = st.checkbox("Auto-refresh", value=True)
    refresh_seconds = st.slider("Intervalle (sec)", min_value=2, max_value=30, value=5, step=1)
    max_points = st.slider("Points affiches", min_value=20, max_value=500, value=120, step=10)

bins_path = PROJECT_ROOT / "data/bins_with_status.csv"
route_path = PROJECT_ROOT / "data/optimized_route.csv"
pred_path = PROJECT_ROOT / "data/fill_predictions.csv"
map_path = PROJECT_ROOT / "maps/smart_waste_map.html"
events_path = PROJECT_ROOT / "data/fill_events.csv"
route_snapshot_bins_path = PROJECT_ROOT / "data/route_snapshot_bins.csv"

if not bins_path.exists():
    st.error("Le fichier data/bins_with_status.csv est introuvable. Lance d'abord `python main.py`.")
    st.stop()

bins_live = pd.read_csv(bins_path)
route = pd.read_csv(route_path) if route_path.exists() else pd.DataFrame()
pred = pd.read_csv(pred_path) if pred_path.exists() else pd.DataFrame()
has_snapshot_route = "route_snapshot_at" in st.session_state and route_snapshot_bins_path.exists()
bins = pd.read_csv(route_snapshot_bins_path) if has_snapshot_route else bins_live

critical_count = int((bins["status"] == "critique").sum())
monitor_count = int((bins["status"] == "a_surveille").sum())
normal_count = int((bins["status"] == "normal").sum())
critical_ids = set(bins.loc[bins["status"] == "critique", "bin_id"].astype(str).tolist())
route_ids = set(route["bin_id"].astype(str).tolist()) if "bin_id" in route.columns else set()
missing_in_route = sorted(list(critical_ids - route_ids)) if has_snapshot_route else []

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
if has_snapshot_route:
    st.caption(f"Trajectoire basee sur le snapshot du: {st.session_state['route_snapshot_at']}")

if missing_in_route:
    st.warning(
        f"Attention: la trajectoire ne couvre pas toutes les bennes critiques du snapshot: {', '.join(missing_in_route)}."
    )
if route.empty:
    st.info("Aucun trajet disponible.")
else:
    st.dataframe(route, use_container_width=True)

st.subheader("Predictions IA remplissage")
if pred.empty:
    st.info("Aucune prediction disponible. Lance `python main.py`.")
else:
    st.dataframe(pred, use_container_width=True)

st.subheader("Evolution du remplissage (temps reel)")
if events_path.exists():
    events = pd.read_csv(events_path)
    if not events.empty and {"event_type", "timestamp", "bin_id"}.issubset(events.columns):
        events = events[events["event_type"] == "fill_update"].copy()
        if not events.empty and "fill_level" in events.columns:
            events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True, errors="coerce")
            events["fill_level"] = pd.to_numeric(events["fill_level"], errors="coerce")
            events = events.dropna(subset=["timestamp", "fill_level", "bin_id"]).sort_values("timestamp")
            events = events.tail(max_points)
            chart_df = events.pivot_table(index="timestamp", columns="bin_id", values="fill_level", aggfunc="last")
            if not chart_df.empty:
                st.line_chart(chart_df)
            else:
                st.info("Pas assez de points pour tracer la courbe.")
        else:
            st.info("Aucun evenement de remplissage disponible.")
    else:
        st.info("Le journal d'evenements est vide.")
else:
    st.info("Lance `python main.py --live` pour generer des donnees en continu.")

st.subheader("Carte")
action_col1, action_col2, action_col3 = st.columns([1, 1, 2])
with action_col1:
    if st.button("Voir la trajectoire", use_container_width=True):
        with st.spinner("Calcul de la trajectoire..."):
            bins_live.to_csv(route_snapshot_bins_path, index=False)
            optimize_route(
                bins_with_status_path=str(route_snapshot_bins_path),
                depot_path=str(PROJECT_ROOT / "data/depot.csv"),
                output_path=str(route_path),
            )
            create_map(
                bins_with_status_path=str(route_snapshot_bins_path),
                depot_path=str(PROJECT_ROOT / "data/depot.csv"),
                route_path=str(route_path),
                output_path=str(map_path),
            )
            route_after = pd.read_csv(route_path) if route_path.exists() else pd.DataFrame()
            route_after_ids = set(route_after["bin_id"].astype(str).tolist()) if "bin_id" in route_after.columns else set()
            snapshot_critical = set(
                bins_live.loc[bins_live["status"] == "critique", "bin_id"].astype(str).tolist()
            )
            missing_after = sorted(list(snapshot_critical - route_after_ids))
            st.session_state["route_snapshot_at"] = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S UTC")
            if missing_after:
                st.error(
                    "Trajectoire calculee mais incomplète. "
                    f"Bennes critiques manquantes: {', '.join(missing_after)}."
                )
            else:
                st.success("Trajectoire calculee sur l'etat actuel des poubelles.")
        st.rerun()

with action_col2:
    if st.button("Lancer collecte", type="primary", use_container_width=True):
        with st.spinner("Collecte en cours..."):
            if not route.empty:
                reset_result = apply_collection_reset(
                    route_path=str(route_path),
                    state_path=str(PROJECT_ROOT / "data/bins_state.csv"),
                    output_path=str(bins_path),
                    events_path=str(events_path),
                )
                pd.DataFrame(
                    columns=[
                        "vehicle_id",
                        "order",
                        "bin_id",
                        "node",
                        "distance_from_previous_m",
                        "cumulative_distance_m",
                    ]
                ).to_csv(route_path, index=False)
                if route_snapshot_bins_path.exists():
                    route_snapshot_bins_path.unlink()
                st.session_state.pop("route_snapshot_at", None)
                create_map(
                    bins_with_status_path=str(bins_path),
                    depot_path=str(PROJECT_ROOT / "data/depot.csv"),
                    route_path=str(route_path),
                    output_path=str(map_path),
                )
                st.success(f"Collecte terminee: {reset_result['collected_count']} poubelles videes.")
            else:
                st.info("Aucune poubelle critique a collecter.")
        st.rerun()

if map_path.exists():
    html_content = map_path.read_text(encoding="utf-8")
    map_placeholder = st.empty()
    with map_placeholder.container():
        components.html(html_content, height=600, scrolling=True)
else:
    st.warning("Carte non générée. Lance `python main.py`.")

if auto_refresh:
    time.sleep(refresh_seconds)
    st.rerun()
