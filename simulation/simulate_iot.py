from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def assign_status(fill_level):
    if fill_level >= 80:
        return "critique"
    elif fill_level >= 50:
        return "a_surveille"
    return "normal"


def _project_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[1] / path


def _current_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _zone_base_rate(zone: str) -> float:
    rates = {
        "Marche": 1.6,
        "Centre": 1.1,
        "Residentiel": 0.7,
    }
    return rates.get(str(zone), 1.0)


def _initialize_state(bins_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    state = bins_df.copy()
    zone_start_fill = {
        "Marche": 45.0,
        "Centre": 32.0,
        "Residentiel": 24.0,
    }
    base_fill = state["zone"].map(zone_start_fill).fillna(30.0).astype(float)
    state["fill_level"] = np.clip(base_fill + rng.normal(0, 6.0, len(state)), 8, 75)
    critical_count = min(3, len(state))
    critical_indices = rng.choice(state.index.to_numpy(), size=critical_count, replace=False)
    state.loc[critical_indices, "fill_level"] = rng.uniform(82.0, 97.0, size=critical_count)
    state["fill_level"] = state["fill_level"].round(2)
    state["growth_rate_per_hour"] = state["zone"].apply(
        lambda z: round(float(np.clip(_zone_base_rate(z) + rng.normal(0, 0.2), 0.3, 2.2)), 3)
    )
    state["last_update"] = _current_timestamp()
    state["last_collected_at"] = pd.NA
    return state


def _load_or_create_state(input_path: str, state_path: str, seed: int = 42) -> pd.DataFrame:
    bins_path = _project_path(input_path)
    state_file = _project_path(state_path)
    bins = pd.read_csv(bins_path)

    if state_file.exists():
        state = pd.read_csv(state_file)
        base_cols = ["bin_id", "latitude", "longitude", "zone"]
        metadata_cols = ["latitude", "longitude", "zone"]
        state = state.drop(columns=[c for c in metadata_cols if c in state.columns], errors="ignore")
        state = bins[base_cols].merge(state, on="bin_id", how="left")

        missing_rate = state["growth_rate_per_hour"].isna() if "growth_rate_per_hour" in state.columns else True
        if isinstance(missing_rate, bool) or missing_rate.any():
            init = _initialize_state(bins, seed=seed).set_index("bin_id")
            if "growth_rate_per_hour" not in state.columns:
                state["growth_rate_per_hour"] = np.nan
            state.loc[state["growth_rate_per_hour"].isna(), "growth_rate_per_hour"] = state.loc[
                state["growth_rate_per_hour"].isna(), "bin_id"
            ].map(init["growth_rate_per_hour"])

        if "fill_level" not in state.columns:
            state["fill_level"] = bins["fill_level"].astype(float)
        state["fill_level"] = state["fill_level"].fillna(bins["fill_level"]).astype(float).clip(0, 100)

        if "last_update" not in state.columns:
            state["last_update"] = _current_timestamp()
        state["last_update"] = state["last_update"].fillna(_current_timestamp())

        if "last_collected_at" not in state.columns:
            state["last_collected_at"] = pd.NA
        state["last_collected_at"] = state["last_collected_at"].astype("string")
    else:
        state = _initialize_state(bins, seed=seed)

    return state


def _save_state(state_df: pd.DataFrame, state_path: str) -> None:
    path = _project_path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    state_df.to_csv(tmp, index=False)
    tmp.replace(path)


def _append_events(events_rows: list[dict], events_path: str, max_rows: int = 5000) -> None:
    if not events_rows:
        return
    events_file = _project_path(events_path)
    events_file.parent.mkdir(parents=True, exist_ok=True)
    event_cols = [
        "timestamp",
        "event_type",
        "bin_id",
        "fill_level",
        "status",
        "fill_level_before",
        "fill_level_after",
    ]
    new_events = pd.DataFrame(events_rows)
    for col in event_cols:
        if col not in new_events.columns:
            new_events[col] = pd.NA
    new_events = new_events[event_cols]
    if events_file.exists():
        old = pd.read_csv(events_file)
        for col in event_cols:
            if col not in old.columns:
                old[col] = pd.NA
        old = old[event_cols]
        out = pd.concat([old.astype("object"), new_events.astype("object")], ignore_index=True)
    else:
        out = new_events
    if max_rows and len(out) > int(max_rows):
        out = out.tail(int(max_rows)).reset_index(drop=True)
    tmp = events_file.with_suffix(events_file.suffix + ".tmp")
    out.to_csv(tmp, index=False)
    tmp.replace(events_file)


def _build_snapshot(state_df: pd.DataFrame, output_path: str) -> pd.DataFrame:
    snapshot = state_df[["bin_id", "latitude", "longitude", "zone", "fill_level"]].copy()
    snapshot["fill_level"] = snapshot["fill_level"].round(2)
    snapshot["status"] = snapshot["fill_level"].apply(assign_status)
    out_file = _project_path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_file.with_suffix(out_file.suffix + ".tmp")
    snapshot.to_csv(tmp, index=False)
    tmp.replace(out_file)
    return snapshot


def generate_bins_with_status(
    input_path="data/bins.csv",
    output_path="data/bins_with_status.csv",
    state_path="data/bins_state.csv",
    events_path="data/fill_events.csv",
    min_elapsed_hours=1.0,
    forced_elapsed_hours=None,
    speed_factor=1.0,
    max_events_rows=5000,
    seed=42,
):
    state = _load_or_create_state(input_path=input_path, state_path=state_path, seed=seed)
    now = datetime.now(timezone.utc)
    last_update = pd.to_datetime(state["last_update"].astype("string"), utc=True, errors="coerce").max()

    if forced_elapsed_hours is not None:
        elapsed_hours = float(forced_elapsed_hours)
    elif pd.isna(last_update):
        elapsed_hours = float(min_elapsed_hours)
    else:
        elapsed_hours = max((now - last_update.to_pydatetime()).total_seconds() / 3600.0, float(min_elapsed_hours))

    rng = np.random.default_rng(seed + int(now.timestamp()) // 3600)
    noise = rng.normal(0, 0.35, len(state)) * np.sqrt(elapsed_hours)
    speed_factor = max(0.1, float(speed_factor))
    state["fill_level"] = np.clip(
        state["fill_level"].astype(float)
        + state["growth_rate_per_hour"].astype(float) * elapsed_hours * speed_factor
        + noise,
        0,
        100,
    )
    state["last_update"] = now.replace(microsecond=0).isoformat()

    _save_state(state, state_path)
    bins = _build_snapshot(state, output_path)
    _append_events(
        [
            {
                "timestamp": now.replace(microsecond=0).isoformat(),
                "event_type": "fill_update",
                "bin_id": row["bin_id"],
                "fill_level": round(float(row["fill_level"]), 2),
                "status": assign_status(float(row["fill_level"])),
            }
            for _, row in bins.iterrows()
        ],
        events_path=events_path,
        max_rows=max_events_rows,
    )
    return bins


def apply_collection_reset(
    route_path="data/optimized_route.csv",
    state_path="data/bins_state.csv",
    output_path="data/bins_with_status.csv",
    events_path="data/fill_events.csv",
):
    route_file = _project_path(route_path)
    if not route_file.exists():
        return {"collected_count": 0, "collected_bins": []}

    route_df = pd.read_csv(route_file)
    if route_df.empty or "bin_id" not in route_df.columns:
        return {"collected_count": 0, "collected_bins": []}

    state = _load_or_create_state(input_path="data/bins.csv", state_path=state_path)
    collected_ids = [b for b in route_df["bin_id"].dropna().astype(str).unique().tolist() if b in set(state["bin_id"])]
    if not collected_ids:
        return {"collected_count": 0, "collected_bins": []}

    now_iso = _current_timestamp()
    events = []
    for bin_id in collected_ids:
        idx = state.index[state["bin_id"] == bin_id][0]
        fill_before = float(state.at[idx, "fill_level"])
        state.at[idx, "fill_level"] = 0.0
        state.at[idx, "last_collected_at"] = now_iso
        state.at[idx, "last_update"] = now_iso
        events.append(
            {
                "timestamp": now_iso,
                "event_type": "collection_reset",
                "bin_id": bin_id,
                "fill_level_before": round(fill_before, 2),
                "fill_level_after": 0.0,
            }
        )

    _save_state(state, state_path)
    _build_snapshot(state, output_path)
    _append_events(events, events_path=events_path)
    return {"collected_count": len(collected_ids), "collected_bins": collected_ids}


def main():
    bins = generate_bins_with_status()
    print("Données IoT simulées dynamiques mises à jour avec succès.\n")
    print(bins)

    print("\nRépartition des statuts :")
    print(bins["status"].value_counts())


if __name__ == "__main__":
    main()
