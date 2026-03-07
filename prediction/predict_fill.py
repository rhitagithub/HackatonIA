from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


def status_from_fill(fill_level: float) -> str:
    if fill_level >= 80:
        return "critique"
    if fill_level >= 50:
        return "a_surveille"
    return "normal"


def _bootstrap_history(bins_df: pd.DataFrame, days: int, end_date: date) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range(end=end_date, periods=days, freq="D")
    records = []

    for _, row in bins_df.iterrows():
        base = float(row["fill_level"])
        daily_growth = float(rng.uniform(2.0, 7.0))
        history_vals = []

        current = base
        for _ in range(days):
            noise = float(rng.normal(0, 2.0))
            current = float(np.clip(current - daily_growth + noise, 0, 100))
            history_vals.append(current)

        history_vals[-1] = base
        for d, fill in zip(dates, history_vals):
            records.append(
                {
                    "date": d.date().isoformat(),
                    "bin_id": row["bin_id"],
                    "fill_level": round(float(fill), 2),
                }
            )

    return pd.DataFrame(records)


def _ensure_history(
    bins_path: str = "data/bins.csv",
    history_path: str = "data/fill_history.csv",
    snapshot_date: date | None = None,
) -> pd.DataFrame:
    bins = pd.read_csv(bins_path)
    if snapshot_date is None:
        snapshot_date = date.today()

    history_file = Path(history_path)
    if history_file.exists():
        history = pd.read_csv(history_file)
    else:
        history = _bootstrap_history(bins, days=30, end_date=snapshot_date - timedelta(days=1))

    snapshot = bins[["bin_id", "fill_level"]].copy()
    snapshot["date"] = snapshot_date.isoformat()
    history = pd.concat([history, snapshot[["date", "bin_id", "fill_level"]]], ignore_index=True)
    history = history.drop_duplicates(subset=["date", "bin_id"], keep="last")
    history["fill_level"] = history["fill_level"].clip(0, 100)
    history = history.sort_values(["bin_id", "date"]).reset_index(drop=True)
    history.to_csv(history_path, index=False)
    return history


def _build_training_set(history: pd.DataFrame) -> pd.DataFrame:
    h = history.copy()
    h["date"] = pd.to_datetime(h["date"])
    h = h.sort_values(["bin_id", "date"])

    grouped = h.groupby("bin_id")
    h["lag1"] = grouped["fill_level"].shift(1)
    h["lag2"] = grouped["fill_level"].shift(2)
    h["lag3"] = grouped["fill_level"].shift(3)
    h["roll3"] = grouped["fill_level"].shift(1).rolling(3).mean().reset_index(level=0, drop=True)
    h["day_of_week"] = h["date"].dt.dayofweek
    h["target_next"] = grouped["fill_level"].shift(-1)

    features = ["lag1", "lag2", "lag3", "roll3", "day_of_week"]
    train = h.dropna(subset=features + ["target_next"]).copy()
    return train


def _train_model(train_df: pd.DataFrame) -> tuple[RandomForestRegressor, float | None]:
    x_cols = ["lag1", "lag2", "lag3", "roll3", "day_of_week"]
    x = train_df[x_cols]
    y = train_df["target_next"]

    if len(train_df) >= 30:
        split = int(len(train_df) * 0.8)
        x_train, x_test = x.iloc[:split], x.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        model = RandomForestRegressor(n_estimators=300, random_state=42)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        mae = float(mean_absolute_error(y_test, y_pred)) if len(y_test) else None
    else:
        model = RandomForestRegressor(n_estimators=300, random_state=42)
        model.fit(x, y)
        mae = None

    return model, mae


def _predict_for_bin(
    model: RandomForestRegressor,
    bin_history: pd.DataFrame,
    start_day_of_week: int,
) -> tuple[float, float]:
    recent = bin_history.sort_values("date")["fill_level"].tail(3).astype(float).tolist()
    if len(recent) < 3:
        recent = ([recent[0]] * (3 - len(recent))) + recent if recent else [0.0, 0.0, 0.0]

    preds = []
    for step in (1, 2, 3):
        lag1, lag2, lag3 = recent[-1], recent[-2], recent[-3]
        roll3 = float(np.mean([lag1, lag2, lag3]))
        dow = (start_day_of_week + step) % 7
        x_next = pd.DataFrame([{"lag1": lag1, "lag2": lag2, "lag3": lag3, "roll3": roll3, "day_of_week": dow}])
        pred = float(np.clip(model.predict(x_next)[0], 0, 100))
        preds.append(pred)
        recent.append(pred)

    return round(preds[0], 2), round(preds[2], 2)


def predict_fill_levels(
    bins_path: str = "data/bins.csv",
    history_path: str = "data/fill_history.csv",
    output_path: str = "data/fill_predictions.csv",
) -> dict:
    history = _ensure_history(bins_path=bins_path, history_path=history_path)
    train_df = _build_training_set(history)

    if train_df.empty:
        raise RuntimeError("Historique insuffisant pour entraîner un modèle de prédiction.")

    model, mae = _train_model(train_df)
    bins = pd.read_csv(bins_path)
    history["date"] = pd.to_datetime(history["date"])
    last_date = history["date"].max().date()
    start_dow = int(last_date.weekday())

    rows = []
    for _, bin_row in bins.iterrows():
        bin_id = bin_row["bin_id"]
        bin_hist = history[history["bin_id"] == bin_id].copy()
        pred_j1, pred_j3 = _predict_for_bin(model, bin_hist, start_dow)
        risk = "eleve" if pred_j3 >= 95 else "moyen" if pred_j3 >= 80 else "faible"
        rows.append(
            {
                "bin_id": bin_id,
                "pred_fill_j1": pred_j1,
                "pred_fill_j3": pred_j3,
                "pred_status_j1": status_from_fill(pred_j1),
                "pred_status_j3": status_from_fill(pred_j3),
                "overflow_risk_j3": risk,
            }
        )

    pred_df = pd.DataFrame(rows).sort_values("bin_id").reset_index(drop=True)
    pred_df.to_csv(output_path, index=False)

    return {
        "predictions": pred_df,
        "history_points": int(len(history)),
        "model_mae": round(mae, 2) if mae is not None else None,
        "history_path": history_path,
        "output_path": output_path,
        "last_snapshot_date": last_date.isoformat(),
    }


def main():
    result = predict_fill_levels()
    print(f"Predictions sauvegardees: {result['output_path']}")
    print(f"Historique: {result['history_path']} ({result['history_points']} lignes)")
    if result["model_mae"] is not None:
        print(f"MAE validation: {result['model_mae']}")
    print(result["predictions"])


if __name__ == "__main__":
    main()
