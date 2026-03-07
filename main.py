import argparse
import time
from pathlib import Path

import pandas as pd

from simulation.simulate_iot import apply_collection_reset, generate_bins_with_status
from prediction.predict_fill import predict_fill_levels
from routing.optimize_route import optimize_route
from maps.visualize_map import create_map


def run_batch_pipeline():
    print("1) Simulation IoT...")
    bins = generate_bins_with_status()
    print(f"   - {len(bins)} poubelles traitées")

    print("2) Prediction IA (J+1 / J+3)...")
    pred_result = predict_fill_levels()
    print(f"   - fichier predictions: {pred_result['output_path']}")
    if pred_result["model_mae"] is not None:
        print(f"   - MAE validation modele: {pred_result['model_mae']}")

    print("3) Optimisation de la tournée...")
    route_result = optimize_route()
    print(f"   - {route_result['critical_count']} poubelles critiques dans la tournée")
    print(f"   - distance totale estimée (avec retour): {route_result['total_distance_m']:.2f} m")

    print("4) Génération de la carte...")
    output_map = create_map()
    print(f"   - carte: {output_map}")

    print("5) Mise à jour post-collecte...")
    reset_result = apply_collection_reset()
    print(f"   - poubelles vidées après passage camion: {reset_result['collected_count']}")

    print("\nPipeline terminé.")


def run_live_pipeline(
    ticks: int,
    tick_seconds: int,
    simulated_hours_per_tick: float,
    speed_factor: float,
):
    ticks = max(1, int(ticks))
    tick_seconds = max(1, int(tick_seconds))
    simulated_hours_per_tick = max(0.01, float(simulated_hours_per_tick))
    speed_factor = max(0.1, float(speed_factor))

    project_root = Path(__file__).resolve().parent
    route_path = project_root / "data/optimized_route.csv"
    route_snapshot_path = project_root / "data/route_snapshot_bins.csv"

    print(
        f"Mode live (remplissage uniquement): {ticks} ticks, {tick_seconds}s/tick, "
        f"{simulated_hours_per_tick:.2f}h/tick, speed x{speed_factor:.1f}"
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
    if route_snapshot_path.exists():
        route_snapshot_path.unlink()

    for i in range(1, ticks + 1):
        bins = generate_bins_with_status(
            min_elapsed_hours=0.0,
            forced_elapsed_hours=simulated_hours_per_tick,
            speed_factor=speed_factor,
            seed=42 + i,
        )
        if i == 1:
            create_map()

        critical_count = int((bins["status"] == "critique").sum())
        print(
            f"[tick {i}/{ticks}] critiques={critical_count} "
            f"max_fill={float(bins['fill_level'].max()):.1f}%"
        )

        if i < ticks:
            time.sleep(tick_seconds)


def main():
    parser = argparse.ArgumentParser(description="Smart Waste pipeline")
    parser.add_argument("--live", action="store_true", help="Lance une simulation continue des niveaux de remplissage.")
    parser.add_argument("--ticks", type=int, default=120, help="Nombre de ticks en mode live.")
    parser.add_argument("--tick-seconds", type=int, default=5, help="Secondes entre deux ticks en mode live.")
    parser.add_argument(
        "--sim-hours-per-tick",
        type=float,
        default=0.5,
        help="Heures simulees ajoutees a chaque tick en mode live.",
    )
    parser.add_argument(
        "--speed-factor",
        type=float,
        default=6.0,
        help="Multiplicateur de vitesse de remplissage en mode live.",
    )
    args = parser.parse_args()

    if args.live:
        run_live_pipeline(
            ticks=args.ticks,
            tick_seconds=args.tick_seconds,
            simulated_hours_per_tick=args.sim_hours_per_tick,
            speed_factor=args.speed_factor,
        )
    else:
        run_batch_pipeline()


if __name__ == "__main__":
    main()
