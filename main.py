from simulation.simulate_iot import generate_bins_with_status
from prediction.predict_fill import predict_fill_levels
from routing.optimize_route import optimize_route
from maps.visualize_map import create_map


def main():
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

    print("\nPipeline terminé.")


if __name__ == "__main__":
    main()
