import pandas as pd
import osmnx as ox
import networkx as nx


def build_drive_graph(depot_lat, depot_lon, bins_df):
    all_lats = [depot_lat] + bins_df["latitude"].tolist()
    all_lons = [depot_lon] + bins_df["longitude"].tolist()
    lat_range = max(all_lats) - min(all_lats)
    lon_range = max(all_lons) - min(all_lons)

    # Rough city-scale radius in meters with a safety margin.
    dist_m = max(1500, int(max(lat_range, lon_range) * 111_000 * 1.8))

    ox.settings.use_cache = True
    ox.settings.cache_folder = "cache"
    return ox.graph_from_point((depot_lat, depot_lon), dist=dist_m, network_type="drive")


def optimize_route(
    bins_with_status_path="data/bins_with_status.csv",
    depot_path="data/depot.csv",
    output_path="data/optimized_route.csv"
):
    bins = pd.read_csv(bins_with_status_path)
    depot = pd.read_csv(depot_path)

    critical_bins = bins[bins["status"] == "critique"].copy()
    if critical_bins.empty:
        pd.DataFrame(
            columns=["order", "bin_id", "node", "distance_from_previous_m", "cumulative_distance_m"]
        ).to_csv(output_path, index=False)
        return {
            "route_df": pd.DataFrame(),
            "total_distance_m": 0.0,
            "return_to_depot_m": 0.0,
            "critical_count": 0,
        }

    print("Poubelles critiques détectées :")
    print(critical_bins[["bin_id", "fill_level", "status"]])

    depot_lat = depot.loc[0, "latitude"]
    depot_lon = depot.loc[0, "longitude"]
    print("\nChargement du réseau routier local OpenStreetMap...")
    graph = build_drive_graph(depot_lat, depot_lon, critical_bins)
    print("Réseau routier prêt.")

    depot_node = ox.nearest_nodes(graph, X=depot_lon, Y=depot_lat)
    critical_bins["node"] = critical_bins.apply(
        lambda row: ox.nearest_nodes(graph, X=row["longitude"], Y=row["latitude"]),
        axis=1,
    )

    remaining = critical_bins[["bin_id", "node"]].values.tolist()
    current_node = depot_node
    route_order = []
    total_distance = 0.0

    while remaining:
        nearest = None
        nearest_distance = float("inf")

        for bin_id, node in remaining:
            try:
                dist = nx.shortest_path_length(graph, current_node, node, weight="length")
            except nx.NetworkXNoPath:
                continue
            if dist < nearest_distance:
                nearest_distance = dist
                nearest = (bin_id, node)

        if nearest is None:
            break

        total_distance += nearest_distance
        route_order.append((nearest[0], nearest[1], nearest_distance, total_distance))
        current_node = nearest[1]
        remaining.remove([nearest[0], nearest[1]])

    return_to_depot = 0.0
    if route_order:
        try:
            return_to_depot = nx.shortest_path_length(graph, current_node, depot_node, weight="length")
            total_distance += return_to_depot
        except nx.NetworkXNoPath:
            return_to_depot = 0.0

    route_df = pd.DataFrame(
        route_order,
        columns=["bin_id", "node", "distance_from_previous_m", "cumulative_distance_m"],
    )
    if not route_df.empty:
        route_df.insert(0, "order", range(1, len(route_df) + 1))
    route_df.to_csv(output_path, index=False)

    return {
        "route_df": route_df,
        "total_distance_m": round(total_distance, 2),
        "return_to_depot_m": round(return_to_depot, 2),
        "critical_count": len(route_df),
    }


def main():
    result = optimize_route()
    route_df = result["route_df"]
    if route_df.empty:
        print("Aucune poubelle critique. Aucun trajet à générer.")
        return

    print("\nOrdre optimisé de collecte :")
    for _, step in route_df.iterrows():
        print(
            f"{step['order']}. {step['bin_id']} -> "
            f"{step['distance_from_previous_m']:.2f} m depuis le point précédent "
            f"(cumul {step['cumulative_distance_m']:.2f} m)"
        )

    print(f"\nRetour au dépôt : {result['return_to_depot_m']:.2f} m")
    print(f"Distance totale (avec retour) : {result['total_distance_m']:.2f} m")
    print("\nTrajet sauvegardé dans data/optimized_route.csv")


if __name__ == "__main__":
    main()
