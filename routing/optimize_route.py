import pandas as pd
import osmnx as ox
import networkx as nx
from ortools.constraint_solver import pywrapcp, routing_enums_pb2


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
    output_path="data/optimized_route.csv",
    num_vehicles=1,
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

    reachable_rows = []
    unreachable_bin_ids = []
    for _, row in critical_bins.iterrows():
        node = row["node"]
        try:
            nx.shortest_path_length(graph, depot_node, node, weight="length")
            nx.shortest_path_length(graph, node, depot_node, weight="length")
            reachable_rows.append(row.to_dict())
        except nx.NetworkXNoPath:
            unreachable_bin_ids.append(row["bin_id"])

    if not reachable_rows:
        pd.DataFrame(
            columns=[
                "vehicle_id",
                "order",
                "bin_id",
                "node",
                "distance_from_previous_m",
                "cumulative_distance_m",
            ]
        ).to_csv(output_path, index=False)
        return {
            "route_df": pd.DataFrame(),
            "total_distance_m": 0.0,
            "return_to_depot_m": 0.0,
            "critical_count": 0,
            "unreachable_bins": unreachable_bin_ids,
            "solver": "ortools_vrp",
        }

    reachable_bins = pd.DataFrame(reachable_rows).reset_index(drop=True)
    nodes = [depot_node] + reachable_bins["node"].tolist()
    node_count = len(nodes)
    large_penalty = 10_000_000

    distance_matrix = [[0] * node_count for _ in range(node_count)]
    for i in range(node_count):
        for j in range(node_count):
            if i == j:
                continue
            try:
                d = nx.shortest_path_length(graph, nodes[i], nodes[j], weight="length")
                distance_matrix[i][j] = int(round(d))
            except nx.NetworkXNoPath:
                distance_matrix[i][j] = large_penalty

    vehicle_count = max(1, min(int(num_vehicles), len(reachable_bins)))
    manager = pywrapcp.RoutingIndexManager(node_count, vehicle_count, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_params.time_limit.seconds = 15

    solution = routing.SolveWithParameters(search_params)
    if solution is None:
        raise RuntimeError("Le solveur VRP n'a pas trouvé de solution.")

    route_rows = []
    total_distance = 0.0
    total_return_to_depot = 0.0

    for vehicle_id in range(vehicle_count):
        index = routing.Start(vehicle_id)
        order = 1
        cumulative = 0.0

        while not routing.IsEnd(index):
            next_index = solution.Value(routing.NextVar(index))
            from_node_idx = manager.IndexToNode(index)
            to_node_idx = manager.IndexToNode(next_index)
            leg_distance = float(distance_matrix[from_node_idx][to_node_idx])

            if to_node_idx != 0:
                cumulative += leg_distance
                total_distance += leg_distance
                route_rows.append(
                    {
                        "vehicle_id": vehicle_id + 1,
                        "order": order,
                        "bin_id": reachable_bins.iloc[to_node_idx - 1]["bin_id"],
                        "node": nodes[to_node_idx],
                        "distance_from_previous_m": leg_distance,
                        "cumulative_distance_m": cumulative,
                    }
                )
                order += 1
            elif from_node_idx != 0:
                total_distance += leg_distance
                total_return_to_depot += leg_distance

            index = next_index

    route_df = pd.DataFrame(route_rows)
    if not route_df.empty:
        route_df = route_df.sort_values(["vehicle_id", "order"]).reset_index(drop=True)
    route_df.to_csv(output_path, index=False)

    return {
        "route_df": route_df,
        "total_distance_m": round(total_distance, 2),
        "return_to_depot_m": round(total_return_to_depot, 2),
        "critical_count": len(route_df),
        "unreachable_bins": unreachable_bin_ids,
        "solver": "ortools_vrp",
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
