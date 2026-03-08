import pandas as pd
import folium
import osmnx as ox
import networkx as nx


def _build_graph(depot_lat, depot_lon, bins):
    all_lats = [depot_lat] + bins["latitude"].tolist()
    all_lons = [depot_lon] + bins["longitude"].tolist()
    lat_range = max(all_lats) - min(all_lats)
    lon_range = max(all_lons) - min(all_lons)
    dist_m = max(1500, int(max(lat_range, lon_range) * 111_000 * 1.8))

    ox.settings.use_cache = True
    ox.settings.cache_folder = "cache"
    return ox.graph_from_point((depot_lat, depot_lon), dist=dist_m, network_type="drive")


def create_map(
    bins_with_status_path="data/bins_with_status.csv",
    depot_path="data/depot.csv",
    route_path="data/optimized_route.csv",
    output_path="maps/smart_waste_map.html",
):
    bins = pd.read_csv(bins_with_status_path)
    depot = pd.read_csv(depot_path)
    route_df = pd.read_csv(route_path)

    center_lat = bins["latitude"].mean()
    center_lon = bins["longitude"].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=15)

    depot_lat = depot.loc[0, "latitude"]
    depot_lon = depot.loc[0, "longitude"]

    folium.Marker(
        location=[depot_lat, depot_lon],
        popup="Dépôt Camion",
        icon=folium.Icon(color="blue", icon="home")
    ).add_to(m)

    for _, row in bins.iterrows():
        if row["status"] == "critique":
            color = "red"
        elif row["status"] == "a_surveille":
            color = "orange"
        else:
            color = "green"

        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            popup=f"{row['bin_id']} - {row['status']} - {row['fill_level']}%",
            icon=folium.Icon(color=color)
        ).add_to(m)

    critical_count = int((bins["status"] == "critique").sum())
    monitored_count = int((bins["status"] == "a_surveille").sum())
    normal_count = int((bins["status"] == "normal").sum())

    # Keep one single source of truth for displayed distance: route CSV legs.
    total_distance_from_file = 0.0
    if "distance_from_previous_m" in route_df.columns:
        total_distance_from_file += route_df["distance_from_previous_m"].sum()
    elif "distance_from_previous" in route_df.columns:
        total_distance_from_file += route_df["distance_from_previous"].sum()

    if not route_df.empty:
        graph = _build_graph(depot_lat, depot_lon, bins)
        depot_node = ox.nearest_nodes(graph, X=depot_lon, Y=depot_lat)
        route_nodes = [depot_node]
        bins_ids = set(bins["bin_id"])

        for _, route_row in route_df.iterrows():
            if route_row["bin_id"] not in bins_ids:
                continue
            bin_row = bins[bins["bin_id"] == route_row["bin_id"]].iloc[0]
            route_nodes.append(ox.nearest_nodes(graph, X=bin_row["longitude"], Y=bin_row["latitude"]))

        if len(route_nodes) > 1:
            route_nodes.append(depot_node)
            for i in range(len(route_nodes) - 1):
                try:
                    path = nx.shortest_path(graph, route_nodes[i], route_nodes[i + 1], weight="length")
                    path_coords = [(graph.nodes[n]["y"], graph.nodes[n]["x"]) for n in path]
                    folium.PolyLine(path_coords, color="blue", weight=4, opacity=0.8).add_to(m)
                except nx.NetworkXNoPath:
                    continue

    summary_html = f"""
    <div style="
        position: fixed;
        top: 10px;
        left: 10px;
        z-index: 9999;
        background: white;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 8px;
        font-size: 13px;
    ">
        <b>Resume collecte</b><br>
        Critiques: {critical_count}<br>
        A surveiller: {monitored_count}<br>
        Normales: {normal_count}<br>
        Distance estimee: {total_distance_from_file:.0f} m
    </div>
    """
    m.get_root().html.add_child(folium.Element(summary_html))

    m.save(output_path)
    return output_path


def main():
    out = create_map()
    print(f"Carte générée : {out}")


if __name__ == "__main__":
    main()
