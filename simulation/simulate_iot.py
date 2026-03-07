import pandas as pd


def assign_status(fill_level):
    if fill_level >= 80:
        return "critique"
    elif fill_level >= 50:
        return "a_surveille"
    return "normal"


def generate_bins_with_status(
    input_path="data/bins.csv",
    output_path="data/bins_with_status.csv"
):
    bins = pd.read_csv(input_path)
    bins["status"] = bins["fill_level"].apply(assign_status)
    bins.to_csv(output_path, index=False)
    return bins


def main():
    bins = generate_bins_with_status()
    print("Données IoT simulées générées avec succès.\n")
    print(bins)

    print("\nRépartition des statuts :")
    print(bins["status"].value_counts())


if __name__ == "__main__":
    main()
