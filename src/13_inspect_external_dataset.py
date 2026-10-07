from pathlib import Path

import pandas as pd


DATA_PATH = Path(
    "data/processed/bundesliga_2015_2016_external_validation.csv"
)


def main():

    print("Loading external validation dataset...\n")

    data = pd.read_csv(
        DATA_PATH
    )

    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print(
        f"Total shots: "
        f"{len(data)}"
    )

    print(
        f"Total goals: "
        f"{data['goal'].sum()}"
    )

    print(
        f"Unique matches: "
        f"{data['match_id'].nunique()}"
    )

    print(
        f"Unique teams: "
        f"{data['team'].nunique()}"
    )

    print(
        f"Average shots per match: "
        f"{len(data) / data['match_id'].nunique():.2f}"
    )

    print("\n")
    print("=" * 60)
    print("TEAMS")
    print("=" * 60)

    teams = (
        data["team"]
        .value_counts()
        .sort_index()
    )

    print(
        teams.to_string()
    )

    print("\n")
    print("=" * 60)
    print("SHOTS PER MATCH")
    print("=" * 60)

    shots_per_match = (
        data
        .groupby("match_id")
        .size()
    )

    print(
        shots_per_match
        .describe()
        .round(2)
        .to_string()
    )

    print("\n")
    print("=" * 60)
    print("GOALS PER MATCH")
    print("=" * 60)

    goals_per_match = (
        data
        .groupby("match_id")["goal"]
        .sum()
    )

    print(
        goals_per_match
        .describe()
        .round(2)
        .to_string()
    )


if __name__ == "__main__":
    main()