from statsbombpy import sb


MATCH_ID = 3754079


def main():
    print("Loading shot features...\n")

    events = sb.events(match_id=MATCH_ID)

    shots = events[events["type"] == "Shot"].copy()

    print(f"Total shots: {len(shots)}\n")

    # Βασικές στήλες που μπορεί να μας ενδιαφέρουν
    base_columns = [
        "location",
        "under_pressure",
        "play_pattern",
    ]

    # Όλες οι στήλες που αφορούν shots
    shot_columns = [
        column
        for column in shots.columns
        if column.startswith("shot_")
    ]

    columns_to_check = base_columns + shot_columns

    print("AVAILABLE SHOT FEATURES")
    print("-" * 70)

    for column in columns_to_check:

        if column not in shots.columns:
            continue

        non_null = shots[column].notna().sum()

        print(
            f"{column:<35}"
            f"{non_null:>3}/{len(shots)} values"
        )

    print("\n")
    print("CATEGORICAL VALUES")
    print("-" * 70)

    categorical_columns = [
        "under_pressure",
        "play_pattern",
        "shot_body_part",
        "shot_technique",
        "shot_type",
        "shot_outcome",
        "shot_first_time",
        "shot_one_on_one",
        "shot_open_goal",
    ]

    for column in categorical_columns:

        if column not in shots.columns:
            continue

        values = (
            shots[column]
            .dropna()
            .astype(str)
            .unique()
        )

        print(f"\n{column}:")
        print(values)


if __name__ == "__main__":
    main()