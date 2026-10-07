from pathlib import Path

import pandas as pd
from statsbombpy import sb


OUTPUT_PATH = Path("data/raw/premier_league_2015_2016_shots.csv")


def main():
    print("Loading all Premier League 2015/2016 events...")
    print("This may take a little time.\n")

    events = sb.competition_events(
        country="England",
        division="Premier League",
        season="2015/2016",
        gender="male",
    )

    print(f"Total events loaded: {len(events)}")

    shots = events[events["type"] == "Shot"].copy()

    print(f"Total shots found: {len(shots)}")

    columns_to_keep = [
        "match_id",
        "minute",
        "team",
        "player",
        "location",
        "under_pressure",
        "play_pattern",
        "shot_body_part",
        "shot_technique",
        "shot_type",
        "shot_first_time",
        "shot_one_on_one",
        "shot_open_goal",
        "shot_outcome",
        "shot_statsbomb_xg",
    ]

    available_columns = [
        column
        for column in columns_to_keep
        if column in shots.columns
    ]

    shots_clean = shots[available_columns].copy()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    shots_clean.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nDataset saved successfully:")
    print(OUTPUT_PATH)

    print("\n--------------------------------")
    print(f"Total shots: {len(shots_clean)}")
    print("--------------------------------")

    print("\nShot outcomes:")
    print(
        shots_clean["shot_outcome"]
        .value_counts(dropna=False)
    )


if __name__ == "__main__":
    main()