from pathlib import Path
import time

import pandas as pd
from statsbombpy import sb


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_PATH = Path(
    "data/raw/multicompetition_training_shots.csv"
)


# ============================================================
# TRAINING COMPETITIONS
#
# Euro 2024 and Copa America 2024 are intentionally excluded.
# They will remain blind external test datasets.
# ============================================================

TRAINING_COMPETITIONS = [
    {
        "name": "Premier League 2015/16",
        "competition_id": 2,
        "season_id": 27,
    },
    {
        "name": "La Liga 2015/16",
        "competition_id": 11,
        "season_id": 27,
    },
    {
        "name": "Ligue 1 2015/16",
        "competition_id": 7,
        "season_id": 27,
    },
    {
        "name": "Serie A 2015/16",
        "competition_id": 12,
        "season_id": 27,
    },
    {
        "name": "World Cup 2018",
        "competition_id": 43,
        "season_id": 3,
    },
    {
        "name": "Euro 2020",
        "competition_id": 55,
        "season_id": 43,
    },
    {
        "name": "World Cup 2022",
        "competition_id": 43,
        "season_id": 106,
    },
]


# ============================================================
# COLUMNS TO KEEP
# ============================================================

SHOT_COLUMNS = [
    "id",
    "match_id",
    "period",
    "minute",
    "second",
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


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "BUILDING MULTI-COMPETITION TRAINING DATASET"
    )

    print(
        "=" * 65
    )

    competitions_table = sb.competitions()

    all_shots = []

    total_shootout_shots = 0

    # ========================================================
    # LOOP THROUGH TRAINING COMPETITIONS
    # ========================================================

    for index, config in enumerate(
        TRAINING_COMPETITIONS,
        start=1
    ):

        competition_id = (
            config["competition_id"]
        )

        season_id = (
            config["season_id"]
        )

        name = (
            config["name"]
        )

        print("\n")
        print(
            f"[{index}/{len(TRAINING_COMPETITIONS)}] "
            f"{name}"
        )

        print(
            "-" * 65
        )

        # ----------------------------------------------------
        # Find exact StatsBomb metadata
        # ----------------------------------------------------

        metadata = (
            competitions_table[
                (
                    competitions_table[
                        "competition_id"
                    ]
                    == competition_id
                )
                &
                (
                    competitions_table[
                        "season_id"
                    ]
                    == season_id
                )
            ]
        )

        if metadata.empty:

            print(
                "Competition metadata not found. "
                "Skipping."
            )

            continue

        row = metadata.iloc[0]

        country = (
            row["country_name"]
        )

        division = (
            row["competition_name"]
        )

        season = (
            row["season_name"]
        )

        gender = (
            row["competition_gender"]
        )

        print(
            f"Country:     {country}"
        )

        print(
            f"Competition: {division}"
        )

        print(
            f"Season:      {season}"
        )

        print(
            f"Gender:      {gender}"
        )

        # ----------------------------------------------------
        # Match count
        # ----------------------------------------------------

        matches = sb.matches(
            competition_id=competition_id,
            season_id=season_id,
        )

        match_count = len(
            matches
        )

        print(
            f"Matches:     {match_count}"
        )

        # ----------------------------------------------------
        # Load competition events
        # ----------------------------------------------------

        print(
            "Downloading competition events..."
        )

        events = (
            sb.competition_events(
                country=country,
                division=division,
                season=season,
                gender=gender,
            )
        )

        print(
            f"Events loaded: "
            f"{len(events)}"
        )

        # ----------------------------------------------------
        # Shots only
        # ----------------------------------------------------

        shots = (
            events[
                events["type"]
                == "Shot"
            ]
            .copy()
        )

        shots_before_filter = (
            len(shots)
        )

        # ----------------------------------------------------
        # Remove penalty shootouts
        #
        # StatsBomb period 5 = penalty shootout
        # ----------------------------------------------------

        if "period" in shots.columns:

            shootout_count = (
                shots[
                    "period"
                ]
                .eq(5)
                .sum()
            )

            shots = (
                shots[
                    shots["period"]
                    != 5
                ]
                .copy()
            )

        else:

            shootout_count = 0

        total_shootout_shots += (
            int(shootout_count)
        )

        # ----------------------------------------------------
        # Keep available columns
        # ----------------------------------------------------

        available_columns = [
            column
            for column in SHOT_COLUMNS
            if column in shots.columns
        ]

        shots = (
            shots[
                available_columns
            ]
            .copy()
        )

        # ----------------------------------------------------
        # Add source metadata
        # ----------------------------------------------------

        shots[
            "source_name"
        ] = name

        shots[
            "competition_id"
        ] = competition_id

        shots[
            "season_id"
        ] = season_id

        shots[
            "country_name"
        ] = country

        shots[
            "competition_name"
        ] = division

        shots[
            "season_name"
        ] = season

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        final_shot_count = (
            len(shots)
        )

        goals = (
            shots[
                "shot_outcome"
            ]
            .eq("Goal")
            .sum()
        )

        print(
            f"Raw shots: "
            f"{shots_before_filter}"
        )

        print(
            f"Shootout shots removed: "
            f"{shootout_count}"
        )

        print(
            f"Training shots kept: "
            f"{final_shot_count}"
        )

        print(
            f"Goals: "
            f"{goals}"
        )

        if final_shot_count > 0:

            print(
                f"Goal rate: "
                f"{goals / final_shot_count:.3f}"
            )

        all_shots.append(
            shots
        )

        time.sleep(
            0.5
        )

    # ========================================================
    # COMBINE
    # ========================================================

    if not all_shots:

        raise RuntimeError(
            "No training shots were collected."
        )

    combined = pd.concat(
        all_shots,
        ignore_index=True,
        sort=False,
    )

    # ========================================================
    # DUPLICATE CHECK
    #
    # StatsBomb event ID is unique.
    # We do NOT use location because location is stored
    # as a Python list and is therefore unhashable.
    # ========================================================

    before_duplicates = (
        len(combined)
    )

    if "id" in combined.columns:

        combined = (
            combined
            .drop_duplicates(
                subset=["id"]
            )
            .copy()
        )

    else:

        # Fallback in case event ID is unavailable.
        # Convert location to string first so it can
        # safely participate in duplicate detection.

        combined[
            "_location_key"
        ] = (
            combined[
                "location"
            ]
            .astype(str)
        )

        duplicate_columns = [
            column
            for column in [
                "match_id",
                "period",
                "minute",
                "second",
                "player",
                "_location_key",
            ]
            if column in combined.columns
        ]

        combined = (
            combined
            .drop_duplicates(
                subset=duplicate_columns
            )
            .copy()
        )

        combined = (
            combined
            .drop(
                columns=[
                    "_location_key"
                ]
            )
        )

    duplicates_removed = (
        before_duplicates
        - len(combined)
    )

    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    combined.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n")
    print("=" * 65)
    print("FINAL TRAINING DATASET")
    print("=" * 65)

    print(
        f"Competitions collected: "
        f"{len(all_shots)}"
    )

    print(
        f"Matches represented: "
        f"{combined['match_id'].nunique()}"
    )

    print(
        f"Total shots: "
        f"{len(combined)}"
    )

    print(
        f"Total goals: "
        f"{combined['shot_outcome'].eq('Goal').sum()}"
    )

    print(
        f"Overall goal rate: "
        f"{combined['shot_outcome'].eq('Goal').mean():.3f}"
    )

    print(
        f"Shootout shots removed: "
        f"{total_shootout_shots}"
    )

    print(
        f"Duplicates removed: "
        f"{duplicates_removed}"
    )

    print("\nShots by source:")

    source_summary = (
        combined
        .groupby(
            "source_name"
        )
        .agg(
            matches=(
                "match_id",
                "nunique"
            ),
            shots=(
                "match_id",
                "size"
            ),
            goals=(
                "shot_outcome",
                lambda x:
                x.eq("Goal").sum()
            ),
        )
        .sort_values(
            "shots",
            ascending=False
        )
    )

    source_summary[
        "goal_rate"
    ] = (
        source_summary[
            "goals"
        ]
        /
        source_summary[
            "shots"
        ]
    )

    source_summary[
        "goal_rate"
    ] = (
        source_summary[
            "goal_rate"
        ]
        .round(3)
    )

    print(
        source_summary.to_string()
    )

    print("\nDataset saved:")

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()