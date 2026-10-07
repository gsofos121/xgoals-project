from pathlib import Path

import pandas as pd


DATA_PATH = Path(
    "data/processed/premier_league_2015_2016_features.csv"
)


def print_categorical_counts(
    data,
    column
):
    """
    Print value counts and goal rates
    for a categorical variable.
    """

    print("\n")
    print("=" * 70)
    print(column.upper())
    print("=" * 70)

    if column not in data.columns:

        print(
            "Column not found."
        )

        return

    summary = (
        data
        .groupby(
            column,
            dropna=False
        )
        .agg(
            shots=(
                "goal",
                "size"
            ),
            goals=(
                "goal",
                "sum"
            ),
            goal_rate=(
                "goal",
                "mean"
            ),
        )
        .reset_index()
    )

    summary = (
        summary
        .sort_values(
            "shots",
            ascending=False
        )
    )

    summary[
        "goal_rate"
    ] = (
        summary[
            "goal_rate"
        ]
        .round(3)
    )

    print(
        summary.to_string(
            index=False
        )
    )


def print_binary_summary(
    data,
    column
):
    """
    Print counts and goal rates
    for binary 0/1 variables.
    """

    print("\n")
    print("=" * 70)
    print(column.upper())
    print("=" * 70)

    if column not in data.columns:

        print(
            "Column not found."
        )

        return

    summary = (
        data
        .groupby(
            column,
            dropna=False
        )
        .agg(
            shots=(
                "goal",
                "size"
            ),
            goals=(
                "goal",
                "sum"
            ),
            goal_rate=(
                "goal",
                "mean"
            ),
        )
        .reset_index()
    )

    summary[
        "goal_rate"
    ] = (
        summary[
            "goal_rate"
        ]
        .round(3)
    )

    print(
        summary.to_string(
            index=False
        )
    )


def main():

    print(
        "Loading processed season dataset...\n"
    )

    data = pd.read_csv(
        DATA_PATH
    )

    print(
        f"Total shots: "
        f"{len(data)}"
    )

    print(
        f"Total goals: "
        f"{data['goal'].sum()}"
    )

    print(
        f"Overall goal rate: "
        f"{data['goal'].mean():.3f}"
    )

    # ========================================================
    # CATEGORICAL FEATURES
    # ========================================================

    categorical_columns = [
        "shot_body_part",
        "shot_technique",
        "shot_type",
        "play_pattern",
    ]

    for column in categorical_columns:

        print_categorical_counts(
            data,
            column
        )

    # ========================================================
    # BINARY FEATURES
    # ========================================================

    binary_columns = [
        "under_pressure",
        "shot_first_time",
        "shot_one_on_one",
        "shot_open_goal",
    ]

    for column in binary_columns:

        print_binary_summary(
            data,
            column
        )

    # ========================================================
    # DISTANCE GROUPS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("DISTANCE GROUPS")
    print("=" * 70)

    distance_bins = [
        0,
        5,
        10,
        15,
        20,
        25,
        30,
        40,
        60,
        100,
    ]

    data[
        "distance_group"
    ] = pd.cut(
        data[
            "distance_m"
        ],
        bins=distance_bins,
        right=False
    )

    distance_summary = (
        data
        .groupby(
            "distance_group",
            observed=True
        )
        .agg(
            shots=(
                "goal",
                "size"
            ),
            goals=(
                "goal",
                "sum"
            ),
            goal_rate=(
                "goal",
                "mean"
            ),
        )
        .reset_index()
    )

    distance_summary[
        "goal_rate"
    ] = (
        distance_summary[
            "goal_rate"
        ]
        .round(3)
    )

    print(
        distance_summary.to_string(
            index=False
        )
    )

    # ========================================================
    # ANGLE GROUPS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("ANGLE GROUPS")
    print("=" * 70)

    angle_bins = [
        0,
        10,
        20,
        30,
        40,
        50,
        60,
        90,
        120,
        180,
    ]

    data[
        "angle_group"
    ] = pd.cut(
        data[
            "angle_deg"
        ],
        bins=angle_bins,
        right=False
    )

    angle_summary = (
        data
        .groupby(
            "angle_group",
            observed=True
        )
        .agg(
            shots=(
                "goal",
                "size"
            ),
            goals=(
                "goal",
                "sum"
            ),
            goal_rate=(
                "goal",
                "mean"
            ),
        )
        .reset_index()
    )

    angle_summary[
        "goal_rate"
    ] = (
        angle_summary[
            "goal_rate"
        ]
        .round(3)
    )

    print(
        angle_summary.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()