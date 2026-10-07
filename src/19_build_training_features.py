from pathlib import Path
import ast
import math

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

RAW_PATH = Path(
    "data/raw/multicompetition_training_shots.csv"
)

OUTPUT_PATH = Path(
    "data/processed/multicompetition_training_features.csv"
)


# ============================================================
# PITCH CONSTANTS
# ============================================================

# StatsBomb coordinate system
STATSBOMB_LENGTH = 120.0
STATSBOMB_WIDTH = 80.0

# Standard pitch dimensions in metres
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0

# Goal dimensions
GOAL_WIDTH_M = 7.32

GOAL_X_M = PITCH_LENGTH_M
GOAL_CENTER_Y_M = PITCH_WIDTH_M / 2

LEFT_POST_Y_M = (
    GOAL_CENTER_Y_M
    - GOAL_WIDTH_M / 2
)

RIGHT_POST_Y_M = (
    GOAL_CENTER_Y_M
    + GOAL_WIDTH_M / 2
)


# ============================================================
# LOCATION PARSER
# ============================================================

def parse_location(value):
    """
    Convert StatsBomb location into x, y.

    Example:
    [115.8, 38.2]
    or
    "[115.8, 38.2]"
    """

    if value is None:
        return np.nan, np.nan

    # Already a list / tuple / array
    if isinstance(
        value,
        (list, tuple, np.ndarray)
    ):

        if len(value) >= 2:

            return (
                float(value[0]),
                float(value[1]),
            )

        return np.nan, np.nan

    if pd.isna(value):
        return np.nan, np.nan

    try:

        location = ast.literal_eval(
            str(value)
        )

        if len(location) < 2:
            return np.nan, np.nan

        return (
            float(location[0]),
            float(location[1]),
        )

    except (
        ValueError,
        SyntaxError,
        TypeError,
    ):

        return np.nan, np.nan


# ============================================================
# SHOT ANGLE
# ============================================================

def calculate_shot_angle(
    x_m,
    y_m
):
    """
    Calculate the angle between the shot position
    and the two goalposts.

    Returned in degrees.
    """

    if (
        pd.isna(x_m)
        or pd.isna(y_m)
    ):
        return np.nan

    # Vector from shot to left post
    v1_x = (
        GOAL_X_M
        - x_m
    )

    v1_y = (
        LEFT_POST_Y_M
        - y_m
    )

    # Vector from shot to right post
    v2_x = (
        GOAL_X_M
        - x_m
    )

    v2_y = (
        RIGHT_POST_Y_M
        - y_m
    )

    dot_product = (
        v1_x * v2_x
        +
        v1_y * v2_y
    )

    magnitude_1 = math.sqrt(
        v1_x ** 2
        +
        v1_y ** 2
    )

    magnitude_2 = math.sqrt(
        v2_x ** 2
        +
        v2_y ** 2
    )

    if (
        magnitude_1 == 0
        or magnitude_2 == 0
    ):
        return 180.0

    cosine_angle = (
        dot_product
        /
        (
            magnitude_1
            * magnitude_2
        )
    )

    cosine_angle = np.clip(
        cosine_angle,
        -1.0,
        1.0
    )

    angle_radians = math.acos(
        cosine_angle
    )

    return math.degrees(
        angle_radians
    )


# ============================================================
# BOOLEAN CONVERTER
# ============================================================

def convert_boolean_column(
    series
):
    """
    StatsBomb boolean fields are usually True or NaN.

    True -> 1
    NaN / False -> 0
    """

    return (
        series
        .fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("true")
        .astype(int)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "BUILDING MULTI-COMPETITION FEATURES"
    )

    print(
        "=" * 65
    )

    # ========================================================
    # LOAD RAW SHOTS
    # ========================================================

    shots = pd.read_csv(
        RAW_PATH
    )

    print(
        f"Raw shots loaded: "
        f"{len(shots)}"
    )

    print(
        f"Matches: "
        f"{shots['match_id'].nunique()}"
    )

    print(
        f"Sources: "
        f"{shots['source_name'].nunique()}"
    )

    # ========================================================
    # VERIFY SHOOTOUTS ARE GONE
    # ========================================================

    if "period" in shots.columns:

        period_5_count = (
            shots[
                "period"
            ]
            .eq(5)
            .sum()
        )

        print(
            f"Period 5 shots found: "
            f"{period_5_count}"
        )

    else:

        period_5_count = 0

    # ========================================================
    # PARSE LOCATION
    # ========================================================

    print(
        "\nParsing shot locations..."
    )

    parsed_locations = (
        shots[
            "location"
        ]
        .apply(
            parse_location
        )
    )

    shots[
        "x_sb"
    ] = (
        parsed_locations
        .apply(
            lambda value:
            value[0]
        )
    )

    shots[
        "y_sb"
    ] = (
        parsed_locations
        .apply(
            lambda value:
            value[1]
        )
    )

    # ========================================================
    # COORDINATE VALIDATION
    # ========================================================

    valid_coordinate_mask = (
        shots[
            "x_sb"
        ]
        .between(
            0,
            STATSBOMB_LENGTH
        )
        &
        shots[
            "y_sb"
        ]
        .between(
            0,
            STATSBOMB_WIDTH
        )
    )

    invalid_coordinates = (
        (~valid_coordinate_mask)
        .sum()
    )

    print(
        f"Invalid/missing coordinates: "
        f"{invalid_coordinates}"
    )

    # Remove unusable location rows
    shots = (
        shots[
            valid_coordinate_mask
        ]
        .copy()
    )

    # ========================================================
    # CONVERT STATSBOMB COORDINATES TO METRES
    # ========================================================

    shots[
        "x_m"
    ] = (
        shots[
            "x_sb"
        ]
        /
        STATSBOMB_LENGTH
        *
        PITCH_LENGTH_M
    )

    shots[
        "y_m"
    ] = (
        shots[
            "y_sb"
        ]
        /
        STATSBOMB_WIDTH
        *
        PITCH_WIDTH_M
    )

    # ========================================================
    # DISTANCE FROM GOAL CENTRE
    # ========================================================

    shots[
        "distance_m"
    ] = np.sqrt(
        (
            GOAL_X_M
            -
            shots[
                "x_m"
            ]
        ) ** 2
        +
        (
            GOAL_CENTER_Y_M
            -
            shots[
                "y_m"
            ]
        ) ** 2
    )

    # ========================================================
    # SHOT ANGLE
    # ========================================================

    print(
        "Calculating shot angles..."
    )

    shots[
        "angle_deg"
    ] = shots.apply(
        lambda row:
        calculate_shot_angle(
            row[
                "x_m"
            ],
            row[
                "y_m"
            ],
        ),
        axis=1,
    )

    # ========================================================
    # BOOLEAN FEATURES
    # ========================================================

    binary_columns = [
        "under_pressure",
        "shot_first_time",
        "shot_one_on_one",
        "shot_open_goal",
    ]

    for column in binary_columns:

        if column not in shots.columns:

            shots[
                column
            ] = 0

        else:

            shots[
                column
            ] = (
                convert_boolean_column(
                    shots[
                        column
                    ]
                )
            )

    # ========================================================
    # CLEAN CATEGORICAL FEATURES
    # ========================================================

    categorical_columns = [
        "play_pattern",
        "shot_body_part",
        "shot_technique",
        "shot_type",
    ]

    for column in categorical_columns:

        if column not in shots.columns:

            shots[
                column
            ] = "Unknown"

        else:

            shots[
                column
            ] = (
                shots[
                    column
                ]
                .fillna(
                    "Unknown"
                )
                .astype(str)
            )

    # ========================================================
    # TARGET
    # ========================================================

    shots[
        "goal"
    ] = (
        shots[
            "shot_outcome"
        ]
        .eq("Goal")
        .astype(int)
    )

    # ========================================================
    # MODEL DUMMY FEATURES
    #
    # These match the enhanced model structure we already
    # validated.
    # ========================================================

    # Right Foot = reference
    shots[
        "body_left_foot"
    ] = (
        shots[
            "shot_body_part"
        ]
        .eq(
            "Left Foot"
        )
        .astype(int)
    )

    shots[
        "body_head"
    ] = (
        shots[
            "shot_body_part"
        ]
        .eq(
            "Head"
        )
        .astype(int)
    )

    # Open Play = reference
    shots[
        "type_free_kick"
    ] = (
        shots[
            "shot_type"
        ]
        .eq(
            "Free Kick"
        )
        .astype(int)
    )

    shots[
        "type_penalty"
    ] = (
        shots[
            "shot_type"
        ]
        .eq(
            "Penalty"
        )
        .astype(int)
    )

    # ========================================================
    # FINAL QUALITY CHECK
    # ========================================================

    required_model_columns = [
        "distance_m",
        "angle_deg",
        "under_pressure",
        "shot_first_time",
        "shot_one_on_one",
        "shot_open_goal",
        "body_left_foot",
        "body_head",
        "type_free_kick",
        "type_penalty",
        "goal",
    ]

    missing_model_values = (
        shots[
            required_model_columns
        ]
        .isna()
        .sum()
    )

    print(
        "\nMissing model values:"
    )

    print(
        missing_model_values
        .to_string()
    )

    # ========================================================
    # FINAL COLUMNS
    # ========================================================

    final_columns = [
        # Identifiers
        "id",
        "match_id",
        "period",
        "minute",
        "second",

        # Source metadata
        "source_name",
        "competition_id",
        "season_id",
        "country_name",
        "competition_name",
        "season_name",

        # Human-readable information
        "team",
        "player",

        # Raw geometry
        "x_sb",
        "y_sb",

        # Geometry in metres
        "x_m",
        "y_m",
        "distance_m",
        "angle_deg",

        # Context
        "under_pressure",
        "play_pattern",
        "shot_body_part",
        "shot_technique",
        "shot_type",
        "shot_first_time",
        "shot_one_on_one",
        "shot_open_goal",

        # Model dummy variables
        "body_left_foot",
        "body_head",
        "type_free_kick",
        "type_penalty",

        # Outcome
        "shot_outcome",
        "goal",

        # StatsBomb benchmark only
        "shot_statsbomb_xg",
    ]

    final_columns = [
        column
        for column in final_columns
        if column in shots.columns
    ]

    dataset = (
        shots[
            final_columns
        ]
        .copy()
    )

    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    dataset.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n")
    print("=" * 65)
    print("FEATURE DATASET SUMMARY")
    print("=" * 65)

    print(
        f"Final rows: "
        f"{len(dataset)}"
    )

    print(
        f"Matches: "
        f"{dataset['match_id'].nunique()}"
    )

    print(
        f"Goals: "
        f"{dataset['goal'].sum()}"
    )

    print(
        f"Goal rate: "
        f"{dataset['goal'].mean():.3f}"
    )

    print(
        f"Missing distance: "
        f"{dataset['distance_m'].isna().sum()}"
    )

    print(
        f"Missing angle: "
        f"{dataset['angle_deg'].isna().sum()}"
    )

    print(
        f"Minimum distance: "
        f"{dataset['distance_m'].min():.2f} m"
    )

    print(
        f"Maximum distance: "
        f"{dataset['distance_m'].max():.2f} m"
    )

    print(
        f"Minimum angle: "
        f"{dataset['angle_deg'].min():.2f}°"
    )

    print(
        f"Maximum angle: "
        f"{dataset['angle_deg'].max():.2f}°"
    )

    # ========================================================
    # SOURCE SUMMARY
    # ========================================================

    print("\nBy source:")

    source_summary = (
        dataset
        .groupby(
            "source_name"
        )
        .agg(
            matches=(
                "match_id",
                "nunique"
            ),
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
            mean_distance=(
                "distance_m",
                "mean"
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
            "goal_rate"
        ]
        .round(3)
    )

    source_summary[
        "mean_distance"
    ] = (
        source_summary[
            "mean_distance"
        ]
        .round(2)
    )

    print(
        source_summary.to_string()
    )

    # ========================================================
    # FEATURE FREQUENCIES
    # ========================================================

    print("\n")
    print("=" * 65)
    print("IMPORTANT FEATURE COUNTS")
    print("=" * 65)

    feature_counts = {
        "Under pressure":
            dataset[
                "under_pressure"
            ].sum(),

        "First time":
            dataset[
                "shot_first_time"
            ].sum(),

        "One-on-one":
            dataset[
                "shot_one_on_one"
            ].sum(),

        "Open goal":
            dataset[
                "shot_open_goal"
            ].sum(),

        "Left foot":
            dataset[
                "body_left_foot"
            ].sum(),

        "Head":
            dataset[
                "body_head"
            ].sum(),

        "Free kick":
            dataset[
                "type_free_kick"
            ].sum(),

        "Penalty":
            dataset[
                "type_penalty"
            ].sum(),
    }

    for name, count in (
        feature_counts.items()
    ):

        print(
            f"{name:<20}: "
            f"{int(count)}"
        )

    print("\nDataset saved:")

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()