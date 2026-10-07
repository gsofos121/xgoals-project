from pathlib import Path
import ast
import math

import numpy as np
import pandas as pd


RAW_PATH = Path(
    "data/raw/premier_league_2015_2016_shots.csv"
)

OUTPUT_PATH = Path(
    "data/processed/premier_league_2015_2016_features.csv"
)


# StatsBomb coordinate system
STATSBOMB_LENGTH = 120.0
STATSBOMB_WIDTH = 80.0

# Standard football pitch dimensions in metres
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0

# Goal dimensions
GOAL_WIDTH_M = 7.32

GOAL_X_M = PITCH_LENGTH_M
GOAL_CENTER_Y_M = PITCH_WIDTH_M / 2

LEFT_POST_Y_M = GOAL_CENTER_Y_M - GOAL_WIDTH_M / 2
RIGHT_POST_Y_M = GOAL_CENTER_Y_M + GOAL_WIDTH_M / 2


def parse_location(value):
    """
    Convert StatsBomb location such as:
    '[115.8, 38.2]'

    into:
    x, y
    """

    if pd.isna(value):
        return np.nan, np.nan

    try:
        location = ast.literal_eval(str(value))

        if len(location) < 2:
            return np.nan, np.nan

        return float(location[0]), float(location[1])

    except (ValueError, SyntaxError, TypeError):
        return np.nan, np.nan


def calculate_shot_angle(x_m, y_m):
    """
    Calculate the angle between the shot position
    and the two goalposts.
    """

    if pd.isna(x_m) or pd.isna(y_m):
        return np.nan

    # Vector from shot to left post
    v1_x = GOAL_X_M - x_m
    v1_y = LEFT_POST_Y_M - y_m

    # Vector from shot to right post
    v2_x = GOAL_X_M - x_m
    v2_y = RIGHT_POST_Y_M - y_m

    dot_product = (v1_x * v2_x) + (v1_y * v2_y)

    magnitude_1 = math.sqrt(
        (v1_x ** 2) + (v1_y ** 2)
    )

    magnitude_2 = math.sqrt(
        (v2_x ** 2) + (v2_y ** 2)
    )

    if magnitude_1 == 0 or magnitude_2 == 0:
        return 180.0

    cosine_angle = dot_product / (
        magnitude_1 * magnitude_2
    )

    cosine_angle = np.clip(
        cosine_angle,
        -1.0,
        1.0
    )

    angle_radians = math.acos(cosine_angle)

    return math.degrees(angle_radians)


def main():

    print("Loading raw shots...\n")

    shots = pd.read_csv(RAW_PATH)

    print(f"Raw shots: {len(shots)}")

    # -------------------------------------------------
    # 1. Extract StatsBomb X and Y
    # -------------------------------------------------

    parsed_locations = shots["location"].apply(
        parse_location
    )

    shots["x_sb"] = parsed_locations.apply(
        lambda value: value[0]
    )

    shots["y_sb"] = parsed_locations.apply(
        lambda value: value[1]
    )

    # -------------------------------------------------
    # 2. Convert StatsBomb coordinates to metres
    # -------------------------------------------------

    shots["x_m"] = (
        shots["x_sb"]
        / STATSBOMB_LENGTH
        * PITCH_LENGTH_M
    )

    shots["y_m"] = (
        shots["y_sb"]
        / STATSBOMB_WIDTH
        * PITCH_WIDTH_M
    )

    # -------------------------------------------------
    # 3. Distance from centre of goal
    # -------------------------------------------------

    shots["distance_m"] = np.sqrt(
        (GOAL_X_M - shots["x_m"]) ** 2
        +
        (GOAL_CENTER_Y_M - shots["y_m"]) ** 2
    )

    # -------------------------------------------------
    # 4. Shooting angle
    # -------------------------------------------------

    shots["angle_deg"] = shots.apply(
        lambda row: calculate_shot_angle(
            row["x_m"],
            row["y_m"]
        ),
        axis=1
    )

    # -------------------------------------------------
    # 5. Convert boolean features to 0 / 1
    # -------------------------------------------------

    boolean_columns = [
        "under_pressure",
        "shot_first_time",
        "shot_one_on_one",
        "shot_open_goal",
    ]

    for column in boolean_columns:

        if column in shots.columns:

            shots[column] = (
                shots[column]
                .astype(str)
                .str.strip()
                .str.lower()
                .eq("true")
                .astype(int)
            )

    # -------------------------------------------------
    # 6. Clean categorical variables
    # -------------------------------------------------

    categorical_columns = [
        "play_pattern",
        "shot_body_part",
        "shot_technique",
        "shot_type",
    ]

    for column in categorical_columns:

        if column in shots.columns:

            shots[column] = (
                shots[column]
                .fillna("Unknown")
            )

    # -------------------------------------------------
    # 7. Create target variable
    # -------------------------------------------------

    shots["goal"] = (
        shots["shot_outcome"] == "Goal"
    ).astype(int)

    # -------------------------------------------------
    # 8. Final columns
    # -------------------------------------------------

    final_columns = [
        "match_id",
        "minute",
        "team",
        "player",

        "x_sb",
        "y_sb",

        "x_m",
        "y_m",

        "distance_m",
        "angle_deg",

        "under_pressure",
        "play_pattern",
        "shot_body_part",
        "shot_technique",
        "shot_type",
        "shot_first_time",
        "shot_one_on_one",
        "shot_open_goal",

        "shot_outcome",
        "goal",

        # Keep only for later comparison.
        # It will NOT be used as model input.
        "shot_statsbomb_xg",
    ]

    final_columns = [
        column
        for column in final_columns
        if column in shots.columns
    ]

    dataset = shots[final_columns].copy()

    # -------------------------------------------------
    # 9. Save processed dataset
    # -------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    dataset.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    # -------------------------------------------------
    # 10. Checks
    # -------------------------------------------------

    print("\nFeature engineering complete.")

    print("\n--------------------------------")
    print(f"Total rows: {len(dataset)}")
    print(f"Goals: {dataset['goal'].sum()}")
    print(
        f"Missing distance: "
        f"{dataset['distance_m'].isna().sum()}"
    )
    print(
        f"Missing angle: "
        f"{dataset['angle_deg'].isna().sum()}"
    )
    print("--------------------------------")

    print("\nExample shots:\n")

    example_columns = [
        "player",
        "x_sb",
        "y_sb",
        "distance_m",
        "angle_deg",
        "shot_body_part",
        "goal",
    ]

    print(
        dataset[example_columns]
        .head(10)
        .round(
            {
                "x_sb": 2,
                "y_sb": 2,
                "distance_m": 2,
                "angle_deg": 2,
            }
        )
        .to_string(index=False)
    )

    print("\nDataset saved:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()