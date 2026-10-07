from pathlib import Path
import ast
import json
import math

import numpy as np
import pandas as pd
from statsbombpy import sb


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = Path(
    "models/xg_enhanced_logistic.json"
)

TRAIN_DATA_PATH = Path(
    "data/processed/premier_league_2015_2016_features.csv"
)

OUTPUT_PATH = Path(
    "data/processed/world_cup_2022_external_validation.csv"
)


# ============================================================
# WORLD CUP DATASET
# ============================================================

COUNTRY = "International"
DIVISION = "FIFA World Cup"
SEASON = "2022"
GENDER = "male"


# ============================================================
# PITCH CONSTANTS
# ============================================================

STATSBOMB_LENGTH = 120.0
STATSBOMB_WIDTH = 80.0

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0

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
# BASIC FUNCTIONS
# ============================================================

def sigmoid(z):

    z = np.clip(
        z,
        -500,
        500
    )

    return (
        1.0
        /
        (
            1.0
            + np.exp(-z)
        )
    )


def parse_location(value):

    if value is None:
        return np.nan, np.nan

    if isinstance(
        value,
        (list, tuple, np.ndarray)
    ):

        if len(value) >= 2:

            return (
                float(value[0]),
                float(value[1])
            )

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
            float(location[1])
        )

    except (
        ValueError,
        SyntaxError,
        TypeError
    ):

        return np.nan, np.nan


def calculate_shot_angle(
    x_m,
    y_m
):

    if (
        pd.isna(x_m)
        or pd.isna(y_m)
    ):
        return np.nan

    v1_x = (
        GOAL_X_M
        - x_m
    )

    v1_y = (
        LEFT_POST_Y_M
        - y_m
    )

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

    return math.degrees(
        math.acos(
            cosine_angle
        )
    )


# ============================================================
# BOOLEAN CONVERSION
# ============================================================

def convert_boolean_column(
    series
):

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
# BUILD FEATURES
# ============================================================

def build_features(
    shots
):

    data = shots.copy()

    parsed_locations = (
        data["location"]
        .apply(
            parse_location
        )
    )

    data["x_sb"] = (
        parsed_locations
        .apply(
            lambda value:
            value[0]
        )
    )

    data["y_sb"] = (
        parsed_locations
        .apply(
            lambda value:
            value[1]
        )
    )

    data["x_m"] = (
        data["x_sb"]
        / STATSBOMB_LENGTH
        * PITCH_LENGTH_M
    )

    data["y_m"] = (
        data["y_sb"]
        / STATSBOMB_WIDTH
        * PITCH_WIDTH_M
    )

    data["distance_m"] = np.sqrt(
        (
            GOAL_X_M
            - data["x_m"]
        ) ** 2
        +
        (
            GOAL_CENTER_Y_M
            - data["y_m"]
        ) ** 2
    )

    data["angle_deg"] = (
        data.apply(
            lambda row:
            calculate_shot_angle(
                row["x_m"],
                row["y_m"]
            ),
            axis=1
        )
    )

    binary_columns = [
        "under_pressure",
        "shot_first_time",
        "shot_one_on_one",
        "shot_open_goal",
    ]

    for column in binary_columns:

        if column not in data.columns:

            data[column] = 0

        else:

            data[column] = (
                convert_boolean_column(
                    data[column]
                )
            )

    if (
        "shot_body_part"
        not in data.columns
    ):
        data["shot_body_part"] = "Unknown"

    if (
        "shot_type"
        not in data.columns
    ):
        data["shot_type"] = "Unknown"

    data["body_left_foot"] = (
        data["shot_body_part"]
        .eq("Left Foot")
        .astype(int)
    )

    data["body_head"] = (
        data["shot_body_part"]
        .eq("Head")
        .astype(int)
    )

    data["type_free_kick"] = (
        data["shot_type"]
        .eq("Free Kick")
        .astype(int)
    )

    data["type_penalty"] = (
        data["shot_type"]
        .eq("Penalty")
        .astype(int)
    )

    data["goal"] = (
        data["shot_outcome"]
        .eq("Goal")
        .astype(int)
    )

    return data


# ============================================================
# OUR MODEL
# ============================================================

def predict_our_xg(
    data,
    model
):

    score = np.full(
        len(data),
        model["intercept"],
        dtype=float
    )

    for feature in (
        model["features"]
    ):

        coefficient = (
            model[
                "coefficients"
            ][feature]
        )

        score += (
            data[feature]
            .to_numpy(
                dtype=float
            )
            * coefficient
        )

    return sigmoid(
        score
    )


# ============================================================
# METRICS
# ============================================================

def calculate_log_loss(
    y_true,
    probabilities
):

    probabilities = np.clip(
        probabilities,
        1e-15,
        1 - 1e-15
    )

    return -np.mean(
        y_true
        * np.log(probabilities)
        +
        (
            1 - y_true
        )
        * np.log(
            1 - probabilities
        )
    )


def calculate_brier_score(
    y_true,
    probabilities
):

    return np.mean(
        (
            probabilities
            - y_true
        ) ** 2
    )


def calculate_auc(
    y_true,
    probabilities
):

    ranking = (
        pd.Series(
            probabilities
        )
        .rank(
            method="average"
        )
        .to_numpy()
    )

    positive = (
        y_true == 1
    )

    negative = (
        y_true == 0
    )

    n_positive = (
        positive.sum()
    )

    n_negative = (
        negative.sum()
    )

    if (
        n_positive == 0
        or n_negative == 0
    ):
        return np.nan

    positive_rank_sum = (
        ranking[
            positive
        ].sum()
    )

    return (
        positive_rank_sum
        -
        (
            n_positive
            * (
                n_positive + 1
            )
            / 2
        )
    ) / (
        n_positive
        * n_negative
    )


def print_metrics(
    name,
    y_true,
    probabilities
):

    logloss = (
        calculate_log_loss(
            y_true,
            probabilities
        )
    )

    brier = (
        calculate_brier_score(
            y_true,
            probabilities
        )
    )

    auc = (
        calculate_auc(
            y_true,
            probabilities
        )
    )

    print(
        f"{name:<22}"
        f"{logloss:>13.4f}"
        f"{brier:>13.4f}"
        f"{auc:>13.4f}"
    )


# ============================================================
# CALIBRATION
# ============================================================

def print_calibration(
    data
):

    calibration = (
        data[
            [
                "our_xg",
                "goal"
            ]
        ]
        .copy()
    )

    calibration["bin"] = (
        pd.qcut(
            calibration[
                "our_xg"
            ],
            q=10,
            duplicates="drop"
        )
    )

    summary = (
        calibration
        .groupby(
            "bin",
            observed=True
        )
        .agg(
            shots=(
                "goal",
                "size"
            ),
            predicted_xg=(
                "our_xg",
                "mean"
            ),
            actual_goal_rate=(
                "goal",
                "mean"
            ),
        )
        .reset_index(
            drop=True
        )
    )

    summary["difference"] = (
        summary[
            "actual_goal_rate"
        ]
        -
        summary[
            "predicted_xg"
        ]
    )

    for column in [
        "predicted_xg",
        "actual_goal_rate",
        "difference",
    ]:

        summary[column] = (
            summary[column]
            .round(3)
        )

    print(
        summary.to_string()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "WORLD CUP 2022 EXTERNAL VALIDATION"
    )

    print(
        "=" * 55
    )

    print(
        "Model training data:"
    )

    print(
        "Premier League 2015/2016"
    )

    print(
        "\nExternal test data:"
    )

    print(
        "FIFA World Cup 2022"
    )

    print(
        "\nPenalty shootouts are excluded."
    )

    print(
        "NO RETRAINING IS PERFORMED."
    )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    with open(
        MODEL_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        model = json.load(
            file
        )

    # ========================================================
    # ORIGINAL TRAINING GOAL RATE
    # ========================================================

    training_data = pd.read_csv(
        TRAIN_DATA_PATH
    )

    training_goal_rate = (
        training_data[
            "goal"
        ]
        .mean()
    )

    print(
        f"\nPremier League training "
        f"goal rate: "
        f"{training_goal_rate:.3f}"
    )

    # ========================================================
    # LOAD WORLD CUP
    # ========================================================

    print(
        "\nLoading World Cup 2022 events..."
    )

    events = (
        sb.competition_events(
            country=COUNTRY,
            division=DIVISION,
            season=SEASON,
            gender=GENDER,
        )
    )

    print(
        f"Total events: "
        f"{len(events)}"
    )

    # ========================================================
    # ALL SHOTS
    # ========================================================

    all_shots = (
        events[
            events["type"]
            == "Shot"
        ]
        .copy()
    )

    print(
        f"All shots including shootouts: "
        f"{len(all_shots)}"
    )

    # ========================================================
    # EXCLUDE PENALTY SHOOTOUTS
    #
    # StatsBomb:
    # period == 5 -> penalty shootout
    # ========================================================

    shootout_shots = (
        all_shots[
            all_shots["period"]
            == 5
        ]
        .copy()
    )

    shots = (
        all_shots[
            all_shots["period"]
            != 5
        ]
        .copy()
    )

    print(
        f"Shootout shots excluded: "
        f"{len(shootout_shots)}"
    )

    print(
        f"Shots used for validation: "
        f"{len(shots)}"
    )

    print(
        f"Unique matches: "
        f"{shots['match_id'].nunique()}"
    )

    print(
        f"Unique teams: "
        f"{shots['team'].nunique()}"
    )

    # ========================================================
    # FEATURES
    # ========================================================

    data = (
        build_features(
            shots
        )
    )

    required_columns = (
        model["features"]
        +
        [
            "goal",
            "shot_statsbomb_xg",
        ]
    )

    before = len(data)

    data = (
        data
        .dropna(
            subset=required_columns
        )
        .copy()
    )

    removed = (
        before
        - len(data)
    )

    print(
        f"Usable shots: "
        f"{len(data)}"
    )

    print(
        f"Removed unusable shots: "
        f"{removed}"
    )

    print(
        f"Goals: "
        f"{data['goal'].sum()}"
    )

    print(
        f"Goal rate: "
        f"{data['goal'].mean():.3f}"
    )

    # ========================================================
    # PREDICTIONS
    # ========================================================

    data["our_xg"] = (
        predict_our_xg(
            data,
            model
        )
    )

    y_true = (
        data["goal"]
        .to_numpy(
            dtype=float
        )
    )

    naive_xg = np.full(
        len(data),
        training_goal_rate
    )

    statsbomb_xg = (
        data[
            "shot_statsbomb_xg"
        ]
        .to_numpy(
            dtype=float
        )
    )

    # ========================================================
    # PERFORMANCE
    # ========================================================

    print(
        "\nWORLD CUP EXTERNAL VALIDATION PERFORMANCE"
    )

    print(
        "-" * 64
    )

    print(
        f"{'Model':<22}"
        f"{'Log Loss':>13}"
        f"{'Brier':>13}"
        f"{'ROC-AUC':>13}"
    )

    print(
        "-" * 64
    )

    print_metrics(
        "Naive Goal Rate",
        y_true,
        naive_xg
    )

    print_metrics(
        "Our Enhanced xG",
        y_true,
        data[
            "our_xg"
        ].to_numpy()
    )

    print_metrics(
        "StatsBomb xG",
        y_true,
        statsbomb_xg
    )

    print(
        "-" * 64
    )

    # ========================================================
    # CALIBRATION
    # ========================================================

    print(
        "\nOUR MODEL CALIBRATION"
    )

    print(
        "-" * 55
    )

    print(
        "Each row contains approximately "
        "10% of World Cup shots.\n"
    )

    print_calibration(
        data
    )

    # ========================================================
    # TOTAL EXPECTED GOALS
    # ========================================================

    print(
        "\nTOTAL EXPECTED GOALS"
    )

    print(
        "-" * 45
    )

    print(
        f"Actual goals:       "
        f"{data['goal'].sum()}"
    )

    print(
        f"Our total xG:       "
        f"{data['our_xg'].sum():.1f}"
    )

    print(
        f"StatsBomb total xG: "
        f"{data['shot_statsbomb_xg'].sum():.1f}"
    )

    # ========================================================
    # NORMAL PENALTIES ONLY
    # ========================================================

    penalties = (
        data[
            data["shot_type"]
            == "Penalty"
        ]
        .copy()
    )

    print(
        "\nNORMAL MATCH PENALTY CHECK"
    )

    print(
        "-" * 45
    )

    print(
        f"Normal penalties: "
        f"{len(penalties)}"
    )

    if len(penalties) > 0:

        print(
            f"Penalty goals: "
            f"{penalties['goal'].sum()}"
        )

        print(
            f"Actual conversion: "
            f"{penalties['goal'].mean():.3f}"
        )

        print(
            f"Mean our penalty xG: "
            f"{penalties['our_xg'].mean():.3f}"
        )

        print(
            f"Mean StatsBomb penalty xG: "
            f"{penalties['shot_statsbomb_xg'].mean():.3f}"
        )

    # ========================================================
    # xG RANGE
    # ========================================================

    print(
        "\nOUR xG RANGE"
    )

    print(
        "-" * 45
    )

    print(
        f"Minimum: "
        f"{data['our_xg'].min():.4f}"
    )

    print(
        f"Median:  "
        f"{data['our_xg'].median():.4f}"
    )

    print(
        f"Mean:    "
        f"{data['our_xg'].mean():.4f}"
    )

    print(
        f"Maximum: "
        f"{data['our_xg'].max():.4f}"
    )

    # ========================================================
    # SAVE CLEAN VALIDATION DATASET
    # ========================================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    data.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        "\nClean World Cup external "
        "validation dataset saved:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()