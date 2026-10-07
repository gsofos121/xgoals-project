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
    "models/xg_multicompetition_logistic.json"
)

TRAIN_DATA_PATH = Path(
    "data/processed/multicompetition_training_features.csv"
)

OUTPUT_PATH = Path(
    "data/processed/final_blind_validation.csv"
)


# ============================================================
# BLIND DATASETS
# ============================================================

BLIND_DATASETS = [
    {
        "name": "UEFA Euro 2024",
        "competition_id": 55,
        "season_id": 282,
    },
    {
        "name": "Copa America 2024",
        "competition_id": 223,
        "season_id": 282,
    },
]


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
                float(value[1]),
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
            float(location[1]),
        )

    except (
        ValueError,
        SyntaxError,
        TypeError,
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
# FEATURE ENGINEERING
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
                row["y_m"],
            ),
            axis=1,
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

    # Right Foot = reference
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

    # Open Play = reference
    data["type_free_kick"] = (
        data["shot_type"]
        .eq("Free Kick")
        .astype(int)
    )

    data["goal"] = (
        data["shot_outcome"]
        .eq("Goal")
        .astype(int)
    )

    return data


# ============================================================
# PREDICT OUR MODEL
# ============================================================

def predict_our_xg(
    data,
    model
):

    penalty_xg = (
        model[
            "penalty_rule"
        ][
            "xg"
        ]
    )

    model_info = (
        model[
            "non_penalty_model"
        ]
    )

    features = (
        model_info[
            "features"
        ]
    )

    coefficients = (
        model_info[
            "coefficients"
        ]
    )

    intercept = (
        model_info[
            "intercept"
        ]
    )

    predictions = pd.Series(
        index=data.index,
        dtype=float
    )

    # --------------------------------------------------------
    # PENALTIES
    # --------------------------------------------------------

    penalty_mask = (
        data[
            "shot_type"
        ]
        .eq("Penalty")
    )

    predictions.loc[
        penalty_mask
    ] = penalty_xg

    # --------------------------------------------------------
    # NON-PENALTIES
    # --------------------------------------------------------

    non_penalty_mask = (
        ~penalty_mask
    )

    non_penalty_data = (
        data[
            non_penalty_mask
        ]
    )

    score = np.full(
        len(non_penalty_data),
        intercept,
        dtype=float
    )

    for feature in features:

        score += (
            non_penalty_data[
                feature
            ]
            .to_numpy(
                dtype=float
            )
            *
            coefficients[
                feature
            ]
        )

    predictions.loc[
        non_penalty_data.index
    ] = sigmoid(
        score
    )

    return (
        predictions
        .loc[
            data.index
        ]
        .to_numpy(
            dtype=float
        )
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
        * np.log(
            probabilities
        )
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
        ]
        .sum()
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


def get_metrics(
    y_true,
    probabilities
):

    return {
        "log_loss":
            calculate_log_loss(
                y_true,
                probabilities
            ),

        "brier":
            calculate_brier_score(
                y_true,
                probabilities
            ),

        "roc_auc":
            calculate_auc(
                y_true,
                probabilities
            ),
    }


# ============================================================
# CALIBRATION
# ============================================================

def print_calibration(
    data
):

    calibration = (
        data[
            [
                "goal",
                "our_xg",
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

    summary[
        "difference"
    ] = (
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

        summary[
            column
        ] = (
            summary[
                column
            ]
            .round(3)
        )

    print(
        summary.to_string()
    )


# ============================================================
# LOAD ONE BLIND COMPETITION
# ============================================================

def load_blind_dataset(
    config,
    competitions_table,
    model
):

    competition_id = (
        config[
            "competition_id"
        ]
    )

    season_id = (
        config[
            "season_id"
        ]
    )

    name = (
        config[
            "name"
        ]
    )

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

        raise RuntimeError(
            f"Metadata not found for {name}"
        )

    row = (
        metadata.iloc[0]
    )

    country = (
        row[
            "country_name"
        ]
    )

    division = (
        row[
            "competition_name"
        ]
    )

    season = (
        row[
            "season_name"
        ]
    )

    gender = (
        row[
            "competition_gender"
        ]
    )

    print("\n")
    print(
        "=" * 65
    )

    print(
        name
    )

    print(
        "=" * 65
    )

    print(
        f"Country: "
        f"{country}"
    )

    print(
        f"Competition: "
        f"{division}"
    )

    print(
        f"Season: "
        f"{season}"
    )

    events = (
        sb.competition_events(
            country=country,
            division=division,
            season=season,
            gender=gender,
        )
    )

    all_shots = (
        events[
            events[
                "type"
            ]
            == "Shot"
        ]
        .copy()
    )

    print(
        f"Raw shots: "
        f"{len(all_shots)}"
    )

    # --------------------------------------------------------
    # REMOVE PENALTY SHOOTOUTS
    # --------------------------------------------------------

    if (
        "period"
        in all_shots.columns
    ):

        shootout_count = (
            all_shots[
                "period"
            ]
            .eq(5)
            .sum()
        )

        shots = (
            all_shots[
                all_shots[
                    "period"
                ]
                != 5
            ]
            .copy()
        )

    else:

        shootout_count = 0

        shots = (
            all_shots.copy()
        )

    print(
        f"Shootout shots removed: "
        f"{shootout_count}"
    )

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    data = (
        build_features(
            shots
        )
    )

    required_features = (
        model[
            "non_penalty_model"
        ][
            "features"
        ]
    )

    required_columns = (
        required_features
        +
        [
            "goal",
            "shot_statsbomb_xg",
        ]
    )

    before = (
        len(data)
    )

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

    data[
        "source_name"
    ] = name

    data[
        "competition_id"
    ] = competition_id

    data[
        "season_id"
    ] = season_id

    data[
        "our_xg"
    ] = predict_our_xg(
        data,
        model
    )

    print(
        f"Usable shots: "
        f"{len(data)}"
    )

    print(
        f"Removed unusable: "
        f"{removed}"
    )

    print(
        f"Matches: "
        f"{data['match_id'].nunique()}"
    )

    print(
        f"Teams: "
        f"{data['team'].nunique()}"
    )

    print(
        f"Goals: "
        f"{data['goal'].sum()}"
    )

    print(
        f"Goal rate: "
        f"{data['goal'].mean():.4f}"
    )

    return data


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "FINAL BLIND VALIDATION"
    )

    print(
        "=" * 65
    )

    print(
        "The model is frozen."
    )

    print(
        "No retraining or tuning is performed."
    )

    print(
        "\nBlind datasets:"
    )

    print(
        "- UEFA Euro 2024"
    )

    print(
        "- Copa America 2024"
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
    # TRAINING GOAL RATE FOR NAIVE BENCHMARK
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
        f"\nTraining goal rate: "
        f"{training_goal_rate:.4f}"
    )

    print(
        f"Fixed penalty xG: "
        f"{model['penalty_rule']['xg']:.4f}"
    )

    competitions_table = (
        sb.competitions()
    )

    blind_datasets = []

    # ========================================================
    # LOAD BOTH BLIND DATASETS
    # ========================================================

    for config in (
        BLIND_DATASETS
    ):

        data = (
            load_blind_dataset(
                config,
                competitions_table,
                model
            )
        )

        blind_datasets.append(
            data
        )

    combined = pd.concat(
        blind_datasets,
        ignore_index=True,
        sort=False,
    )

    # ========================================================
    # PERFORMANCE BY COMPETITION
    # ========================================================

    print("\n")
    print("=" * 80)
    print("BLIND PERFORMANCE BY COMPETITION")
    print("=" * 80)

    print(
        f"{'Dataset':<25}"
        f"{'Shots':>9}"
        f"{'Goals':>9}"
        f"{'Log Loss':>12}"
        f"{'Brier':>12}"
        f"{'ROC-AUC':>12}"
    )

    print(
        "-" * 80
    )

    for name in (
        combined[
            "source_name"
        ]
        .unique()
    ):

        subset = (
            combined[
                combined[
                    "source_name"
                ]
                == name
            ]
        )

        y_true = (
            subset[
                "goal"
            ]
            .to_numpy(
                dtype=float
            )
        )

        predictions = (
            subset[
                "our_xg"
            ]
            .to_numpy(
                dtype=float
            )
        )

        metrics = (
            get_metrics(
                y_true,
                predictions
            )
        )

        print(
            f"{name:<25}"
            f"{len(subset):>9}"
            f"{int(y_true.sum()):>9}"
            f"{metrics['log_loss']:>12.4f}"
            f"{metrics['brier']:>12.4f}"
            f"{metrics['roc_auc']:>12.4f}"
        )

    # ========================================================
    # COMBINED BLIND PERFORMANCE
    # ========================================================

    y_all = (
        combined[
            "goal"
        ]
        .to_numpy(
            dtype=float
        )
    )

    our_predictions = (
        combined[
            "our_xg"
        ]
        .to_numpy(
            dtype=float
        )
    )

    statsbomb_predictions = (
        combined[
            "shot_statsbomb_xg"
        ]
        .to_numpy(
            dtype=float
        )
    )

    naive_predictions = np.full(
        len(combined),
        training_goal_rate
    )

    naive_metrics = (
        get_metrics(
            y_all,
            naive_predictions
        )
    )

    our_metrics = (
        get_metrics(
            y_all,
            our_predictions
        )
    )

    statsbomb_metrics = (
        get_metrics(
            y_all,
            statsbomb_predictions
        )
    )

    print("\n")
    print("=" * 70)
    print("COMBINED FINAL BLIND PERFORMANCE")
    print("=" * 70)

    print(
        f"Blind shots: "
        f"{len(combined)}"
    )

    print(
        f"Blind matches: "
        f"{combined['match_id'].nunique()}"
    )

    print(
        f"Blind goals: "
        f"{int(y_all.sum())}"
    )

    print(
        f"Blind goal rate: "
        f"{y_all.mean():.4f}"
    )

    print("\n")

    print(
        f"{'Model':<23}"
        f"{'Log Loss':>13}"
        f"{'Brier':>13}"
        f"{'ROC-AUC':>13}"
    )

    print(
        "-" * 62
    )

    print(
        f"{'Naive Goal Rate':<23}"
        f"{naive_metrics['log_loss']:>13.4f}"
        f"{naive_metrics['brier']:>13.4f}"
        f"{naive_metrics['roc_auc']:>13.4f}"
    )

    print(
        f"{'Our Final xG':<23}"
        f"{our_metrics['log_loss']:>13.4f}"
        f"{our_metrics['brier']:>13.4f}"
        f"{our_metrics['roc_auc']:>13.4f}"
    )

    print(
        f"{'StatsBomb xG':<23}"
        f"{statsbomb_metrics['log_loss']:>13.4f}"
        f"{statsbomb_metrics['brier']:>13.4f}"
        f"{statsbomb_metrics['roc_auc']:>13.4f}"
    )

    # ========================================================
    # TOTAL xG
    # ========================================================

    print("\n")
    print("=" * 70)
    print("BLIND TOTAL EXPECTED GOALS")
    print("=" * 70)

    print(
        f"Actual goals: "
        f"{int(y_all.sum())}"
    )

    print(
        f"Our total xG: "
        f"{our_predictions.sum():.1f}"
    )

    print(
        f"StatsBomb total xG: "
        f"{statsbomb_predictions.sum():.1f}"
    )

    # ========================================================
    # CALIBRATION
    # ========================================================

    print("\n")
    print("=" * 70)
    print("FINAL BLIND CALIBRATION")
    print("=" * 70)

    print_calibration(
        combined
    )

    # ========================================================
    # PENALTY CHECK
    # ========================================================

    penalties = (
        combined[
            combined[
                "shot_type"
            ]
            == "Penalty"
        ]
        .copy()
    )

    print("\n")
    print("=" * 70)
    print("BLIND PENALTY CHECK")
    print("=" * 70)

    print(
        f"Penalties: "
        f"{len(penalties)}"
    )

    if len(penalties) > 0:

        print(
            f"Penalty goals: "
            f"{penalties['goal'].sum()}"
        )

        print(
            f"Actual conversion: "
            f"{penalties['goal'].mean():.4f}"
        )

        print(
            f"Our fixed penalty xG: "
            f"{model['penalty_rule']['xg']:.4f}"
        )

        print(
            f"Mean StatsBomb penalty xG: "
            f"{penalties['shot_statsbomb_xg'].mean():.4f}"
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
        encoding="utf-8-sig"
    )

    print("\n")
    print("=" * 70)
    print("BLIND VALIDATION COMPLETE")
    print("=" * 70)

    print(
        "Results saved:"
    )

    print(
        OUTPUT_PATH
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Do not modify the model based on these results "
        "and still call these datasets blind."
    )


if __name__ == "__main__":
    main()