from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

DATA_PATH = Path(
    "data/processed/premier_league_2015_2016_features.csv"
)

MODEL_PATH = Path(
    "models/xg_baseline_logistic.json"
)


# ============================================================
# SETTINGS
# ============================================================

TARGET = "goal"

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# SIGMOID
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
        (1 - y_true)
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
        pd.Series(probabilities)
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


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict_our_xg(
    data,
    model
):

    intercept = (
        model["intercept"]
    )

    coefficients = (
        model["coefficients"]
    )

    score = np.full(
        len(data),
        intercept,
        dtype=float
    )

    for feature in model["features"]:

        score += (
            data[feature]
            .to_numpy(dtype=float)
            *
            coefficients[feature]
        )

    return sigmoid(
        score
    )


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def create_test_set(
    data
):

    match_ids = (
        data["match_id"]
        .drop_duplicates()
        .to_numpy()
        .copy()
    )

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    rng.shuffle(
        match_ids
    )

    number_test_matches = int(
        len(match_ids)
        * TEST_SIZE
    )

    number_test_matches = max(
        1,
        number_test_matches
    )

    test_match_ids = set(
        match_ids[
            :number_test_matches
        ]
    )

    test_data = (
        data[
            data["match_id"]
            .isin(test_match_ids)
        ]
        .copy()
    )

    train_data = (
        data[
            ~data["match_id"]
            .isin(test_match_ids)
        ]
        .copy()
    )

    return (
        train_data,
        test_data
    )


# ============================================================
# PRINT METRICS
# ============================================================

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
        f"{name:<20}"
        f"{logloss:>12.4f}"
        f"{brier:>12.4f}"
        f"{auc:>12.4f}"
    )


# ============================================================
# CALIBRATION TABLE
# ============================================================

def print_calibration_table(
    test_data
):

    calibration = (
        test_data[
            [
                "our_xg",
                "goal"
            ]
        ]
        .copy()
    )

    calibration["bin"] = pd.qcut(
        calibration["our_xg"],
        q=10,
        duplicates="drop"
    )

    table = (
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

    table["difference"] = (
        table["actual_goal_rate"]
        -
        table["predicted_xg"]
    )

    table["predicted_xg"] = (
        table["predicted_xg"]
        .round(3)
    )

    table["actual_goal_rate"] = (
        table["actual_goal_rate"]
        .round(3)
    )

    table["difference"] = (
        table["difference"]
        .round(3)
    )

    print(
        table.to_string(
            index=True
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "Loading dataset and model...\n"
    )

    data = pd.read_csv(
        DATA_PATH
    )

    with open(
        MODEL_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        model = json.load(
            file
        )

    train_data, test_data = (
        create_test_set(
            data
        )
    )

    print(
        f"Training shots: "
        f"{len(train_data)}"
    )

    print(
        f"Testing shots:  "
        f"{len(test_data)}"
    )

    print(
        f"Testing goals:  "
        f"{test_data[TARGET].sum()}"
    )

    print(
        f"Testing goal rate: "
        f"{test_data[TARGET].mean():.3f}"
    )

    # ========================================================
    # OUR MODEL
    # ========================================================

    test_data[
        "our_xg"
    ] = predict_our_xg(
        test_data,
        model
    )

    y_test = (
        test_data[TARGET]
        .to_numpy(
            dtype=float
        )
    )

    # ========================================================
    # NAIVE BASELINE
    # ========================================================

    training_goal_rate = (
        train_data[TARGET]
        .mean()
    )

    naive_xg = np.full(
        len(test_data),
        training_goal_rate
    )

    # ========================================================
    # PERFORMANCE COMPARISON
    # ========================================================

    print(
        "\nPERFORMANCE COMPARISON"
    )

    print(
        "-" * 56
    )

    print(
        f"{'Model':<20}"
        f"{'Log Loss':>12}"
        f"{'Brier':>12}"
        f"{'ROC-AUC':>12}"
    )

    print(
        "-" * 56
    )

    print_metrics(
        "Naive Goal Rate",
        y_test,
        naive_xg
    )

    print_metrics(
        "Our xG",
        y_test,
        test_data[
            "our_xg"
        ].to_numpy()
    )

    # ========================================================
    # STATSBOMB COMPARISON
    # ========================================================

    if (
        "shot_statsbomb_xg"
        in test_data.columns
    ):

        statsbomb_mask = (
            test_data[
                "shot_statsbomb_xg"
            ]
            .notna()
        )

        if statsbomb_mask.any():

            statsbomb_data = (
                test_data[
                    statsbomb_mask
                ]
            )

            statsbomb_y = (
                statsbomb_data[
                    TARGET
                ]
                .to_numpy(
                    dtype=float
                )
            )

            statsbomb_xg = (
                statsbomb_data[
                    "shot_statsbomb_xg"
                ]
                .to_numpy(
                    dtype=float
                )
            )

            print_metrics(
                "StatsBomb xG",
                statsbomb_y,
                statsbomb_xg
            )

    print(
        "-" * 56
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
        "10% of the test shots.\n"
    )

    print_calibration_table(
        test_data
    )

    # ========================================================
    # xG RANGE
    # ========================================================

    print(
        "\nOUR xG RANGE"
    )

    print(
        "-" * 30
    )

    print(
        f"Minimum xG: "
        f"{test_data['our_xg'].min():.4f}"
    )

    print(
        f"Median xG:  "
        f"{test_data['our_xg'].median():.4f}"
    )

    print(
        f"Mean xG:    "
        f"{test_data['our_xg'].mean():.4f}"
    )

    print(
        f"Maximum xG: "
        f"{test_data['our_xg'].max():.4f}"
    )

    # ========================================================
    # HIGHEST xG SHOTS
    # ========================================================

    print(
        "\nHIGHEST xG SHOTS"
    )

    print(
        "-" * 70
    )

    columns = [
        "player",
        "distance_m",
        "angle_deg",
        "our_xg",
        "shot_statsbomb_xg",
        "shot_outcome",
    ]

    available_columns = [
        column
        for column in columns
        if column in test_data.columns
    ]

    highest = (
        test_data
        .sort_values(
            "our_xg",
            ascending=False
        )
        [available_columns]
        .head(15)
        .copy()
    )

    numeric_columns = [
        "distance_m",
        "angle_deg",
        "our_xg",
        "shot_statsbomb_xg",
    ]

    for column in numeric_columns:

        if column in highest.columns:

            highest[column] = (
                highest[column]
                .round(3)
            )

    print(
        highest.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()