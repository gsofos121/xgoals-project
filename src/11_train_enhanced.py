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

BASELINE_MODEL_PATH = Path(
    "models/xg_baseline_logistic.json"
)

ENHANCED_MODEL_PATH = Path(
    "models/xg_enhanced_logistic.json"
)


# ============================================================
# SETTINGS
# ============================================================

TARGET = "goal"

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# FEATURE NAMES
# ============================================================

FEATURES = [
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
]


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
# CREATE ENHANCED FEATURES
# ============================================================

def build_features(data):

    features = pd.DataFrame(
        index=data.index
    )

    # --------------------------------------------------------
    # Geometry
    # --------------------------------------------------------

    features["distance_m"] = (
        data["distance_m"]
        .astype(float)
    )

    features["angle_deg"] = (
        data["angle_deg"]
        .astype(float)
    )

    # --------------------------------------------------------
    # Binary features
    # --------------------------------------------------------

    binary_columns = [
        "under_pressure",
        "shot_first_time",
        "shot_one_on_one",
        "shot_open_goal",
    ]

    for column in binary_columns:

        features[column] = (
            data[column]
            .fillna(0)
            .astype(float)
        )

    # --------------------------------------------------------
    # Body part
    #
    # Reference category:
    # Right Foot
    # --------------------------------------------------------

    features["body_left_foot"] = (
        data["shot_body_part"]
        .eq("Left Foot")
        .astype(float)
    )

    features["body_head"] = (
        data["shot_body_part"]
        .eq("Head")
        .astype(float)
    )

    # --------------------------------------------------------
    # Shot type
    #
    # Reference category:
    # Open Play
    # --------------------------------------------------------

    features["type_free_kick"] = (
        data["shot_type"]
        .eq("Free Kick")
        .astype(float)
    )

    features["type_penalty"] = (
        data["shot_type"]
        .eq("Penalty")
        .astype(float)
    )

    return features[FEATURES]


# ============================================================
# TRAIN LOGISTIC REGRESSION
# ============================================================

def train_logistic_regression(
    X,
    y,
    max_iter=100,
    tolerance=1e-8,
    l2=1e-5,
):

    # Add intercept
    X_design = np.column_stack(
        [
            np.ones(len(X)),
            X,
        ]
    )

    beta = np.zeros(
        X_design.shape[1]
    )

    for iteration in range(
        max_iter
    ):

        scores = (
            X_design
            @ beta
        )

        probabilities = sigmoid(
            scores
        )

        # ----------------------------------------------------
        # Gradient
        # ----------------------------------------------------

        gradient = (
            X_design.T
            @ (
                y
                - probabilities
            )
        )

        regularization = (
            l2
            * beta
        )

        # Do not regularize intercept
        regularization[0] = 0

        gradient -= (
            regularization
        )

        # ----------------------------------------------------
        # IRLS weights
        # ----------------------------------------------------

        weights = (
            probabilities
            * (
                1.0
                - probabilities
            )
        )

        weights = np.clip(
            weights,
            1e-10,
            None
        )

        weighted_X = (
            X_design
            * weights[:, np.newaxis]
        )

        information_matrix = (
            X_design.T
            @ weighted_X
        )

        # ----------------------------------------------------
        # L2 matrix
        # ----------------------------------------------------

        regularization_matrix = (
            np.eye(
                X_design.shape[1]
            )
            * l2
        )

        regularization_matrix[
            0,
            0
        ] = 0

        information_matrix += (
            regularization_matrix
        )

        # ----------------------------------------------------
        # Newton step
        # ----------------------------------------------------

        try:

            step = np.linalg.solve(
                information_matrix,
                gradient
            )

        except np.linalg.LinAlgError:

            step = (
                np.linalg.pinv(
                    information_matrix
                )
                @ gradient
            )

        beta += step

        max_change = np.max(
            np.abs(step)
        )

        if (
            max_change
            < tolerance
        ):

            print(
                f"Converged after "
                f"{iteration + 1} iterations."
            )

            break

    return beta


# ============================================================
# PREDICT
# ============================================================

def predict_proba(
    X,
    beta
):

    X_design = np.column_stack(
        [
            np.ones(len(X)),
            X,
        ]
    )

    return sigmoid(
        X_design
        @ beta
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
            1
            - y_true
        )
        * np.log(
            1
            - probabilities
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
                n_positive
                + 1
            )
            / 2
        )
    ) / (
        n_positive
        * n_negative
    )


# ============================================================
# ORIGINAL BASELINE MODEL PREDICTION
# ============================================================

def predict_baseline(
    data,
    baseline_model
):

    intercept = (
        baseline_model[
            "intercept"
        ]
    )

    score = np.full(
        len(data),
        intercept,
        dtype=float
    )

    for feature in (
        baseline_model[
            "features"
        ]
    ):

        coefficient = (
            baseline_model[
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
# MAIN
# ============================================================

def main():

    print(
        "Loading dataset...\n"
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
        f"{data[TARGET].sum()}"
    )

    print(
        f"Goal rate: "
        f"{data[TARGET].mean():.3f}"
    )

    # ========================================================
    # SAME TRAIN / TEST SPLIT AS BASELINE
    # ========================================================

    match_ids = (
        data["match_id"]
        .drop_duplicates()
        .to_numpy()
        .copy()
    )

    rng = (
        np.random.default_rng(
            RANDOM_STATE
        )
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

    test_mask = (
        data["match_id"]
        .isin(
            test_match_ids
        )
    )

    train_data = (
        data[
            ~test_mask
        ]
        .copy()
    )

    test_data = (
        data[
            test_mask
        ]
        .copy()
    )

    print(
        "\nTrain/Test split"
    )

    print(
        "----------------------------"
    )

    print(
        f"Training matches: "
        f"{train_data['match_id'].nunique()}"
    )

    print(
        f"Testing matches: "
        f"{test_data['match_id'].nunique()}"
    )

    print(
        f"Training shots: "
        f"{len(train_data)}"
    )

    print(
        f"Testing shots: "
        f"{len(test_data)}"
    )

    # ========================================================
    # BUILD FEATURES
    # ========================================================

    X_train_df = (
        build_features(
            train_data
        )
    )

    X_test_df = (
        build_features(
            test_data
        )
    )

    X_train_raw = (
        X_train_df
        .to_numpy(
            dtype=float
        )
    )

    X_test_raw = (
        X_test_df
        .to_numpy(
            dtype=float
        )
    )

    y_train = (
        train_data[
            TARGET
        ]
        .to_numpy(
            dtype=float
        )
    )

    y_test = (
        test_data[
            TARGET
        ]
        .to_numpy(
            dtype=float
        )
    )

    # ========================================================
    # STANDARDIZATION
    # ========================================================

    feature_mean = (
        X_train_raw
        .mean(
            axis=0
        )
    )

    feature_std = (
        X_train_raw
        .std(
            axis=0
        )
    )

    feature_std[
        feature_std == 0
    ] = 1.0

    X_train = (
        X_train_raw
        - feature_mean
    ) / feature_std

    X_test = (
        X_test_raw
        - feature_mean
    ) / feature_std

    # ========================================================
    # TRAIN
    # ========================================================

    print(
        "\nTraining enhanced "
        "Logistic Regression...\n"
    )

    beta = (
        train_logistic_regression(
            X_train,
            y_train
        )
    )

    enhanced_xg = (
        predict_proba(
            X_test,
            beta
        )
    )

    # ========================================================
    # ENHANCED PERFORMANCE
    # ========================================================

    enhanced_logloss = (
        calculate_log_loss(
            y_test,
            enhanced_xg
        )
    )

    enhanced_brier = (
        calculate_brier_score(
            y_test,
            enhanced_xg
        )
    )

    enhanced_auc = (
        calculate_auc(
            y_test,
            enhanced_xg
        )
    )

    # ========================================================
    # BASELINE MODEL
    # ========================================================

    with open(
        BASELINE_MODEL_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        baseline_model = (
            json.load(
                file
            )
        )

    baseline_xg = (
        predict_baseline(
            test_data,
            baseline_model
        )
    )

    baseline_logloss = (
        calculate_log_loss(
            y_test,
            baseline_xg
        )
    )

    baseline_brier = (
        calculate_brier_score(
            y_test,
            baseline_xg
        )
    )

    baseline_auc = (
        calculate_auc(
            y_test,
            baseline_xg
        )
    )

    # ========================================================
    # STATSBOMB MODEL
    # ========================================================

    statsbomb_xg = (
        test_data[
            "shot_statsbomb_xg"
        ]
        .to_numpy(
            dtype=float
        )
    )

    statsbomb_logloss = (
        calculate_log_loss(
            y_test,
            statsbomb_xg
        )
    )

    statsbomb_brier = (
        calculate_brier_score(
            y_test,
            statsbomb_xg
        )
    )

    statsbomb_auc = (
        calculate_auc(
            y_test,
            statsbomb_xg
        )
    )

    # ========================================================
    # PERFORMANCE TABLE
    # ========================================================

    print(
        "\nMODEL COMPARISON"
    )

    print(
        "-" * 61
    )

    print(
        f"{'Model':<20}"
        f"{'Log Loss':>13}"
        f"{'Brier':>13}"
        f"{'ROC-AUC':>13}"
    )

    print(
        "-" * 61
    )

    print(
        f"{'Baseline xG':<20}"
        f"{baseline_logloss:>13.4f}"
        f"{baseline_brier:>13.4f}"
        f"{baseline_auc:>13.4f}"
    )

    print(
        f"{'Enhanced xG':<20}"
        f"{enhanced_logloss:>13.4f}"
        f"{enhanced_brier:>13.4f}"
        f"{enhanced_auc:>13.4f}"
    )

    print(
        f"{'StatsBomb xG':<20}"
        f"{statsbomb_logloss:>13.4f}"
        f"{statsbomb_brier:>13.4f}"
        f"{statsbomb_auc:>13.4f}"
    )

    print(
        "-" * 61
    )

    # ========================================================
    # ORIGINAL-SCALE COEFFICIENTS
    # ========================================================

    standardized_coefficients = (
        beta[1:]
    )

    original_coefficients = (
        standardized_coefficients
        / feature_std
    )

    original_intercept = (
        beta[0]
        -
        np.sum(
            standardized_coefficients
            * feature_mean
            / feature_std
        )
    )

    print(
        "\nENHANCED MODEL COEFFICIENTS"
    )

    print(
        "-" * 45
    )

    for feature, coefficient in zip(
        FEATURES,
        original_coefficients
    ):

        print(
            f"{feature:<22}: "
            f"{coefficient:>10.5f}"
        )

    print(
        f"{'Intercept':<22}: "
        f"{original_intercept:>10.5f}"
    )

    # ========================================================
    # EXAMPLE PREDICTIONS
    # ========================================================

    examples = (
        test_data[
            [
                "player",
                "distance_m",
                "angle_deg",
                "shot_body_part",
                "shot_type",
                "shot_one_on_one",
                "shot_open_goal",
                "shot_outcome",
            ]
        ]
        .copy()
    )

    examples[
        "baseline_xg"
    ] = baseline_xg

    examples[
        "enhanced_xg"
    ] = enhanced_xg

    examples[
        "statsbomb_xg"
    ] = statsbomb_xg

    print(
        "\nEXAMPLE PREDICTIONS"
    )

    print(
        "-" * 100
    )

    print(
        examples[
            [
                "player",
                "distance_m",
                "angle_deg",
                "shot_body_part",
                "shot_type",
                "baseline_xg",
                "enhanced_xg",
                "statsbomb_xg",
                "shot_outcome",
            ]
        ]
        .head(20)
        .round(
            {
                "distance_m": 2,
                "angle_deg": 2,
                "baseline_xg": 3,
                "enhanced_xg": 3,
                "statsbomb_xg": 3,
            }
        )
        .to_string(
            index=False
        )
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    ENHANCED_MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    model_data = {

        "model_type":
            "enhanced_logistic_regression",

        "features":
            FEATURES,

        "intercept":
            float(
                original_intercept
            ),

        "coefficients": {

            feature: float(
                coefficient
            )

            for feature, coefficient
            in zip(
                FEATURES,
                original_coefficients
            )
        },

        "categorical_rules": {

            "shot_body_part": {
                "reference":
                    "Right Foot",

                "Left Foot":
                    "body_left_foot",

                "Head":
                    "body_head",
            },

            "shot_type": {
                "reference":
                    "Open Play",

                "Free Kick":
                    "type_free_kick",

                "Penalty":
                    "type_penalty",
            },
        },

        "performance": {

            "log_loss":
                float(
                    enhanced_logloss
                ),

            "brier_score":
                float(
                    enhanced_brier
                ),

            "roc_auc":
                float(
                    enhanced_auc
                ),
        },
    }

    with open(
        ENHANCED_MODEL_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            model_data,
            file,
            indent=4
        )

    print(
        "\nEnhanced model saved:"
    )

    print(
        ENHANCED_MODEL_PATH
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()