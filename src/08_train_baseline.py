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
# MODEL SETTINGS
# ============================================================

FEATURES = [
    "distance_m",
    "angle_deg",
]

TARGET = "goal"

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# SIGMOID
# ============================================================

def sigmoid(z):
    """
    Logistic sigmoid function.

    Converts any real number into a probability
    between 0 and 1.
    """

    z = np.clip(
        z,
        -500,
        500
    )

    return 1.0 / (
        1.0 + np.exp(-z)
    )


# ============================================================
# TRAIN LOGISTIC REGRESSION
# ============================================================

def train_logistic_regression(
    X,
    y,
    max_iter=100,
    tolerance=1e-8,
    l2=1e-6,
):
    """
    Train Logistic Regression using
    Newton-Raphson / IRLS.

    Returns:
        beta:
            coefficients including intercept
    """

    # --------------------------------------------------------
    # Add intercept column
    # --------------------------------------------------------

    X_design = np.column_stack(
        [
            np.ones(len(X)),
            X,
        ]
    )

    # Start coefficients at zero
    beta = np.zeros(
        X_design.shape[1]
    )

    # --------------------------------------------------------
    # Optimization
    # --------------------------------------------------------

    for iteration in range(max_iter):

        scores = (
            X_design @ beta
        )

        probabilities = sigmoid(
            scores
        )

        # ----------------------------------------------------
        # Gradient
        # ----------------------------------------------------

        gradient = (
            X_design.T
            @ (y - probabilities)
        )

        # Small L2 regularization
        regularization = (
            l2 * beta
        )

        # Do not regularize intercept
        regularization[0] = 0

        gradient -= regularization

        # ----------------------------------------------------
        # Observation weights
        # ----------------------------------------------------

        weights = (
            probabilities
            * (1.0 - probabilities)
        )

        # Prevent numerical problems
        weights = np.clip(
            weights,
            1e-10,
            None
        )

        # ----------------------------------------------------
        # Information matrix
        # ----------------------------------------------------

        weighted_X = (
            X_design
            * weights[:, np.newaxis]
        )

        information_matrix = (
            X_design.T
            @ weighted_X
        )

        # L2 regularization matrix
        regularization_matrix = (
            np.eye(
                X_design.shape[1]
            )
            * l2
        )

        # Do not regularize intercept
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

            # Fallback if matrix cannot be inverted normally
            step = (
                np.linalg.pinv(
                    information_matrix
                )
                @ gradient
            )

        beta += step

        # ----------------------------------------------------
        # Convergence check
        # ----------------------------------------------------

        max_change = np.max(
            np.abs(step)
        )

        if max_change < tolerance:

            print(
                f"Converged after "
                f"{iteration + 1} iterations."
            )

            break

    return beta


# ============================================================
# PREDICT xG
# ============================================================

def predict_proba(
    X,
    beta
):
    """
    Calculate goal probability.

    The returned probability is the xG value.
    """

    X_design = np.column_stack(
        [
            np.ones(len(X)),
            X,
        ]
    )

    scores = (
        X_design @ beta
    )

    probabilities = sigmoid(
        scores
    )

    return probabilities


# ============================================================
# LOG LOSS
# ============================================================

def calculate_log_loss(
    y_true,
    probabilities
):
    """
    Binary Log Loss.
    """

    probabilities = np.clip(
        probabilities,
        1e-15,
        1 - 1e-15
    )

    loss = -np.mean(
        y_true
        * np.log(probabilities)
        +
        (1 - y_true)
        * np.log(
            1 - probabilities
        )
    )

    return loss


# ============================================================
# BRIER SCORE
# ============================================================

def calculate_brier_score(
    y_true,
    probabilities
):
    """
    Mean squared difference between
    predicted probability and actual outcome.
    """

    return np.mean(
        (
            probabilities
            - y_true
        ) ** 2
    )


# ============================================================
# ROC-AUC
# ============================================================

def calculate_auc(
    y_true,
    probabilities
):
    """
    ROC-AUC calculated using rank statistics.
    """

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

    auc = (
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

    return auc


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

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
        f"Overall goal rate: "
        f"{data[TARGET].mean():.3f}"
    )

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    required_columns = (
        FEATURES
        + [
            TARGET,
            "match_id",
        ]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(
                missing_columns
            )
        )

    # --------------------------------------------------------
    # Remove rows with missing model values
    # --------------------------------------------------------

    before_drop = len(data)

    data = (
        data
        .dropna(
            subset=required_columns
        )
        .copy()
    )

    removed_rows = (
        before_drop
        - len(data)
    )

    if removed_rows > 0:

        print(
            f"Removed rows with missing values: "
            f"{removed_rows}"
        )

    # ========================================================
    # TRAIN / TEST SPLIT BY MATCH
    # ========================================================

    # IMPORTANT:
    # .copy() makes the NumPy array writable.
    # Without it, rng.shuffle() may produce:
    # ValueError: array is read-only

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

    # Ensure at least one match is used for testing
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
        f"Testing matches:  "
        f"{test_data['match_id'].nunique()}"
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
        f"Training goals: "
        f"{train_data[TARGET].sum()}"
    )

    print(
        f"Testing goals:  "
        f"{test_data[TARGET].sum()}"
    )

    # ========================================================
    # FEATURES
    # ========================================================

    X_train_raw = (
        train_data[
            FEATURES
        ]
        .to_numpy(
            dtype=float
        )
    )

    X_test_raw = (
        test_data[
            FEATURES
        ]
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
        X_train_raw.mean(
            axis=0
        )
    )

    feature_std = (
        X_train_raw.std(
            axis=0
        )
    )

    # Avoid division by zero
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
    # TRAIN MODEL
    # ========================================================

    print(
        "\nTraining Logistic Regression...\n"
    )

    beta = train_logistic_regression(
        X_train,
        y_train
    )

    # ========================================================
    # PREDICTIONS
    # ========================================================

    predicted_xg = (
        predict_proba(
            X_test,
            beta
        )
    )

    # ========================================================
    # EVALUATION
    # ========================================================

    logloss = (
        calculate_log_loss(
            y_test,
            predicted_xg
        )
    )

    brier = (
        calculate_brier_score(
            y_test,
            predicted_xg
        )
    )

    auc = (
        calculate_auc(
            y_test,
            predicted_xg
        )
    )

    print(
        "\nMODEL PERFORMANCE"
    )

    print(
        "----------------------------"
    )

    print(
        f"Log Loss:    "
        f"{logloss:.4f}"
    )

    print(
        f"Brier Score: "
        f"{brier:.4f}"
    )

    print(
        f"ROC-AUC:     "
        f"{auc:.4f}"
    )

    # ========================================================
    # CONVERT COEFFICIENTS BACK TO ORIGINAL UNITS
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
        "\nMODEL COEFFICIENTS"
    )

    print(
        "----------------------------"
    )

    for feature, coefficient in zip(
        FEATURES,
        original_coefficients
    ):

        print(
            f"{feature:<15}: "
            f"{coefficient:.6f}"
        )

    print(
        f"{'Intercept':<15}: "
        f"{original_intercept:.6f}"
    )

    # ========================================================
    # EXAMPLE PREDICTIONS
    # ========================================================

    examples = (
        test_data[
            FEATURES
        ]
        .copy()
    )

    examples[
        "predicted_xg"
    ] = predicted_xg

    examples[
        "actual_goal"
    ] = y_test.astype(
        int
    )

    print(
        "\nEXAMPLE PREDICTIONS"
    )

    print(
        "----------------------------"
    )

    print(
        examples[
            [
                "distance_m",
                "angle_deg",
                "predicted_xg",
                "actual_goal",
            ]
        ]
        .head(15)
        .round(
            {
                "distance_m": 2,
                "angle_deg": 2,
                "predicted_xg": 3,
            }
        )
        .to_string(
            index=False
        )
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    model_data = {

        "model_type":
            "logistic_regression",

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

        "training_info": {

            "total_shots":
                int(
                    len(data)
                ),

            "training_shots":
                int(
                    len(train_data)
                ),

            "testing_shots":
                int(
                    len(test_data)
                ),

            "training_matches":
                int(
                    train_data[
                        "match_id"
                    ].nunique()
                ),

            "testing_matches":
                int(
                    test_data[
                        "match_id"
                    ].nunique()
                ),

            "random_state":
                RANDOM_STATE,

            "test_size":
                TEST_SIZE,
        },

        "performance": {

            "log_loss":
                float(
                    logloss
                ),

            "brier_score":
                float(
                    brier
                ),

            "roc_auc":
                float(
                    auc
                ),
        },
    }

    with open(
        MODEL_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            model_data,
            file,
            indent=4
        )

    print(
        "\nModel saved:"
    )

    print(
        MODEL_PATH
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()