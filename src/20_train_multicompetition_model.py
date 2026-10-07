from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

DATA_PATH = Path(
    "data/processed/multicompetition_training_features.csv"
)

MODEL_PATH = Path(
    "models/xg_multicompetition_logistic.json"
)


# ============================================================
# SETTINGS
# ============================================================

TARGET = "goal"

N_FOLDS = 5
RANDOM_STATE = 42


# ============================================================
# NON-PENALTY MODEL FEATURES
#
# Penalties are handled separately.
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
# TRAIN LOGISTIC REGRESSION
# ============================================================

def train_logistic_regression(
    X,
    y,
    max_iter=100,
    tolerance=1e-8,
    l2=1e-5,
):

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
        # GRADIENT
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
        # IRLS WEIGHTS
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
        # REGULARIZATION MATRIX
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
        # NEWTON STEP
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

            return (
                beta,
                iteration + 1
            )

    return (
        beta,
        max_iter
    )


# ============================================================
# PREDICT
# ============================================================

def predict_logistic(
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
# BUILD NUMPY FEATURES
# ============================================================

def get_feature_matrix(
    data
):

    return (
        data[
            FEATURES
        ]
        .to_numpy(
            dtype=float
        )
    )


# ============================================================
# STANDARDIZE
# ============================================================

def standardize_train_test(
    X_train_raw,
    X_test_raw
):

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

    return (
        X_train,
        X_test,
        feature_mean,
        feature_std,
    )


# ============================================================
# CALIBRATION TABLE
# ============================================================

def print_calibration_table(
    y_true,
    predictions
):

    calibration = pd.DataFrame(
        {
            "goal": y_true,
            "predicted_xg": predictions,
        }
    )

    calibration[
        "bin"
    ] = pd.qcut(
        calibration[
            "predicted_xg"
        ],
        q=10,
        duplicates="drop"
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
                "predicted_xg",
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
# MAIN
# ============================================================

def main():

    print(
        "MULTI-COMPETITION xG MODEL"
    )

    print(
        "=" * 65
    )

    # ========================================================
    # LOAD DATA
    # ========================================================

    data = pd.read_csv(
        DATA_PATH
    )

    print(
        f"Total shots: "
        f"{len(data)}"
    )

    print(
        f"Matches: "
        f"{data['match_id'].nunique()}"
    )

    print(
        f"Goals: "
        f"{data[TARGET].sum()}"
    )

    print(
        f"Overall goal rate: "
        f"{data[TARGET].mean():.4f}"
    )

    # ========================================================
    # PENALTY INFORMATION
    # ========================================================

    penalties = (
        data[
            data[
                "shot_type"
            ]
            == "Penalty"
        ]
        .copy()
    )

    non_penalties = (
        data[
            data[
                "shot_type"
            ]
            != "Penalty"
        ]
        .copy()
    )

    print(
        f"\nPenalties: "
        f"{len(penalties)}"
    )

    print(
        f"Penalty goals: "
        f"{penalties[TARGET].sum()}"
    )

    print(
        f"Overall penalty conversion: "
        f"{penalties[TARGET].mean():.4f}"
    )

    print(
        f"\nNon-penalty shots: "
        f"{len(non_penalties)}"
    )

    # ========================================================
    # CREATE GROUPED FOLDS BY MATCH
    # ========================================================

    match_ids = (
        data[
            "match_id"
        ]
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

    fold_match_ids = (
        np.array_split(
            match_ids,
            N_FOLDS
        )
    )

    # Store out-of-fold predictions
    oof_predictions = pd.Series(
        index=data.index,
        dtype=float
    )

    fold_results = []

    # ========================================================
    # CROSS VALIDATION
    # ========================================================

    print("\n")
    print("=" * 65)
    print("5-FOLD MATCH-LEVEL CROSS VALIDATION")
    print("=" * 65)

    for fold_number in range(
        N_FOLDS
    ):

        print(
            f"\nFold "
            f"{fold_number + 1}/{N_FOLDS}"
        )

        print(
            "-" * 45
        )

        validation_matches = set(
            fold_match_ids[
                fold_number
            ]
        )

        validation_mask = (
            data[
                "match_id"
            ]
            .isin(
                validation_matches
            )
        )

        train_data = (
            data[
                ~validation_mask
            ]
            .copy()
        )

        validation_data = (
            data[
                validation_mask
            ]
            .copy()
        )

        # ----------------------------------------------------
        # PENALTIES
        #
        # IMPORTANT:
        # Penalty probability is calculated only from
        # the training fold to avoid data leakage.
        # ----------------------------------------------------

        train_penalties = (
            train_data[
                train_data[
                    "shot_type"
                ]
                == "Penalty"
            ]
        )

        penalty_xg = (
            train_penalties[
                TARGET
            ]
            .mean()
        )

        # ----------------------------------------------------
        # NON-PENALTY TRAINING DATA
        # ----------------------------------------------------

        train_non_penalties = (
            train_data[
                train_data[
                    "shot_type"
                ]
                != "Penalty"
            ]
            .copy()
        )

        validation_non_penalty_mask = (
            validation_data[
                "shot_type"
            ]
            != "Penalty"
        )

        validation_penalty_mask = (
            validation_data[
                "shot_type"
            ]
            == "Penalty"
        )

        # ----------------------------------------------------
        # TRAIN FEATURE MATRICES
        # ----------------------------------------------------

        X_train_raw = (
            get_feature_matrix(
                train_non_penalties
            )
        )

        y_train = (
            train_non_penalties[
                TARGET
            ]
            .to_numpy(
                dtype=float
            )
        )

        validation_non_penalties = (
            validation_data[
                validation_non_penalty_mask
            ]
        )

        X_validation_raw = (
            get_feature_matrix(
                validation_non_penalties
            )
        )

        # ----------------------------------------------------
        # STANDARDIZATION
        # ----------------------------------------------------

        (
            X_train,
            X_validation,
            feature_mean,
            feature_std,
        ) = standardize_train_test(
            X_train_raw,
            X_validation_raw
        )

        # ----------------------------------------------------
        # TRAIN MODEL
        # ----------------------------------------------------

        beta, iterations = (
            train_logistic_regression(
                X_train,
                y_train
            )
        )

        # ----------------------------------------------------
        # PREDICT NON-PENALTIES
        # ----------------------------------------------------

        non_penalty_predictions = (
            predict_logistic(
                X_validation,
                beta
            )
        )

        # ----------------------------------------------------
        # BUILD COMPLETE FOLD PREDICTIONS
        # ----------------------------------------------------

        fold_predictions = pd.Series(
            index=validation_data.index,
            dtype=float
        )

        fold_predictions.loc[
            validation_non_penalties.index
        ] = (
            non_penalty_predictions
        )

        # Penalties get fixed training-fold xG
        fold_predictions.loc[
            validation_data[
                validation_penalty_mask
            ].index
        ] = (
            penalty_xg
        )

        # Save out-of-fold predictions
        oof_predictions.loc[
            validation_data.index
        ] = (
            fold_predictions.loc[
                validation_data.index
            ]
        )

        # ----------------------------------------------------
        # FOLD METRICS
        # ----------------------------------------------------

        y_validation = (
            validation_data[
                TARGET
            ]
            .to_numpy(
                dtype=float
            )
        )

        predictions = (
            fold_predictions.loc[
                validation_data.index
            ]
            .to_numpy(
                dtype=float
            )
        )

        logloss = (
            calculate_log_loss(
                y_validation,
                predictions
            )
        )

        brier = (
            calculate_brier_score(
                y_validation,
                predictions
            )
        )

        auc = (
            calculate_auc(
                y_validation,
                predictions
            )
        )

        fold_results.append(
            {
                "fold":
                    fold_number + 1,

                "matches":
                    validation_data[
                        "match_id"
                    ].nunique(),

                "shots":
                    len(
                        validation_data
                    ),

                "goals":
                    int(
                        validation_data[
                            TARGET
                        ].sum()
                    ),

                "penalty_xg":
                    float(
                        penalty_xg
                    ),

                "log_loss":
                    float(
                        logloss
                    ),

                "brier":
                    float(
                        brier
                    ),

                "roc_auc":
                    float(
                        auc
                    ),

                "iterations":
                    int(
                        iterations
                    ),
            }
        )

        print(
            f"Validation matches: "
            f"{validation_data['match_id'].nunique()}"
        )

        print(
            f"Validation shots: "
            f"{len(validation_data)}"
        )

        print(
            f"Penalty xG learned: "
            f"{penalty_xg:.4f}"
        )

        print(
            f"Converged after: "
            f"{iterations} iterations"
        )

        print(
            f"Log Loss: "
            f"{logloss:.4f}"
        )

        print(
            f"Brier: "
            f"{brier:.4f}"
        )

        print(
            f"ROC-AUC: "
            f"{auc:.4f}"
        )

    # ========================================================
    # OVERALL OOF PERFORMANCE
    # ========================================================

    if (
        oof_predictions
        .isna()
        .any()
    ):

        raise RuntimeError(
            "Some out-of-fold predictions "
            "are missing."
        )

    y_all = (
        data[
            TARGET
        ]
        .to_numpy(
            dtype=float
        )
    )

    predictions_all = (
        oof_predictions
        .loc[
            data.index
        ]
        .to_numpy(
            dtype=float
        )
    )

    overall_logloss = (
        calculate_log_loss(
            y_all,
            predictions_all
        )
    )

    overall_brier = (
        calculate_brier_score(
            y_all,
            predictions_all
        )
    )

    overall_auc = (
        calculate_auc(
            y_all,
            predictions_all
        )
    )

    # ========================================================
    # STATSBOMB OOF BENCHMARK
    # ========================================================

    statsbomb_predictions = (
        data[
            "shot_statsbomb_xg"
        ]
        .to_numpy(
            dtype=float
        )
    )

    statsbomb_logloss = (
        calculate_log_loss(
            y_all,
            statsbomb_predictions
        )
    )

    statsbomb_brier = (
        calculate_brier_score(
            y_all,
            statsbomb_predictions
        )
    )

    statsbomb_auc = (
        calculate_auc(
            y_all,
            statsbomb_predictions
        )
    )

    # ========================================================
    # NAIVE MODEL
    # ========================================================

    naive_probability = (
        data[
            TARGET
        ]
        .mean()
    )

    naive_predictions = np.full(
        len(data),
        naive_probability
    )

    naive_logloss = (
        calculate_log_loss(
            y_all,
            naive_predictions
        )
    )

    naive_brier = (
        calculate_brier_score(
            y_all,
            naive_predictions
        )
    )

    naive_auc = (
        calculate_auc(
            y_all,
            naive_predictions
        )
    )

    # ========================================================
    # PRINT FOLD TABLE
    # ========================================================

    fold_table = pd.DataFrame(
        fold_results
    )

    print("\n")
    print("=" * 80)
    print("CROSS-VALIDATION FOLD RESULTS")
    print("=" * 80)

    print(
        fold_table
        .round(
            {
                "penalty_xg": 4,
                "log_loss": 4,
                "brier": 4,
                "roc_auc": 4,
            }
        )
        .to_string(
            index=False
        )
    )

    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    print("\n")
    print("=" * 65)
    print("OVERALL OUT-OF-FOLD PERFORMANCE")
    print("=" * 65)

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
        f"{naive_logloss:>13.4f}"
        f"{naive_brier:>13.4f}"
        f"{naive_auc:>13.4f}"
    )

    print(
        f"{'Our Multi-Comp xG':<23}"
        f"{overall_logloss:>13.4f}"
        f"{overall_brier:>13.4f}"
        f"{overall_auc:>13.4f}"
    )

    print(
        f"{'StatsBomb xG':<23}"
        f"{statsbomb_logloss:>13.4f}"
        f"{statsbomb_brier:>13.4f}"
        f"{statsbomb_auc:>13.4f}"
    )

    # ========================================================
    # CALIBRATION
    # ========================================================

    print("\n")
    print("=" * 65)
    print("OUT-OF-FOLD CALIBRATION")
    print("=" * 65)

    print_calibration_table(
        y_all,
        predictions_all
    )

    # ========================================================
    # TOTAL OOF xG
    # ========================================================

    print("\n")
    print("=" * 65)
    print("OUT-OF-FOLD TOTALS")
    print("=" * 65)

    print(
        f"Actual goals: "
        f"{int(y_all.sum())}"
    )

    print(
        f"Our total OOF xG: "
        f"{predictions_all.sum():.1f}"
    )

    print(
        f"StatsBomb total xG: "
        f"{statsbomb_predictions.sum():.1f}"
    )

    # ========================================================
    # FINAL MODEL TRAINING
    #
    # Train using ALL NON-PENALTY shots.
    # ========================================================

    print("\n")
    print("=" * 65)
    print("TRAINING FINAL MODEL ON ALL TRAINING DATA")
    print("=" * 65)

    final_non_penalties = (
        data[
            data[
                "shot_type"
            ]
            != "Penalty"
        ]
        .copy()
    )

    final_penalties = (
        data[
            data[
                "shot_type"
            ]
            == "Penalty"
        ]
        .copy()
    )

    final_penalty_xg = (
        final_penalties[
            TARGET
        ]
        .mean()
    )

    X_final_raw = (
        get_feature_matrix(
            final_non_penalties
        )
    )

    y_final = (
        final_non_penalties[
            TARGET
        ]
        .to_numpy(
            dtype=float
        )
    )

    feature_mean = (
        X_final_raw.mean(
            axis=0
        )
    )

    feature_std = (
        X_final_raw.std(
            axis=0
        )
    )

    feature_std[
        feature_std == 0
    ] = 1.0

    X_final = (
        X_final_raw
        - feature_mean
    ) / feature_std

    final_beta, final_iterations = (
        train_logistic_regression(
            X_final,
            y_final
        )
    )

    # ========================================================
    # CONVERT COEFFICIENTS BACK TO ORIGINAL SCALE
    # ========================================================

    standardized_coefficients = (
        final_beta[
            1:
        ]
    )

    original_coefficients = (
        standardized_coefficients
        / feature_std
    )

    original_intercept = (
        final_beta[0]
        -
        np.sum(
            standardized_coefficients
            * feature_mean
            / feature_std
        )
    )

    print(
        f"\nFinal model converged after "
        f"{final_iterations} iterations."
    )

    print(
        f"Final learned penalty xG: "
        f"{final_penalty_xg:.4f}"
    )

    print("\nFINAL COEFFICIENTS")
    print("-" * 45)

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
    # SAVE FINAL MODEL
    # ========================================================

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    model_data = {

        "model_name":
            "multi_competition_xg_v1",

        "model_type":
            "logistic_regression_with_separate_penalty_probability",

        "penalty_rule": {

            "shot_type":
                "Penalty",

            "xg":
                float(
                    final_penalty_xg
                ),

            "training_penalties":
                int(
                    len(
                        final_penalties
                    )
                ),

            "training_penalty_goals":
                int(
                    final_penalties[
                        TARGET
                    ].sum()
                ),
        },

        "non_penalty_model": {

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
        },

        "training": {

            "shots":
                int(
                    len(data)
                ),

            "non_penalty_shots":
                int(
                    len(
                        final_non_penalties
                    )
                ),

            "penalties":
                int(
                    len(
                        final_penalties
                    )
                ),

            "matches":
                int(
                    data[
                        "match_id"
                    ]
                    .nunique()
                ),

            "goals":
                int(
                    data[
                        TARGET
                    ].sum()
                ),

            "competitions":
                int(
                    data[
                        "source_name"
                    ]
                    .nunique()
                ),

            "cross_validation_folds":
                N_FOLDS,
        },

        "cross_validation": {

            "log_loss":
                float(
                    overall_logloss
                ),

            "brier_score":
                float(
                    overall_brier
                ),

            "roc_auc":
                float(
                    overall_auc
                ),

            "fold_results":
                fold_results,
        },

        "benchmark": {

            "statsbomb_log_loss":
                float(
                    statsbomb_logloss
                ),

            "statsbomb_brier":
                float(
                    statsbomb_brier
                ),

            "statsbomb_roc_auc":
                float(
                    statsbomb_auc
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

    print("\n")
    print("=" * 65)
    print("FINAL MODEL SAVED")
    print("=" * 65)

    print(
        MODEL_PATH
    )

    print(
        "\nEuro 2024 and Copa America 2024 "
        "remain untouched."
    )

    print(
        "They must NOT be used until the "
        "final blind validation."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()