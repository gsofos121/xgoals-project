from pathlib import Path
import json

import numpy as np
import pandas as pd
import xgboost as xgb


# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "multicompetition_training_features.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "xg_multicompetition_xgboost_v4.json"
)

METADATA_FILE = (
    PROJECT_ROOT
    / "models"
    / "xg_multicompetition_xgboost_v4_metadata.json"
)


# ==========================================================
# SETTINGS
# ==========================================================

N_FOLDS = 5
RANDOM_SEED = 42

MAX_BOOST_ROUNDS = 2000
EARLY_STOPPING_ROUNDS = 100

EPSILON = 1e-15


# ==========================================================
# XGBOOST PARAMETERS
# ==========================================================

XGB_PARAMS = {

    # Binary probability prediction
    "objective":
        "binary:logistic",

    # Main optimization / monitoring metric
    "eval_metric":
        "logloss",

    # Conservative tree depth
    "max_depth":
        3,

    # Small learning rate
    "eta":
        0.03,

    # Row subsampling
    "subsample":
        0.80,

    # Feature subsampling
    "colsample_bytree":
        0.80,

    # Minimum weight required in child node
    "min_child_weight":
        5,

    # L2 regularization
    "lambda":
        1.0,

    # L1 regularization
    "alpha":
        0.0,

    # Minimum loss reduction required to split
    "gamma":
        0.0,

    # Reproducibility
    "seed":
        RANDOM_SEED,

    # CPU training
    "device":
        "cpu",

    # Histogram algorithm is efficient on CPU
    "tree_method":
        "hist",

    # Keep output clean
    "verbosity":
        0,
}


# ==========================================================
# FEATURES
#
# Same 12 base features as V2.
#
# We deliberately do NOT add:
#
# distance_sq
# angle_sq
# distance_angle
# distance_head
# counter_one_on_one
# corner_head
#
# XGBoost can learn nonlinearities/interactions itself.
# ==========================================================

FEATURE_COLUMNS = [

    "distance_m",
    "angle_deg",

    "under_pressure",
    "shot_first_time",
    "shot_one_on_one",
    "shot_open_goal",

    "body_left_foot",
    "body_head",

    "type_free_kick",

    "play_regular",
    "play_counter",
    "play_corner",
]


# ==========================================================
# HELPERS
# ==========================================================

def bool_to_int(series):

    return (
        series
        .fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(
            [
                "true",
                "1",
                "yes",
            ]
        )
        .astype(int)
    )


# ==========================================================
# METRICS
# ==========================================================

def log_loss(
    y_true,
    probabilities
):

    probabilities = np.clip(
        probabilities,
        EPSILON,
        1 - EPSILON
    )

    return float(
        -np.mean(
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
    )


def brier_score(
    y_true,
    probabilities
):

    return float(
        np.mean(
            (
                probabilities
                - y_true
            )
            ** 2
        )
    )


def roc_auc(
    y_true,
    scores
):

    y_true = np.asarray(
        y_true
    )

    scores = np.asarray(
        scores
    )

    positive_mask = (
        y_true == 1
    )

    negative_mask = (
        y_true == 0
    )

    n_positive = int(
        positive_mask.sum()
    )

    n_negative = int(
        negative_mask.sum()
    )

    if (
        n_positive == 0
        or
        n_negative == 0
    ):

        return float("nan")


    order = np.argsort(
        scores
    )

    sorted_scores = (
        scores[
            order
        ]
    )

    ranks = np.empty(
        len(scores),
        dtype=float
    )

    i = 0

    while i < len(scores):

        j = i + 1

        while (
            j < len(scores)
            and
            sorted_scores[j]
            == sorted_scores[i]
        ):

            j += 1


        average_rank = (
            (
                i + 1
            )
            + j
        ) / 2.0


        ranks[
            order[i:j]
        ] = average_rank

        i = j


    positive_rank_sum = (
        ranks[
            positive_mask
        ]
        .sum()
    )


    auc = (
        positive_rank_sum
        -
        n_positive
        * (
            n_positive + 1
        )
        / 2
    ) / (
        n_positive
        * n_negative
    )


    return float(
        auc
    )


def calculate_metrics(
    y_true,
    predictions
):

    return {

        "log_loss":
            log_loss(
                y_true,
                predictions
            ),

        "brier_score":
            brier_score(
                y_true,
                predictions
            ),

        "roc_auc":
            roc_auc(
                y_true,
                predictions
            ),

        "predicted_goals":
            float(
                predictions.sum()
            ),
    }


# ==========================================================
# LOAD DATA
# ==========================================================

print("=" * 95)
print("MULTI-COMPETITION xG — XGBOOST V4")
print("=" * 95)

print(
    f"\nXGBoost version: "
    f"{xgb.__version__}"
)

print(
    f"\nLoading:\n"
    f"{INPUT_FILE}"
)


df = pd.read_csv(
    INPUT_FILE
)


print(
    f"\nRows loaded: "
    f"{len(df):,}"
)

print(
    f"Columns: "
    f"{len(df.columns):,}"
)


# ==========================================================
# TARGET
# ==========================================================

print("\n" + "=" * 95)
print("TARGET")
print("=" * 95)


if "is_goal" in df.columns:

    df["is_goal"] = (
        pd.to_numeric(
            df[
                "is_goal"
            ],
            errors="coerce"
        )
        .fillna(0)
        .astype(int)
    )

    print(
        "\nUsing existing is_goal column."
    )


elif "shot_outcome" in df.columns:

    print(
        "\nNo is_goal column found."
        "\nCreating is_goal from shot_outcome."
    )

    df["is_goal"] = (
        df[
            "shot_outcome"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .eq(
            "goal"
        )
        .astype(int)
    )


else:

    raise SystemExit(
        "\nERROR: "
        "Neither is_goal nor shot_outcome exists."
    )


print(
    f"Goals detected: "
    f"{int(df['is_goal'].sum()):,}"
)


# ==========================================================
# REQUIRED RAW COLUMNS
# ==========================================================

required_columns = [

    "match_id",

    "distance_m",
    "angle_deg",

    "under_pressure",
    "shot_first_time",
    "shot_one_on_one",
    "shot_open_goal",

    "shot_body_part",
    "shot_type",
    "play_pattern",
]


missing_columns = [

    column
    for column in required_columns

    if column not in df.columns
]


if missing_columns:

    print(
        "\nMissing required columns:"
    )

    for column in missing_columns:

        print(
            f"  {column}"
        )

    raise SystemExit(
        "\nTraining stopped."
    )


# ==========================================================
# BUILD FEATURES
# ==========================================================

features = pd.DataFrame(
    index=df.index
)


# ----------------------------------------------------------
# Geometry
# ----------------------------------------------------------

features[
    "distance_m"
] = pd.to_numeric(
    df[
        "distance_m"
    ],
    errors="coerce"
)


features[
    "angle_deg"
] = pd.to_numeric(
    df[
        "angle_deg"
    ],
    errors="coerce"
)


# ----------------------------------------------------------
# Shot conditions
# ----------------------------------------------------------

features[
    "under_pressure"
] = bool_to_int(
    df[
        "under_pressure"
    ]
)


features[
    "shot_first_time"
] = bool_to_int(
    df[
        "shot_first_time"
    ]
)


features[
    "shot_one_on_one"
] = bool_to_int(
    df[
        "shot_one_on_one"
    ]
)


features[
    "shot_open_goal"
] = bool_to_int(
    df[
        "shot_open_goal"
    ]
)


# ----------------------------------------------------------
# Body part
#
# Right Foot = baseline:
#
# body_left_foot = 0
# body_head = 0
# ----------------------------------------------------------

body_part = (
    df[
        "shot_body_part"
    ]
    .fillna("")
    .astype(str)
    .str.strip()
)


features[
    "body_left_foot"
] = (
    body_part
    .eq(
        "Left Foot"
    )
    .astype(int)
)


features[
    "body_head"
] = (
    body_part
    .eq(
        "Head"
    )
    .astype(int)
)


# ----------------------------------------------------------
# Shot type
# ----------------------------------------------------------

shot_type = (
    df[
        "shot_type"
    ]
    .fillna("")
    .astype(str)
    .str.strip()
)


features[
    "type_free_kick"
] = (
    shot_type
    .eq(
        "Free Kick"
    )
    .astype(int)
)


# ----------------------------------------------------------
# Play pattern
# ----------------------------------------------------------

play_pattern = (
    df[
        "play_pattern"
    ]
    .fillna("")
    .astype(str)
    .str.strip()
)


features[
    "play_regular"
] = (
    play_pattern
    .eq(
        "Regular Play"
    )
    .astype(int)
)


features[
    "play_counter"
] = (
    play_pattern
    .eq(
        "From Counter"
    )
    .astype(int)
)


features[
    "play_corner"
] = (
    play_pattern
    .eq(
        "From Corner"
    )
    .astype(int)
)


# ==========================================================
# FEATURE SUMMARY
# ==========================================================

print("\n" + "=" * 95)
print("V4 FEATURE SET")
print("=" * 95)


print(
    f"\nTotal features: "
    f"{len(FEATURE_COLUMNS)}"
)


for feature_name in FEATURE_COLUMNS:

    print(
        f"  {feature_name}"
    )


# ==========================================================
# PENALTIES
# ==========================================================

is_penalty = (
    shot_type
    .eq(
        "Penalty"
    )
)


penalty_df = (
    df.loc[
        is_penalty
    ]
    .copy()
)


penalty_shots = int(
    len(
        penalty_df
    )
)


penalty_goals = int(
    penalty_df[
        "is_goal"
    ]
    .sum()
)


penalty_xg = (
    penalty_goals
    /
    penalty_shots
)


# ==========================================================
# NON-PENALTIES
# ==========================================================

non_penalty_mask = (
    ~is_penalty
)


valid_feature_mask = (
    features[
        FEATURE_COLUMNS
    ]
    .notna()
    .all(
        axis=1
    )
)


usable_non_penalty_mask = (
    non_penalty_mask
    &
    valid_feature_mask
)


invalid_non_penalty_rows = int(
    (
        non_penalty_mask
        &
        ~valid_feature_mask
    )
    .sum()
)


if invalid_non_penalty_rows > 0:

    print(
        f"\nInvalid non-penalty rows removed: "
        f"{invalid_non_penalty_rows:,}"
    )


non_penalty_df = (
    df.loc[
        usable_non_penalty_mask
    ]
    .copy()
)


non_penalty_features = (
    features.loc[
        usable_non_penalty_mask,
        FEATURE_COLUMNS
    ]
    .copy()
)


# ==========================================================
# DATA SUMMARY
# ==========================================================

print("\n" + "=" * 95)
print("DATA SUMMARY")
print("=" * 95)


print(
    f"Total shots: "
    f"{len(df):,}"
)


print(
    f"Total goals: "
    f"{int(df['is_goal'].sum()):,}"
)


print(
    f"Matches: "
    f"{df['match_id'].nunique():,}"
)


print(
    f"Non-penalty shots: "
    f"{len(non_penalty_df):,}"
)


print(
    f"Penalties: "
    f"{penalty_shots:,}"
)


print(
    f"Penalty goals: "
    f"{penalty_goals:,}"
)


print(
    f"Fixed penalty xG: "
    f"{penalty_xg:.6f}"
)


# ==========================================================
# ARRAYS
# ==========================================================

X_all = (
    non_penalty_features[
        FEATURE_COLUMNS
    ]
    .astype(float)
    .to_numpy()
)


y_all = (
    non_penalty_df[
        "is_goal"
    ]
    .astype(float)
    .to_numpy()
)


match_ids = (
    non_penalty_df[
        "match_id"
    ]
    .astype(str)
    .to_numpy()
)


# ==========================================================
# CREATE SAME STYLE MATCH-LEVEL FOLDS
# ==========================================================

unique_matches = np.array(
    sorted(
        pd.unique(
            match_ids
        )
    )
)


rng = np.random.default_rng(
    RANDOM_SEED
)


rng.shuffle(
    unique_matches
)


fold_match_groups = np.array_split(
    unique_matches,
    N_FOLDS
)


# ==========================================================
# OOF PREDICTIONS
# ==========================================================

oof_non_penalty = np.zeros(
    len(
        y_all
    ),
    dtype=float
)


fold_results = []

best_rounds = []


# ==========================================================
# 5-FOLD CROSS-VALIDATION
# ==========================================================

print("\n" + "=" * 95)
print("5-FOLD MATCH-LEVEL XGBOOST CROSS-VALIDATION")
print("=" * 95)


for (
    fold_number,
    validation_matches
) in enumerate(
    fold_match_groups,
    start=1
):


    validation_match_set = set(
        validation_matches
    )


    validation_mask = np.array(
        [
            match_id
            in validation_match_set

            for match_id in match_ids
        ],
        dtype=bool
    )


    training_mask = (
        ~validation_mask
    )


    # ------------------------------------------------------
    # Fold data
    # ------------------------------------------------------

    X_train = (
        X_all[
            training_mask
        ]
    )


    y_train = (
        y_all[
            training_mask
        ]
    )


    X_validation = (
        X_all[
            validation_mask
        ]
    )


    y_validation = (
        y_all[
            validation_mask
        ]
    )


    # ------------------------------------------------------
    # DMatrix
    # ------------------------------------------------------

    dtrain = xgb.DMatrix(
        X_train,
        label=y_train,
        feature_names=FEATURE_COLUMNS
    )


    dvalidation = xgb.DMatrix(
        X_validation,
        label=y_validation,
        feature_names=FEATURE_COLUMNS
    )


    # ------------------------------------------------------
    # Train
    #
    # The validation fold is used ONLY for early stopping
    # inside this fold.
    # ------------------------------------------------------

    booster = xgb.train(

        params=
            XGB_PARAMS,

        dtrain=
            dtrain,

        num_boost_round=
            MAX_BOOST_ROUNDS,

        evals=[
            (
                dtrain,
                "train"
            ),
            (
                dvalidation,
                "validation"
            ),
        ],

        early_stopping_rounds=
            EARLY_STOPPING_ROUNDS,

        verbose_eval=
            False,
    )


    # ------------------------------------------------------
    # Best number of boosting rounds
    #
    # best_iteration is zero-based.
    # ------------------------------------------------------

    best_iteration = int(
        booster.best_iteration
    )


    fold_best_rounds = (
        best_iteration
        + 1
    )


    best_rounds.append(
        fold_best_rounds
    )


    # ------------------------------------------------------
    # Predict ONLY using trees up to best iteration
    # ------------------------------------------------------

    validation_predictions = (
        booster.predict(
            dvalidation,
            iteration_range=(
                0,
                fold_best_rounds
            )
        )
    )


    oof_non_penalty[
        validation_mask
    ] = (
        validation_predictions
    )


    # ------------------------------------------------------
    # Fold metrics
    # ------------------------------------------------------

    metrics = calculate_metrics(
        y_validation,
        validation_predictions
    )


    fold_result = {

        "fold":
            fold_number,

        "validation_matches":
            int(
                len(
                    validation_matches
                )
            ),

        "validation_shots":
            int(
                validation_mask.sum()
            ),

        "validation_goals":
            int(
                y_validation.sum()
            ),

        "best_boost_rounds":
            int(
                fold_best_rounds
            ),

        "best_validation_logloss":
            float(
                booster.best_score
            ),

        "log_loss":
            metrics[
                "log_loss"
            ],

        "brier_score":
            metrics[
                "brier_score"
            ],

        "roc_auc":
            metrics[
                "roc_auc"
            ],

        "predicted_goals":
            metrics[
                "predicted_goals"
            ],
    }


    fold_results.append(
        fold_result
    )


    # ------------------------------------------------------
    # Print fold
    # ------------------------------------------------------

    print(
        f"\nFold {fold_number}"
    )

    print(
        "-" * 95
    )

    print(
        f"Validation matches: "
        f"{len(validation_matches):,}"
    )

    print(
        f"Validation shots:   "
        f"{int(validation_mask.sum()):,}"
    )

    print(
        f"Validation goals:   "
        f"{int(y_validation.sum()):,}"
    )

    print(
        f"Best boost rounds:  "
        f"{fold_best_rounds:,}"
    )

    print(
        f"Log Loss:           "
        f"{metrics['log_loss']:.6f}"
    )

    print(
        f"Brier:              "
        f"{metrics['brier_score']:.6f}"
    )

    print(
        f"ROC-AUC:            "
        f"{metrics['roc_auc']:.6f}"
    )

    print(
        f"Predicted goals:    "
        f"{metrics['predicted_goals']:.2f}"
    )


# ==========================================================
# FULL OOF INCLUDING PENALTIES
# ==========================================================

full_y = (
    df[
        "is_goal"
    ]
    .astype(float)
    .to_numpy()
)


full_predictions = np.zeros(
    len(df),
    dtype=float
)


non_penalty_indices = (
    non_penalty_df
    .index
    .to_numpy()
)


penalty_indices = (
    penalty_df
    .index
    .to_numpy()
)


full_predictions[
    non_penalty_indices
] = (
    oof_non_penalty
)


full_predictions[
    penalty_indices
] = (
    penalty_xg
)


# ==========================================================
# COVERAGE CHECK
# ==========================================================

expected_indices = set(
    df.index
)


covered_indices = set(
    non_penalty_indices
).union(
    set(
        penalty_indices
    )
)


if (
    expected_indices
    !=
    covered_indices
):

    missing_indices = (
        expected_indices
        -
        covered_indices
    )

    raise SystemExit(
        "\nERROR: "
        f"{len(missing_indices)} rows do not have predictions."
    )


# ==========================================================
# FULL OOF METRICS
# ==========================================================

overall_metrics = calculate_metrics(
    full_y,
    full_predictions
)


print("\n" + "=" * 95)
print("FULL OOF RESULTS INCLUDING PENALTIES")
print("=" * 95)


print(
    f"\nLog Loss:       "
    f"{overall_metrics['log_loss']:.6f}"
)


print(
    f"Brier:          "
    f"{overall_metrics['brier_score']:.6f}"
)


print(
    f"ROC-AUC:        "
    f"{overall_metrics['roc_auc']:.6f}"
)


print(
    f"Actual goals:   "
    f"{int(full_y.sum()):,}"
)


print(
    f"Predicted goals:"
    f" {overall_metrics['predicted_goals']:.2f}"
)


# ==========================================================
# BEST ROUND SUMMARY
# ==========================================================

best_rounds_array = np.array(
    best_rounds,
    dtype=int
)


median_best_rounds = int(
    np.median(
        best_rounds_array
    )
)


mean_best_rounds = float(
    np.mean(
        best_rounds_array
    )
)


print("\n" + "=" * 95)
print("BOOSTING ROUND SUMMARY")
print("=" * 95)


for (
    fold_number,
    rounds
) in enumerate(
    best_rounds,
    start=1
):

    print(
        f"Fold {fold_number}: "
        f"{rounds:,}"
    )


print(
    f"\nMean best rounds: "
    f"{mean_best_rounds:.1f}"
)


print(
    f"Median best rounds: "
    f"{median_best_rounds:,}"
)


# ==========================================================
# OOF CALIBRATION DECILES
# ==========================================================

calibration_df = pd.DataFrame(
    {
        "actual":
            full_y,

        "prediction":
            full_predictions,
    }
)


calibration_df[
    "bin"
] = pd.qcut(
    calibration_df[
        "prediction"
    ],
    q=10,
    duplicates="drop"
)


calibration_summary = (
    calibration_df
    .groupby(
        "bin",
        observed=True
    )
    .agg(

        shots=(
            "actual",
            "size"
        ),

        predicted=(
            "prediction",
            "mean"
        ),

        actual=(
            "actual",
            "mean"
        ),
    )
    .reset_index(
        drop=True
    )
)


print("\n" + "=" * 95)
print("OOF CALIBRATION DECILES")
print("=" * 95)


print(
    calibration_summary.to_string(

        index=False,

        formatters={

            "predicted":
                lambda value:
                    f"{value:.3f}",

            "actual":
                lambda value:
                    f"{value:.3f}",
        }
    )
)


# ==========================================================
# TRAIN FINAL MODEL
#
# We use the median number of best boosting rounds found
# across the 5 validation folds.
#
# The final model sees ALL non-penalty training data.
# ==========================================================

print("\n" + "=" * 95)
print("TRAINING FINAL XGBOOST V4")
print("=" * 95)


print(
    f"\nFinal boost rounds: "
    f"{median_best_rounds:,}"
)


dfull = xgb.DMatrix(

    X_all,

    label=
        y_all,

    feature_names=
        FEATURE_COLUMNS
)


final_booster = xgb.train(

    params=
        XGB_PARAMS,

    dtrain=
        dfull,

    num_boost_round=
        median_best_rounds,

    evals=[
        (
            dfull,
            "train"
        )
    ],

    verbose_eval=
        False,
)


# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

importance_gain = (
    final_booster
    .get_score(
        importance_type="gain"
    )
)


importance_weight = (
    final_booster
    .get_score(
        importance_type="weight"
    )
)


importance_rows = []


for feature_name in FEATURE_COLUMNS:

    importance_rows.append(
        {
            "feature":
                feature_name,

            "gain":
                float(
                    importance_gain.get(
                        feature_name,
                        0.0
                    )
                ),

            "weight":
                float(
                    importance_weight.get(
                        feature_name,
                        0.0
                    )
                ),
        }
    )


importance_rows = sorted(

    importance_rows,

    key=
        lambda item:
            item["gain"],

    reverse=
        True
)


print("\n" + "=" * 95)
print("FEATURE IMPORTANCE — GAIN")
print("=" * 95)


for row in importance_rows:

    print(
        f"{row['feature']:<24}"
        f"{row['gain']:>14.6f}"
    )


# ==========================================================
# SAVE NATIVE XGBOOST MODEL
# ==========================================================

MODEL_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


final_booster.save_model(
    MODEL_FILE
)


# ==========================================================
# METADATA
# ==========================================================

metadata = {

    "model_name":
        "multi_competition_xg_v4_xgboost",

    "model_type":
        "xgboost_with_separate_penalty_probability",

    "xgboost_version":
        xgb.__version__,

    "model_file":
        MODEL_FILE.name,

    "feature_columns":
        FEATURE_COLUMNS,

    "penalty_rule": {

        "shot_type":
            "Penalty",

        "xg":
            float(
                penalty_xg
            ),

        "training_penalties":
            int(
                penalty_shots
            ),

        "training_penalty_goals":
            int(
                penalty_goals
            ),
    },

    "play_pattern_mapping": {

        "Regular Play":
            "play_regular",

        "Counterattack":
            "play_counter",

        "Corner":
            "play_corner",
    },

    "source_play_pattern_mapping": {

        "Regular Play":
            "Regular Play",

        "Counterattack":
            "From Counter",

        "Corner":
            "From Corner",
    },

    "xgboost_parameters":
        XGB_PARAMS,

    "training": {

        "shots":
            int(
                len(df)
            ),

        "non_penalty_shots":
            int(
                len(non_penalty_df)
            ),

        "penalties":
            int(
                penalty_shots
            ),

        "goals":
            int(
                df[
                    "is_goal"
                ]
                .sum()
            ),

        "matches":
            int(
                df[
                    "match_id"
                ]
                .nunique()
            ),

        "cross_validation_folds":
            N_FOLDS,
    },

    "boosting": {

        "maximum_cv_rounds":
            MAX_BOOST_ROUNDS,

        "early_stopping_rounds":
            EARLY_STOPPING_ROUNDS,

        "fold_best_rounds":
            [
                int(value)
                for value in best_rounds
            ],

        "mean_best_rounds":
            float(
                mean_best_rounds
            ),

        "median_best_rounds":
            int(
                median_best_rounds
            ),

        "final_model_rounds":
            int(
                median_best_rounds
            ),
    },

    "cross_validation": {

        "log_loss":
            float(
                overall_metrics[
                    "log_loss"
                ]
            ),

        "brier_score":
            float(
                overall_metrics[
                    "brier_score"
                ]
            ),

        "roc_auc":
            float(
                overall_metrics[
                    "roc_auc"
                ]
            ),

        "actual_goals":
            int(
                full_y.sum()
            ),

        "predicted_goals":
            float(
                overall_metrics[
                    "predicted_goals"
                ]
            ),

        "fold_results":
            fold_results,
    },

    "feature_importance_gain":
        {
            row[
                "feature"
            ]:
                row[
                    "gain"
                ]

            for row in importance_rows
        },

    "feature_importance_weight":
        {
            row[
                "feature"
            ]:
                row[
                    "weight"
                ]

            for row in importance_rows
        },
}


with open(
    METADATA_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metadata,
        file,
        indent=2,
        ensure_ascii=False
    )


# ==========================================================
# FINAL OUTPUT
# ==========================================================

print("\n" + "=" * 95)
print("MODEL SAVED")
print("=" * 95)


print(
    f"\nNative XGBoost model:\n"
    f"{MODEL_FILE}"
)


print(
    f"\nMetadata:\n"
    f"{METADATA_FILE}"
)


print(
    "\nExisting logistic models remain untouched:"
)


print(
    PROJECT_ROOT
    / "models"
    / "xg_multicompetition_logistic.json"
)


print(
    PROJECT_ROOT
    / "models"
    / "xg_multicompetition_logistic_v2.json"
)


print(
    PROJECT_ROOT
    / "models"
    / "xg_multicompetition_logistic_v3.json"
)


print("\n" + "=" * 95)
print("V4 XGBOOST TRAINING COMPLETE")
print("=" * 95)