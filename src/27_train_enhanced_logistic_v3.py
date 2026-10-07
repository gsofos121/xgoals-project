from pathlib import Path
import json

import numpy as np
import pandas as pd


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

OUTPUT_MODEL = (
    PROJECT_ROOT
    / "models"
    / "xg_multicompetition_logistic_v3.json"
)


# ==========================================================
# SETTINGS
# ==========================================================

N_FOLDS = 5

MAX_ITER = 15000
LEARNING_RATE = 0.03
L2 = 0.001

EPSILON = 1e-15
RANDOM_SEED = 42


# ==========================================================
# LOAD DATA
# ==========================================================

print("=" * 90)
print("MULTI-COMPETITION xG — ENHANCED LOGISTIC V3")
print("=" * 90)

print(
    f"\nLoading:\n"
    f"{INPUT_FILE}\n"
)

df = pd.read_csv(
    INPUT_FILE
)

print(
    f"Rows loaded: "
    f"{len(df):,}"
)

print(
    f"Columns: "
    f"{len(df.columns):,}"
)


# ==========================================================
# TARGET
# ==========================================================

print("\n" + "=" * 90)
print("TARGET")
print("=" * 90)


if "is_goal" in df.columns:

    df["is_goal"] = (
        pd.to_numeric(
            df["is_goal"],
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
        df["shot_outcome"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("goal")
        .astype(int)
    )


else:

    raise SystemExit(
        "\nERROR: Could not create target."
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

    print("\nMissing required columns:")

    for column in missing_columns:
        print(
            column
        )

    raise SystemExit(
        "\nTraining stopped."
    )


# ==========================================================
# BOOLEAN HELPER
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
# BUILD BASE FEATURES
# ==========================================================

features = pd.DataFrame(
    index=df.index
)


# ----------------------------------------------------------
# Geometry
# ----------------------------------------------------------

features["distance_m"] = (
    pd.to_numeric(
        df["distance_m"],
        errors="coerce"
    )
)

features["angle_deg"] = (
    pd.to_numeric(
        df["angle_deg"],
        errors="coerce"
    )
)


# ----------------------------------------------------------
# Shot conditions
# ----------------------------------------------------------

features["under_pressure"] = (
    bool_to_int(
        df["under_pressure"]
    )
)

features["shot_first_time"] = (
    bool_to_int(
        df["shot_first_time"]
    )
)

features["shot_one_on_one"] = (
    bool_to_int(
        df["shot_one_on_one"]
    )
)

features["shot_open_goal"] = (
    bool_to_int(
        df["shot_open_goal"]
    )
)


# ----------------------------------------------------------
# Body part
#
# Right Foot = reference category
# ----------------------------------------------------------

body_part = (
    df["shot_body_part"]
    .fillna("")
    .astype(str)
    .str.strip()
)


features["body_left_foot"] = (
    body_part
    .eq("Left Foot")
    .astype(int)
)

features["body_head"] = (
    body_part
    .eq("Head")
    .astype(int)
)


# ----------------------------------------------------------
# Shot type
#
# Open Play = reference
# Penalty handled separately
# ----------------------------------------------------------

shot_type = (
    df["shot_type"]
    .fillna("")
    .astype(str)
    .str.strip()
)


features["type_free_kick"] = (
    shot_type
    .eq("Free Kick")
    .astype(int)
)


# ----------------------------------------------------------
# Play pattern
# ----------------------------------------------------------

play_pattern = (
    df["play_pattern"]
    .fillna("")
    .astype(str)
    .str.strip()
)


features["play_regular"] = (
    play_pattern
    .eq("Regular Play")
    .astype(int)
)

features["play_counter"] = (
    play_pattern
    .eq("From Counter")
    .astype(int)
)

features["play_corner"] = (
    play_pattern
    .eq("From Corner")
    .astype(int)
)


# ==========================================================
# NEW V3 NONLINEAR FEATURES
# ==========================================================

# Distance is unlikely to have a purely linear relationship
# with scoring probability.

features["distance_sq"] = (
    features["distance_m"]
    ** 2
)


# Goal angle may also have nonlinear effects.

features["angle_sq"] = (
    features["angle_deg"]
    ** 2
)


# Geometry interaction:
# distance and angle jointly describe shot quality.

features["distance_angle"] = (
    features["distance_m"]
    *
    features["angle_deg"]
)


# Headers may behave differently depending on distance.

features["distance_head"] = (
    features["distance_m"]
    *
    features["body_head"]
)


# One-on-ones arising from counterattacks may behave
# differently from ordinary one-on-one situations.

features["counter_one_on_one"] = (
    features["play_counter"]
    *
    features["shot_one_on_one"]
)


# Headers originating from corners are a distinct,
# common football situation.

features["corner_head"] = (
    features["play_corner"]
    *
    features["body_head"]
)


# ==========================================================
# FEATURE SETS
# ==========================================================

V2_FEATURES = [
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


V3_EXTRA_FEATURES = [
    "distance_sq",
    "angle_sq",
    "distance_angle",
    "distance_head",
    "counter_one_on_one",
    "corner_head",
]


FEATURE_COLUMNS = (
    V2_FEATURES
    +
    V3_EXTRA_FEATURES
)


# ==========================================================
# FEATURE SUMMARY
# ==========================================================

print("\n" + "=" * 90)
print("V3 FEATURE SET")
print("=" * 90)

print(
    f"\nBase V2 features: "
    f"{len(V2_FEATURES)}"
)

for feature in V2_FEATURES:
    print(
        f"  {feature}"
    )


print(
    f"\nNew V3 features: "
    f"{len(V3_EXTRA_FEATURES)}"
)

for feature in V3_EXTRA_FEATURES:
    print(
        f"  {feature}"
    )


print(
    f"\nTotal V3 features: "
    f"{len(FEATURE_COLUMNS)}"
)


# ==========================================================
# PENALTY SEPARATION
# ==========================================================

is_penalty = (
    shot_type
    .eq("Penalty")
)


penalty_df = (
    df[
        is_penalty
    ]
    .copy()
)


non_penalty_df = (
    df[
        ~is_penalty
    ]
    .copy()
)


non_penalty_features = (
    features
    .loc[
        ~is_penalty
    ]
    .copy()
)


# ==========================================================
# VALID ROWS
# ==========================================================

valid_mask = (
    non_penalty_features[
        FEATURE_COLUMNS
    ]
    .notna()
    .all(axis=1)
)


invalid_rows = int(
    (
        ~valid_mask
    )
    .sum()
)


if invalid_rows > 0:

    print(
        f"\nInvalid non-penalty rows removed: "
        f"{invalid_rows:,}"
    )


non_penalty_features = (
    non_penalty_features[
        valid_mask
    ]
    .copy()
)


non_penalty_df = (
    non_penalty_df
    .loc[
        non_penalty_features.index
    ]
    .copy()
)


# ==========================================================
# PENALTY MODEL
# ==========================================================

penalty_shots = len(
    penalty_df
)

penalty_goals = int(
    penalty_df[
        "is_goal"
    ]
    .sum()
)


penalty_xg = (
    penalty_goals
    / penalty_shots
)


print("\n" + "=" * 90)
print("DATA SUMMARY")
print("=" * 90)

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
    f"Penalty shots: "
    f"{penalty_shots:,}"
)

print(
    f"Penalty goals: "
    f"{penalty_goals:,}"
)

print(
    f"Fixed penalty xG: "
    f"{penalty_xg:.4f}"
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
# STANDARDIZATION
# ==========================================================

def fit_scaler(X):

    mean = (
        X.mean(
            axis=0
        )
    )

    std = (
        X.std(
            axis=0
        )
    )

    std = np.where(
        std == 0,
        1.0,
        std
    )

    return (
        mean,
        std
    )


def transform_scaler(
    X,
    mean,
    std
):

    return (
        X - mean
    ) / std


# ==========================================================
# LOGISTIC REGRESSION
# ==========================================================

def sigmoid(z):

    z = np.clip(
        z,
        -35,
        35
    )

    return (
        1.0
        /
        (
            1.0
            + np.exp(-z)
        )
    )


def train_logistic(
    X,
    y,
    learning_rate=LEARNING_RATE,
    max_iter=MAX_ITER,
    l2=L2
):

    n_samples = (
        X.shape[0]
    )

    n_features = (
        X.shape[1]
    )


    weights = np.zeros(
        n_features,
        dtype=float
    )

    intercept = 0.0


    for _ in range(
        max_iter
    ):

        linear_score = (
            X @ weights
            + intercept
        )


        probabilities = (
            sigmoid(
                linear_score
            )
        )


        error = (
            probabilities
            - y
        )


        gradient_weights = (
            X.T
            @ error
        ) / n_samples


        gradient_weights += (
            l2
            * weights
        )


        gradient_intercept = (
            error.mean()
        )


        weights -= (
            learning_rate
            * gradient_weights
        )


        intercept -= (
            learning_rate
            * gradient_intercept
        )


    return (
        weights,
        intercept
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


# ==========================================================
# SAME MATCH-LEVEL 5-FOLD CV
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


oof_probabilities = np.zeros(
    len(y_all),
    dtype=float
)


print("\n" + "=" * 90)
print("5-FOLD MATCH-LEVEL CROSS-VALIDATION")
print("=" * 90)


fold_results = []


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
            for match_id
            in match_ids
        ]
    )


    training_mask = (
        ~validation_mask
    )


    X_train_raw = (
        X_all[
            training_mask
        ]
    )


    y_train = (
        y_all[
            training_mask
        ]
    )


    X_validation_raw = (
        X_all[
            validation_mask
        ]
    )


    y_validation = (
        y_all[
            validation_mask
        ]
    )


    scaler_mean, scaler_std = (
        fit_scaler(
            X_train_raw
        )
    )


    X_train = (
        transform_scaler(
            X_train_raw,
            scaler_mean,
            scaler_std
        )
    )


    X_validation = (
        transform_scaler(
            X_validation_raw,
            scaler_mean,
            scaler_std
        )
    )


    weights, intercept = (
        train_logistic(
            X_train,
            y_train
        )
    )


    validation_probabilities = (
        sigmoid(
            X_validation
            @ weights
            + intercept
        )
    )


    oof_probabilities[
        validation_mask
    ] = (
        validation_probabilities
    )


    fold_log_loss = (
        log_loss(
            y_validation,
            validation_probabilities
        )
    )


    fold_brier = (
        brier_score(
            y_validation,
            validation_probabilities
        )
    )


    fold_auc = (
        roc_auc(
            y_validation,
            validation_probabilities
        )
    )


    fold_results.append(
        {
            "fold":
                fold_number,

            "log_loss":
                fold_log_loss,

            "brier":
                fold_brier,

            "roc_auc":
                fold_auc,
        }
    )


    print(
        f"\nFold {fold_number}"
    )

    print(
        f"Validation matches: "
        f"{len(validation_matches):,}"
    )

    print(
        f"Validation shots: "
        f"{len(y_validation):,}"
    )

    print(
        f"Validation goals: "
        f"{int(y_validation.sum()):,}"
    )

    print(
        f"Log Loss: "
        f"{fold_log_loss:.4f}"
    )

    print(
        f"Brier: "
        f"{fold_brier:.4f}"
    )

    print(
        f"ROC-AUC: "
        f"{fold_auc:.4f}"
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


full_probabilities = np.zeros(
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


full_probabilities[
    non_penalty_indices
] = (
    oof_probabilities
)


full_probabilities[
    penalty_indices
] = (
    penalty_xg
)


full_log_loss = (
    log_loss(
        full_y,
        full_probabilities
    )
)


full_brier = (
    brier_score(
        full_y,
        full_probabilities
    )
)


full_auc = (
    roc_auc(
        full_y,
        full_probabilities
    )
)


print("\n" + "=" * 90)
print("FULL OOF RESULTS INCLUDING PENALTIES")
print("=" * 90)

print(
    f"Log Loss: "
    f"{full_log_loss:.4f}"
)

print(
    f"Brier: "
    f"{full_brier:.4f}"
)

print(
    f"ROC-AUC: "
    f"{full_auc:.4f}"
)

print(
    f"Actual goals: "
    f"{int(full_y.sum()):,}"
)

print(
    f"Predicted goals: "
    f"{full_probabilities.sum():.1f}"
)


# ==========================================================
# TRAIN FINAL V3
# ==========================================================

print("\n" + "=" * 90)
print("TRAINING FINAL V3 MODEL")
print("=" * 90)


final_mean, final_std = (
    fit_scaler(
        X_all
    )
)


X_scaled = (
    transform_scaler(
        X_all,
        final_mean,
        final_std
    )
)


final_weights_scaled, final_intercept_scaled = (
    train_logistic(
        X_scaled,
        y_all
    )
)


# ==========================================================
# CONVERT BACK TO RAW FEATURE SCALE
# ==========================================================

raw_weights = (
    final_weights_scaled
    / final_std
)


raw_intercept = (
    final_intercept_scaled
    -
    np.sum(
        final_weights_scaled
        * final_mean
        / final_std
    )
)


# ==========================================================
# COEFFICIENTS
# ==========================================================

coefficients = {}


print("\nFinal raw-scale coefficients:\n")


for (
    feature_name,
    coefficient
) in zip(
    FEATURE_COLUMNS,
    raw_weights
):

    coefficients[
        feature_name
    ] = float(
        coefficient
    )


    print(
        f"{feature_name:<24}"
        f"{coefficient:+.8f}"
    )


print(
    f"{'intercept':<24}"
    f"{raw_intercept:+.8f}"
)


print(
    f"{'fixed_penalty_xg':<24}"
    f"{penalty_xg:.8f}"
)


# ==========================================================
# PRINT NEW V3 TERMS
# ==========================================================

print("\n" + "=" * 90)
print("NEW V3 COEFFICIENTS")
print("=" * 90)


for feature_name in V3_EXTRA_FEATURES:

    print(
        f"{feature_name:<24}"
        f"{coefficients[feature_name]:+.8f}"
    )


# ==========================================================
# CALIBRATION DECILES
# ==========================================================

calibration_df = pd.DataFrame(
    {
        "actual":
            full_y,

        "prediction":
            full_probabilities,
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


print("\n" + "=" * 90)
print("OOF CALIBRATION DECILES")
print("=" * 90)


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
# SAVE MODEL
# Same structure as V1 and V2
# ==========================================================

model_payload = {

    "model_name":
        "multi_competition_xg_v3_enhanced_logistic",

    "model_type":
        "enhanced_logistic_regression_with_separate_penalty_probability",

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

    "non_penalty_model": {

        "features":
            FEATURE_COLUMNS,

        "intercept":
            float(
                raw_intercept
            ),

        "coefficients":
            coefficients,
    },

    "feature_engineering": {

        "distance_sq":
            "distance_m ** 2",

        "angle_sq":
            "angle_deg ** 2",

        "distance_angle":
            "distance_m * angle_deg",

        "distance_head":
            "distance_m * body_head",

        "counter_one_on_one":
            "play_counter * shot_one_on_one",

        "corner_head":
            "play_corner * body_head",
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

        "matches":
            int(
                df[
                    "match_id"
                ]
                .nunique()
            ),

        "goals":
            int(
                df[
                    "is_goal"
                ]
                .sum()
            ),

        "cross_validation_folds":
            N_FOLDS,
    },

    "cross_validation": {

        "log_loss":
            float(
                full_log_loss
            ),

        "brier_score":
            float(
                full_brier
            ),

        "roc_auc":
            float(
                full_auc
            ),

        "fold_results":
            fold_results,
    },
}


OUTPUT_MODEL.parent.mkdir(
    parents=True,
    exist_ok=True
)


with open(
    OUTPUT_MODEL,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        model_payload,
        file,
        indent=2,
        ensure_ascii=False
    )


# ==========================================================
# FINISHED
# ==========================================================

print("\n" + "=" * 90)
print("MODEL SAVED")
print("=" * 90)

print(
    f"\nV3:\n"
    f"{OUTPUT_MODEL}"
)


print(
    "\nV1 remains untouched:"
)

print(
    PROJECT_ROOT
    / "models"
    / "xg_multicompetition_logistic.json"
)


print(
    "\nV2 remains untouched:"
)

print(
    PROJECT_ROOT
    / "models"
    / "xg_multicompetition_logistic_v2.json"
)


print("\n" + "=" * 90)
print("V3 TRAINING COMPLETE")
print("=" * 90)