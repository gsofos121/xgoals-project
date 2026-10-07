from pathlib import Path

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
print("DIRECT V1 vs V2 xG COMPARISON")
print("=" * 90)

print(f"\nLoading:\n{INPUT_FILE}\n")

df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded: {len(df):,}")
print(f"Columns: {len(df.columns):,}")


# ==========================================================
# CREATE TARGET
# ==========================================================

if "is_goal" in df.columns:

    df["is_goal"] = (
        pd.to_numeric(
            df["is_goal"],
            errors="coerce"
        )
        .fillna(0)
        .astype(int)
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
# REQUIRED COLUMNS
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
        print(column)

    raise SystemExit(
        "\nComparison stopped."
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
# BUILD COMMON FEATURES
# ==========================================================

features = pd.DataFrame(
    index=df.index
)


# Geometry

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


# Existing shot conditions

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


# Body part

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


# Shot type

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


# ==========================================================
# V2 PLAY-PATTERN FEATURES
# ==========================================================

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
# FEATURE SETS
# ==========================================================

V1_FEATURES = [
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


print("\n" + "=" * 90)
print("FEATURE SETS")
print("=" * 90)

print(
    f"\nV1 features: "
    f"{len(V1_FEATURES)}"
)

for feature in V1_FEATURES:
    print(f"  {feature}")


print(
    f"\nV2 features: "
    f"{len(V2_FEATURES)}"
)

for feature in V2_FEATURES:
    print(f"  {feature}")


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
# SAME VALID ROWS FOR BOTH MODELS
# ==========================================================

all_comparison_features = list(
    dict.fromkeys(
        V1_FEATURES
        + V2_FEATURES
    )
)

valid_mask = (
    non_penalty_features[
        all_comparison_features
    ]
    .notna()
    .all(axis=1)
)

invalid_rows = int(
    (~valid_mask).sum()
)

if invalid_rows > 0:

    print(
        f"\nInvalid rows removed: "
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
# PENALTY xG
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


# ==========================================================
# DATA SUMMARY
# ==========================================================

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
    f"Fixed penalty xG: "
    f"{penalty_xg:.4f}"
)


# ==========================================================
# ARRAYS
# ==========================================================

X_v1 = (
    non_penalty_features[
        V1_FEATURES
    ]
    .astype(float)
    .to_numpy()
)

X_v2 = (
    non_penalty_features[
        V2_FEATURES
    ]
    .astype(float)
    .to_numpy()
)

y = (
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

    mean = X.mean(
        axis=0
    )

    std = X.std(
        axis=0
    )

    std = np.where(
        std == 0,
        1.0,
        std
    )

    return mean, std


def transform_scaler(
    X,
    mean,
    std,
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
):

    n_samples = X.shape[0]
    n_features = X.shape[1]

    weights = np.zeros(
        n_features,
        dtype=float
    )

    intercept = 0.0


    for _ in range(
        MAX_ITER
    ):

        probabilities = sigmoid(
            X @ weights
            + intercept
        )

        error = (
            probabilities
            - y
        )

        gradient_weights = (
            X.T @ error
        ) / n_samples

        gradient_weights += (
            L2
            * weights
        )

        gradient_intercept = (
            error.mean()
        )

        weights -= (
            LEARNING_RATE
            * gradient_weights
        )

        intercept -= (
            LEARNING_RATE
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
    probabilities,
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
    probabilities,
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
    scores,
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
# TRAIN + PREDICT ONE MODEL
# ==========================================================

def train_and_predict(
    X,
    y,
    training_mask,
    validation_mask,
):

    X_train_raw = (
        X[
            training_mask
        ]
    )

    X_validation_raw = (
        X[
            validation_mask
        ]
    )

    y_train = (
        y[
            training_mask
        ]
    )

    mean, std = fit_scaler(
        X_train_raw
    )

    X_train = transform_scaler(
        X_train_raw,
        mean,
        std,
    )

    X_validation = transform_scaler(
        X_validation_raw,
        mean,
        std,
    )

    weights, intercept = (
        train_logistic(
            X_train,
            y_train,
        )
    )

    probabilities = sigmoid(
        X_validation
        @ weights
        + intercept
    )

    return probabilities


# ==========================================================
# CREATE IDENTICAL MATCH FOLDS
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


oof_v1 = np.zeros(
    len(y),
    dtype=float
)

oof_v2 = np.zeros(
    len(y),
    dtype=float
)


# ==========================================================
# 5-FOLD DIRECT COMPARISON
# ==========================================================

print("\n" + "=" * 90)
print("5-FOLD DIRECT V1 vs V2 COMPARISON")
print("=" * 90)


fold_results = []


for (
    fold_number,
    validation_matches
) in enumerate(
    fold_match_groups,
    start=1,
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

    y_validation = (
        y[
            validation_mask
        ]
    )


    # ------------------------------------------------------
    # V1
    # ------------------------------------------------------

    probabilities_v1 = (
        train_and_predict(
            X_v1,
            y,
            training_mask,
            validation_mask,
        )
    )

    oof_v1[
        validation_mask
    ] = probabilities_v1


    # ------------------------------------------------------
    # V2
    # ------------------------------------------------------

    probabilities_v2 = (
        train_and_predict(
            X_v2,
            y,
            training_mask,
            validation_mask,
        )
    )

    oof_v2[
        validation_mask
    ] = probabilities_v2


    # ------------------------------------------------------
    # METRICS
    # ------------------------------------------------------

    v1_logloss = log_loss(
        y_validation,
        probabilities_v1,
    )

    v1_brier = brier_score(
        y_validation,
        probabilities_v1,
    )

    v1_auc = roc_auc(
        y_validation,
        probabilities_v1,
    )


    v2_logloss = log_loss(
        y_validation,
        probabilities_v2,
    )

    v2_brier = brier_score(
        y_validation,
        probabilities_v2,
    )

    v2_auc = roc_auc(
        y_validation,
        probabilities_v2,
    )


    fold_results.append(
        {
            "fold": fold_number,

            "v1_logloss":
                v1_logloss,

            "v2_logloss":
                v2_logloss,

            "v1_brier":
                v1_brier,

            "v2_brier":
                v2_brier,

            "v1_auc":
                v1_auc,

            "v2_auc":
                v2_auc,
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


    print("\nV1")

    print(
        f"  Log Loss: "
        f"{v1_logloss:.4f}"
    )

    print(
        f"  Brier:    "
        f"{v1_brier:.4f}"
    )

    print(
        f"  ROC-AUC:  "
        f"{v1_auc:.4f}"
    )


    print("\nV2")

    print(
        f"  Log Loss: "
        f"{v2_logloss:.4f}"
    )

    print(
        f"  Brier:    "
        f"{v2_brier:.4f}"
    )

    print(
        f"  ROC-AUC:  "
        f"{v2_auc:.4f}"
    )


    print("\nDifference V2 - V1")

    print(
        f"  Log Loss: "
        f"{v2_logloss - v1_logloss:+.5f}"
    )

    print(
        f"  Brier:    "
        f"{v2_brier - v1_brier:+.5f}"
    )

    print(
        f"  ROC-AUC:  "
        f"{v2_auc - v1_auc:+.5f}"
    )


# ==========================================================
# NON-PENALTY OOF RESULTS
# ==========================================================

v1_nonpen_logloss = log_loss(
    y,
    oof_v1,
)

v1_nonpen_brier = brier_score(
    y,
    oof_v1,
)

v1_nonpen_auc = roc_auc(
    y,
    oof_v1,
)


v2_nonpen_logloss = log_loss(
    y,
    oof_v2,
)

v2_nonpen_brier = brier_score(
    y,
    oof_v2,
)

v2_nonpen_auc = roc_auc(
    y,
    oof_v2,
)


print("\n" + "=" * 90)
print("NON-PENALTY OOF COMPARISON")
print("=" * 90)


print("\nV1")

print(
    f"Log Loss: "
    f"{v1_nonpen_logloss:.4f}"
)

print(
    f"Brier: "
    f"{v1_nonpen_brier:.4f}"
)

print(
    f"ROC-AUC: "
    f"{v1_nonpen_auc:.4f}"
)


print("\nV2")

print(
    f"Log Loss: "
    f"{v2_nonpen_logloss:.4f}"
)

print(
    f"Brier: "
    f"{v2_nonpen_brier:.4f}"
)

print(
    f"ROC-AUC: "
    f"{v2_nonpen_auc:.4f}"
)


# ==========================================================
# FULL RESULTS INCLUDING PENALTIES
# ==========================================================

full_y = (
    df[
        "is_goal"
    ]
    .astype(float)
    .to_numpy()
)

full_v1 = np.zeros(
    len(df),
    dtype=float
)

full_v2 = np.zeros(
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


full_v1[
    non_penalty_indices
] = oof_v1

full_v2[
    non_penalty_indices
] = oof_v2


full_v1[
    penalty_indices
] = penalty_xg

full_v2[
    penalty_indices
] = penalty_xg


# V1 metrics

v1_full_logloss = log_loss(
    full_y,
    full_v1,
)

v1_full_brier = brier_score(
    full_y,
    full_v1,
)

v1_full_auc = roc_auc(
    full_y,
    full_v1,
)


# V2 metrics

v2_full_logloss = log_loss(
    full_y,
    full_v2,
)

v2_full_brier = brier_score(
    full_y,
    full_v2,
)

v2_full_auc = roc_auc(
    full_y,
    full_v2,
)


# ==========================================================
# FINAL COMPARISON
# ==========================================================

print("\n" + "=" * 90)
print("FINAL FAIR COMPARISON — SAME DATA, SAME FOLDS")
print("=" * 90)


print("\nV1")

print(
    f"Log Loss:       "
    f"{v1_full_logloss:.4f}"
)

print(
    f"Brier:          "
    f"{v1_full_brier:.4f}"
)

print(
    f"ROC-AUC:        "
    f"{v1_full_auc:.4f}"
)

print(
    f"Predicted goals: "
    f"{full_v1.sum():.1f}"
)


print("\nV2")

print(
    f"Log Loss:       "
    f"{v2_full_logloss:.4f}"
)

print(
    f"Brier:          "
    f"{v2_full_brier:.4f}"
)

print(
    f"ROC-AUC:        "
    f"{v2_full_auc:.4f}"
)

print(
    f"Predicted goals: "
    f"{full_v2.sum():.1f}"
)


print("\nActual goals:")

print(
    f"{int(full_y.sum()):,}"
)


# ==========================================================
# IMPROVEMENT
# ==========================================================

delta_logloss = (
    v2_full_logloss
    - v1_full_logloss
)

delta_brier = (
    v2_full_brier
    - v1_full_brier
)

delta_auc = (
    v2_full_auc
    - v1_full_auc
)


print("\n" + "=" * 90)
print("V2 CHANGE RELATIVE TO V1")
print("=" * 90)

print(
    f"Log Loss change: "
    f"{delta_logloss:+.6f}"
)

print(
    f"Brier change: "
    f"{delta_brier:+.6f}"
)

print(
    f"ROC-AUC change: "
    f"{delta_auc:+.6f}"
)


# ==========================================================
# FOLD WIN COUNTS
# ==========================================================

v2_logloss_wins = sum(
    result["v2_logloss"]
    <
    result["v1_logloss"]

    for result in fold_results
)

v2_brier_wins = sum(
    result["v2_brier"]
    <
    result["v1_brier"]

    for result in fold_results
)

v2_auc_wins = sum(
    result["v2_auc"]
    >
    result["v1_auc"]

    for result in fold_results
)


print("\n" + "=" * 90)
print("FOLD WIN COUNTS FOR V2")
print("=" * 90)

print(
    f"Better Log Loss: "
    f"{v2_logloss_wins}/{N_FOLDS} folds"
)

print(
    f"Better Brier: "
    f"{v2_brier_wins}/{N_FOLDS} folds"
)

print(
    f"Better ROC-AUC: "
    f"{v2_auc_wins}/{N_FOLDS} folds"
)


# ==========================================================
# SIMPLE DECISION
# ==========================================================

print("\n" + "=" * 90)
print("SUMMARY")
print("=" * 90)


if (
    delta_logloss < 0
    and
    delta_brier < 0
    and
    delta_auc > 0
):

    print(
        "\nV2 improved all three main metrics "
        "under identical cross-validation folds."
    )

elif (
    delta_logloss < 0
    and
    delta_brier < 0
):

    print(
        "\nV2 improved probability accuracy, "
        "but discrimination should also be reviewed."
    )

else:

    print(
        "\nV2 does not show a clear improvement "
        "over V1 under identical folds."
    )


print("\nIMPORTANT:")
print(
    "This script does NOT modify or overwrite "
    "either saved model."
)

print("\nV1 remains:")
print(
    PROJECT_ROOT
    / "models"
    / "xg_multicompetition_logistic.json"
)

print("\nV2 remains:")
print(
    PROJECT_ROOT
    / "models"
    / "xg_multicompetition_logistic_v2.json"
)

print("\n" + "=" * 90)
print("COMPARISON COMPLETE")
print("=" * 90)