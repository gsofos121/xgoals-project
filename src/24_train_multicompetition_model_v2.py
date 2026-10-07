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
    / "xg_multicompetition_logistic_v2.json"
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

print("=" * 80)
print("MULTI-COMPETITION xG MODEL V2")
print("=" * 80)

print(f"\nLoading:\n{INPUT_FILE}\n")

df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded: {len(df):,}")
print(f"Columns: {len(df.columns):,}")


# ==========================================================
# CREATE / DETECT TARGET
# ==========================================================

print("\n" + "=" * 80)
print("TARGET")
print("=" * 80)


if "is_goal" in df.columns:

    print("\nUsing existing column: is_goal")

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
        "\nCreating is_goal from: shot_outcome"
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
        "\nERROR:"
        "\nNeither is_goal nor shot_outcome exists."
    )


print(
    f"Goals detected: "
    f"{int(df['is_goal'].sum()):,}"
)


# ==========================================================
# CHECK REQUIRED COLUMNS
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

    print("\n" + "=" * 80)
    print("MISSING REQUIRED COLUMNS")
    print("=" * 80)

    for column in missing_columns:
        print(column)

    print("\nAvailable columns:\n")

    for column in df.columns:
        print(column)

    raise SystemExit(
        "\nTraining stopped because required columns are missing."
    )


# ==========================================================
# BOOLEAN CONVERTER
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
# BUILD FEATURE TABLE
# ==========================================================

features = pd.DataFrame(
    index=df.index
)


# ----------------------------------------------------------
# GEOMETRY
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
# EXISTING V1 CONDITIONS
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
# BODY PART
# Right Foot remains the reference category.
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
# SHOT TYPE
# Open Play remains the reference category.
# Penalties are handled separately.
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


# ==========================================================
# NEW V2 FEATURES — PLAY PATTERN
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
# FEATURE LIST
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
# PRINT FEATURE COUNTS
# ==========================================================

print("\n" + "=" * 80)
print("V2 PLAY PATTERN FEATURE COUNTS")
print("=" * 80)

print(
    f"Regular Play: "
    f"{int(features['play_regular'].sum()):,}"
)

print(
    f"From Counter: "
    f"{int(features['play_counter'].sum()):,}"
)

print(
    f"From Corner: "
    f"{int(features['play_corner'].sum()):,}"
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
# REMOVE INVALID FEATURE ROWS
# ==========================================================

valid_feature_mask = (
    non_penalty_features
    .notna()
    .all(axis=1)
)

invalid_feature_rows = (
    ~valid_feature_mask
).sum()


if invalid_feature_rows > 0:

    print(
        "\nInvalid non-penalty rows removed: "
        f"{invalid_feature_rows:,}"
    )


non_penalty_features = (
    non_penalty_features[
        valid_feature_mask
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
# DATA SUMMARY
# ==========================================================

print("\n" + "=" * 80)
print("DATA SUMMARY")
print("=" * 80)

print(
    f"Total rows: "
    f"{len(df):,}"
)

print(
    f"Total goals: "
    f"{int(df['is_goal'].sum()):,}"
)

print(
    f"Penalty rows: "
    f"{len(penalty_df):,}"
)

print(
    f"Non-penalty rows used: "
    f"{len(non_penalty_df):,}"
)

print(
    f"Matches: "
    f"{df['match_id'].nunique():,}"
)


# ==========================================================
# FIXED PENALTY xG
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


if penalty_shots > 0:

    penalty_xg = (
        penalty_goals
        / penalty_shots
    )

else:

    penalty_xg = 0.0


print("\n" + "=" * 80)
print("PENALTY MODEL")
print("=" * 80)

print(
    f"Penalty shots: "
    f"{penalty_shots:,}"
)

print(
    f"Penalty goals: "
    f"{penalty_goals:,}"
)

print(
    f"Penalty conversion: "
    f"{penalty_xg:.4f}"
)

print(
    f"Fixed penalty xG: "
    f"{penalty_xg:.4f}"
)


# ==========================================================
# NUMPY ARRAYS
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
        X.mean(axis=0)
    )

    std = (
        X.std(axis=0)
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
# LOGISTIC FUNCTIONS
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
    l2=L2,
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
            X.T @ error
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
            n_positive
            + 1
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
# GROUPED 5-FOLD CV
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

fold_match_groups = (
    np.array_split(
        unique_matches,
        N_FOLDS
    )
)

oof_probabilities = np.zeros(
    len(y_all),
    dtype=float
)


print("\n" + "=" * 80)
print("5-FOLD MATCH-LEVEL CROSS-VALIDATION")
print("=" * 80)


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
            scaler_std,
        )
    )

    X_validation = (
        transform_scaler(
            X_validation_raw,
            scaler_mean,
            scaler_std,
        )
    )


    weights, intercept = (
        train_logistic(
            X_train,
            y_train,
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
            validation_probabilities,
        )
    )

    fold_brier = (
        brier_score(
            y_validation,
            validation_probabilities,
        )
    )

    fold_auc = (
        roc_auc(
            y_validation,
            validation_probabilities,
        )
    )


    print(
        f"\nFold {fold_number}"
    )

    print(
        f"Training matches: "
        f"{len(unique_matches) - len(validation_matches):,}"
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
# NON-PENALTY OOF
# ==========================================================

non_penalty_log_loss = (
    log_loss(
        y_all,
        oof_probabilities,
    )
)

non_penalty_brier = (
    brier_score(
        y_all,
        oof_probabilities,
    )
)

non_penalty_auc = (
    roc_auc(
        y_all,
        oof_probabilities,
    )
)


print("\n" + "=" * 80)
print("OOF NON-PENALTY RESULTS")
print("=" * 80)

print(
    f"Log Loss: "
    f"{non_penalty_log_loss:.4f}"
)

print(
    f"Brier: "
    f"{non_penalty_brier:.4f}"
)

print(
    f"ROC-AUC: "
    f"{non_penalty_auc:.4f}"
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
        full_probabilities,
    )
)

full_brier = (
    brier_score(
        full_y,
        full_probabilities,
    )
)

full_auc = (
    roc_auc(
        full_y,
        full_probabilities,
    )
)


print("\n" + "=" * 80)
print("FULL OOF RESULTS INCLUDING PENALTIES")
print("=" * 80)

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
# TRAIN FINAL V2 ON ALL NON-PENALTY DATA
# ==========================================================

print("\n" + "=" * 80)
print("TRAINING FINAL V2 MODEL")
print("=" * 80)


final_mean, final_std = (
    fit_scaler(
        X_all
    )
)

X_all_scaled = (
    transform_scaler(
        X_all,
        final_mean,
        final_std,
    )
)


final_weights_scaled, final_intercept_scaled = (
    train_logistic(
        X_all_scaled,
        y_all,
    )
)


# ==========================================================
# CONVERT TO RAW SCALE
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
# FINAL COEFFICIENTS
# ==========================================================

print("\nFinal raw-scale coefficients:\n")


coefficients = {}


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
        f"{coefficient:+.6f}"
    )


print(
    f"{'intercept':<24}"
    f"{raw_intercept:+.6f}"
)

print(
    f"{'fixed_penalty_xg':<24}"
    f"{penalty_xg:.6f}"
)


# ==========================================================
# PLAY PATTERN COEFFICIENT SUMMARY
# ==========================================================

print("\n" + "=" * 80)
print("NEW V2 PLAY-PATTERN COEFFICIENTS")
print("=" * 80)

print(
    f"Regular Play: "
    f"{coefficients['play_regular']:+.6f}"
)

print(
    f"Counterattack: "
    f"{coefficients['play_counter']:+.6f}"
)

print(
    f"Corner: "
    f"{coefficients['play_corner']:+.6f}"
)


# ==========================================================
# SAVE V2
# ==========================================================

model_payload = {

    "model_name":
        "multi_competition_xg_v2",

    "model_type":
        "logistic_regression_with_separate_penalty_probability",

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
    },
}


OUTPUT_MODEL.parent.mkdir(
    parents=True,
    exist_ok=True
)


with open(
    OUTPUT_MODEL,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        model_payload,
        file,
        indent=2,
        ensure_ascii=False,
    )


# ==========================================================
# FINISHED
# ==========================================================

print("\n" + "=" * 80)
print("MODEL SAVED")
print("=" * 80)

print(
    f"\nV2:\n{OUTPUT_MODEL}"
)

print(
    "\nV1 WAS NOT MODIFIED:"
)

print(
    PROJECT_ROOT
    / "models"
    / "xg_multicompetition_logistic.json"
)

print("\n" + "=" * 80)
print("V2 TRAINING COMPLETE")
print("=" * 80)