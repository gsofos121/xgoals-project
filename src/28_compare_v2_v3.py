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


V3_FEATURES = (
    V2_FEATURES
    +
    V3_EXTRA_FEATURES
)


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
# MANUAL LOGISTIC REGRESSION
# ==========================================================

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


        probabilities = sigmoid(
            linear_score
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

        return float(
            "nan"
        )


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


    while (
        i
        <
        len(scores)
    ):

        j = (
            i + 1
        )


        while (
            j
            <
            len(scores)
            and
            sorted_scores[j]
            ==
            sorted_scores[i]
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
    y,
    predictions
):

    return {

        "log_loss":
            log_loss(
                y,
                predictions
            ),

        "brier":
            brier_score(
                y,
                predictions
            ),

        "auc":
            roc_auc(
                y,
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
print("DIRECT V2 vs V3 — SAME-FOLD COMPARISON")
print("=" * 95)

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

if (
    "is_goal"
    in df.columns
):

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


elif (
    "shot_outcome"
    in df.columns
):

    df[
        "is_goal"
    ] = (
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

    print(
        "\nMissing columns:"
    )


    for column in missing_columns:

        print(
            f"  {column}"
        )


    raise SystemExit(
        "\nComparison stopped."
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
# Context
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
# V3 EXTRA FEATURES
# ==========================================================

features[
    "distance_sq"
] = (
    features[
        "distance_m"
    ]
    ** 2
)


features[
    "angle_sq"
] = (
    features[
        "angle_deg"
    ]
    ** 2
)


features[
    "distance_angle"
] = (
    features[
        "distance_m"
    ]
    *
    features[
        "angle_deg"
    ]
)


features[
    "distance_head"
] = (
    features[
        "distance_m"
    ]
    *
    features[
        "body_head"
    ]
)


features[
    "counter_one_on_one"
] = (
    features[
        "play_counter"
    ]
    *
    features[
        "shot_one_on_one"
    ]
)


features[
    "corner_head"
] = (
    features[
        "play_corner"
    ]
    *
    features[
        "body_head"
    ]
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


penalty_shots = int(
    is_penalty.sum()
)


penalty_goals = int(
    df.loc[
        is_penalty,
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
# NON-PENALTY DATA
# ==========================================================

non_penalty_mask = (
    ~is_penalty
)


required_feature_columns = list(
    dict.fromkeys(
        V2_FEATURES
        +
        V3_FEATURES
    )
)


valid_feature_mask = (
    features[
        required_feature_columns
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


if (
    invalid_non_penalty_rows
    >
    0
):

    print(
        f"\nInvalid non-penalty rows removed: "
        f"{invalid_non_penalty_rows:,}"
    )


comparison_df = (
    df.loc[
        usable_non_penalty_mask
    ]
    .copy()
)


comparison_features = (
    features.loc[
        usable_non_penalty_mask
    ]
    .copy()
)


# ==========================================================
# ARRAYS
# ==========================================================

X_v2_all = (
    comparison_features[
        V2_FEATURES
    ]
    .astype(float)
    .to_numpy()
)


X_v3_all = (
    comparison_features[
        V3_FEATURES
    ]
    .astype(float)
    .to_numpy()
)


y_non_penalty = (
    comparison_df[
        "is_goal"
    ]
    .astype(float)
    .to_numpy()
)


match_ids = (
    comparison_df[
        "match_id"
    ]
    .astype(str)
    .to_numpy()
)


# ==========================================================
# SUMMARY
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
    f"Non-penalty shots used: "
    f"{len(comparison_df):,}"
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


print(
    f"\nV2 features: "
    f"{len(V2_FEATURES)}"
)


print(
    f"V3 features: "
    f"{len(V3_FEATURES)}"
)


# ==========================================================
# CREATE EXACT SAME FOLDS
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
# OOF ARRAYS
# ==========================================================

oof_v2_non_penalty = np.zeros(
    len(
        y_non_penalty
    ),
    dtype=float
)


oof_v3_non_penalty = np.zeros(
    len(
        y_non_penalty
    ),
    dtype=float
)


fold_results = []


wins = {

    "log_loss_v2":
        0,

    "log_loss_v3":
        0,

    "brier_v2":
        0,

    "brier_v3":
        0,

    "auc_v2":
        0,

    "auc_v3":
        0,
}


# ==========================================================
# CROSS-VALIDATION
# ==========================================================

print("\n" + "=" * 95)
print("5-FOLD DIRECT V2 vs V3 COMPARISON")
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
            for match_id
            in match_ids
        ],
        dtype=bool
    )


    training_mask = (
        ~validation_mask
    )


    # ======================================================
    # V2
    # ======================================================

    X_v2_train_raw = (
        X_v2_all[
            training_mask
        ]
    )


    X_v2_validation_raw = (
        X_v2_all[
            validation_mask
        ]
    )


    v2_mean, v2_std = fit_scaler(
        X_v2_train_raw
    )


    X_v2_train = transform_scaler(
        X_v2_train_raw,
        v2_mean,
        v2_std
    )


    X_v2_validation = transform_scaler(
        X_v2_validation_raw,
        v2_mean,
        v2_std
    )


    v2_weights, v2_intercept = train_logistic(
        X_v2_train,
        y_non_penalty[
            training_mask
        ]
    )


    v2_predictions = sigmoid(
        X_v2_validation
        @ v2_weights
        +
        v2_intercept
    )


    oof_v2_non_penalty[
        validation_mask
    ] = (
        v2_predictions
    )


    # ======================================================
    # V3
    # ======================================================

    X_v3_train_raw = (
        X_v3_all[
            training_mask
        ]
    )


    X_v3_validation_raw = (
        X_v3_all[
            validation_mask
        ]
    )


    v3_mean, v3_std = fit_scaler(
        X_v3_train_raw
    )


    X_v3_train = transform_scaler(
        X_v3_train_raw,
        v3_mean,
        v3_std
    )


    X_v3_validation = transform_scaler(
        X_v3_validation_raw,
        v3_mean,
        v3_std
    )


    v3_weights, v3_intercept = train_logistic(
        X_v3_train,
        y_non_penalty[
            training_mask
        ]
    )


    v3_predictions = sigmoid(
        X_v3_validation
        @ v3_weights
        +
        v3_intercept
    )


    oof_v3_non_penalty[
        validation_mask
    ] = (
        v3_predictions
    )


    # ======================================================
    # FOLD METRICS
    # ======================================================

    y_validation = (
        y_non_penalty[
            validation_mask
        ]
    )


    v2_metrics = calculate_metrics(
        y_validation,
        v2_predictions
    )


    v3_metrics = calculate_metrics(
        y_validation,
        v3_predictions
    )


    delta_log_loss = (
        v3_metrics[
            "log_loss"
        ]
        -
        v2_metrics[
            "log_loss"
        ]
    )


    delta_brier = (
        v3_metrics[
            "brier"
        ]
        -
        v2_metrics[
            "brier"
        ]
    )


    delta_auc = (
        v3_metrics[
            "auc"
        ]
        -
        v2_metrics[
            "auc"
        ]
    )


    # ------------------------------------------------------
    # Count wins
    # ------------------------------------------------------

    if (
        v3_metrics[
            "log_loss"
        ]
        <
        v2_metrics[
            "log_loss"
        ]
    ):

        wins[
            "log_loss_v3"
        ] += 1

    else:

        wins[
            "log_loss_v2"
        ] += 1


    if (
        v3_metrics[
            "brier"
        ]
        <
        v2_metrics[
            "brier"
        ]
    ):

        wins[
            "brier_v3"
        ] += 1

    else:

        wins[
            "brier_v2"
        ] += 1


    if (
        v3_metrics[
            "auc"
        ]
        >
        v2_metrics[
            "auc"
        ]
    ):

        wins[
            "auc_v3"
        ] += 1

    else:

        wins[
            "auc_v2"
        ] += 1


    fold_results.append(
        {
            "fold":
                fold_number,

            "shots":
                int(
                    validation_mask.sum()
                ),

            "goals":
                int(
                    y_validation.sum()
                ),

            "v2":
                v2_metrics,

            "v3":
                v3_metrics,

            "delta_log_loss":
                delta_log_loss,

            "delta_brier":
                delta_brier,

            "delta_auc":
                delta_auc,
        }
    )


    # ======================================================
    # PRINT FOLD
    # ======================================================

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
        "\n"
        f"{'Metric':<18}"
        f"{'V2':>14}"
        f"{'V3':>14}"
        f"{'V3 - V2':>16}"
    )


    print(
        f"{'Log Loss':<18}"
        f"{v2_metrics['log_loss']:>14.6f}"
        f"{v3_metrics['log_loss']:>14.6f}"
        f"{delta_log_loss:>+16.6f}"
    )


    print(
        f"{'Brier':<18}"
        f"{v2_metrics['brier']:>14.6f}"
        f"{v3_metrics['brier']:>14.6f}"
        f"{delta_brier:>+16.6f}"
    )


    print(
        f"{'ROC-AUC':<18}"
        f"{v2_metrics['auc']:>14.6f}"
        f"{v3_metrics['auc']:>14.6f}"
        f"{delta_auc:>+16.6f}"
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


full_v2 = np.zeros(
    len(df),
    dtype=float
)


full_v3 = np.zeros(
    len(df),
    dtype=float
)


non_penalty_indices = (
    comparison_df
    .index
    .to_numpy()
)


penalty_indices = (
    df.index[
        is_penalty
    ]
    .to_numpy()
)


full_v2[
    non_penalty_indices
] = (
    oof_v2_non_penalty
)


full_v3[
    non_penalty_indices
] = (
    oof_v3_non_penalty
)


full_v2[
    penalty_indices
] = (
    penalty_xg
)


full_v3[
    penalty_indices
] = (
    penalty_xg
)


# ==========================================================
# CHECK COMPLETE COVERAGE
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

    missing = (
        expected_indices
        -
        covered_indices
    )


    raise SystemExit(
        "\nERROR: "
        f"{len(missing)} rows were not assigned predictions."
    )


# ==========================================================
# OVERALL METRICS
# ==========================================================

overall_v2 = calculate_metrics(
    full_y,
    full_v2
)


overall_v3 = calculate_metrics(
    full_y,
    full_v3
)


overall_delta_log_loss = (
    overall_v3[
        "log_loss"
    ]
    -
    overall_v2[
        "log_loss"
    ]
)


overall_delta_brier = (
    overall_v3[
        "brier"
    ]
    -
    overall_v2[
        "brier"
    ]
)


overall_delta_auc = (
    overall_v3[
        "auc"
    ]
    -
    overall_v2[
        "auc"
    ]
)


# ==========================================================
# OVERALL OUTPUT
# ==========================================================

print("\n" + "=" * 95)
print("OVERALL OOF — SAME FOLDS")
print("=" * 95)


print(
    f"\n{'Metric':<20}"
    f"{'V2':>15}"
    f"{'V3':>15}"
    f"{'V3 - V2':>17}"
)


print(
    f"{'Log Loss':<20}"
    f"{overall_v2['log_loss']:>15.6f}"
    f"{overall_v3['log_loss']:>15.6f}"
    f"{overall_delta_log_loss:>+17.6f}"
)


print(
    f"{'Brier':<20}"
    f"{overall_v2['brier']:>15.6f}"
    f"{overall_v3['brier']:>15.6f}"
    f"{overall_delta_brier:>+17.6f}"
)


print(
    f"{'ROC-AUC':<20}"
    f"{overall_v2['auc']:>15.6f}"
    f"{overall_v3['auc']:>15.6f}"
    f"{overall_delta_auc:>+17.6f}"
)


print(
    f"{'Predicted goals':<20}"
    f"{overall_v2['predicted_goals']:>15.2f}"
    f"{overall_v3['predicted_goals']:>15.2f}"
    f"{(
        overall_v3['predicted_goals']
        -
        overall_v2['predicted_goals']
    ):>+17.2f}"
)


print(
    f"\nActual goals: "
    f"{int(full_y.sum()):,}"
)


# ==========================================================
# FOLD WINS
# ==========================================================

print("\n" + "=" * 95)
print("FOLD WINS")
print("=" * 95)


print(
    f"\nLog Loss:"
    f"\n  V2 wins: "
    f"{wins['log_loss_v2']}/5"
    f"\n  V3 wins: "
    f"{wins['log_loss_v3']}/5"
)


print(
    f"\nBrier:"
    f"\n  V2 wins: "
    f"{wins['brier_v2']}/5"
    f"\n  V3 wins: "
    f"{wins['brier_v3']}/5"
)


print(
    f"\nROC-AUC:"
    f"\n  V2 wins: "
    f"{wins['auc_v2']}/5"
    f"\n  V3 wins: "
    f"{wins['auc_v3']}/5"
)


# ==========================================================
# CALIBRATION
# ==========================================================

def calibration_table(
    y,
    predictions
):

    temp = pd.DataFrame(
        {
            "actual":
                y,

            "prediction":
                predictions,
        }
    )


    temp[
        "bin"
    ] = pd.qcut(
        temp[
            "prediction"
        ],
        q=10,
        duplicates="drop"
    )


    summary = (
        temp
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


    return summary


print("\n" + "=" * 95)
print("V2 OOF CALIBRATION DECILES")
print("=" * 95)


v2_calibration = calibration_table(
    full_y,
    full_v2
)


print(
    v2_calibration.to_string(
        index=False,
        formatters={
            "predicted":
                lambda x:
                    f"{x:.3f}",

            "actual":
                lambda x:
                    f"{x:.3f}",
        }
    )
)


print("\n" + "=" * 95)
print("V3 OOF CALIBRATION DECILES")
print("=" * 95)


v3_calibration = calibration_table(
    full_y,
    full_v3
)


print(
    v3_calibration.to_string(
        index=False,
        formatters={
            "predicted":
                lambda x:
                    f"{x:.3f}",

            "actual":
                lambda x:
                    f"{x:.3f}",
        }
    )
)


# ==========================================================
# FINAL DECISION
# ==========================================================

print("\n" + "=" * 95)
print("FINAL SAME-FOLD COMPARISON")
print("=" * 95)


main_metric_improvements = 0


if (
    overall_delta_log_loss
    <
    0
):

    main_metric_improvements += 1


if (
    overall_delta_brier
    <
    0
):

    main_metric_improvements += 1


if (
    overall_delta_auc
    >
    0
):

    main_metric_improvements += 1


if (
    main_metric_improvements
    ==
    3
):

    print(
        "\nV3 improved Log Loss, Brier "
        "and ROC-AUC relative to V2."
    )


elif (
    main_metric_improvements
    ==
    2
):

    print(
        "\nV3 improved two of the three "
        "main metrics relative to V2."
    )


elif (
    main_metric_improvements
    ==
    1
):

    print(
        "\nV3 improved only one of the "
        "three main metrics relative to V2."
    )


else:

    print(
        "\nV3 did not improve any of the "
        "three main metrics relative to V2."
    )


print(
    "\nNo model files were modified."
)


print("\n" + "=" * 95)
print("COMPARISON COMPLETE")
print("=" * 95)