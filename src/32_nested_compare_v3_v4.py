from pathlib import Path

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


# ==========================================================
# GLOBAL SETTINGS
# ==========================================================

OUTER_FOLDS = 5

INNER_VALIDATION_FRACTION = 0.20

RANDOM_SEED = 42

EPSILON = 1e-15


# ==========================================================
# V3 LOGISTIC SETTINGS
# ==========================================================

LOGISTIC_MAX_ITER = 15000
LOGISTIC_LEARNING_RATE = 0.03
LOGISTIC_L2 = 0.001


# ==========================================================
# V4 XGBOOST SETTINGS
# ==========================================================

MAX_BOOST_ROUNDS = 2000
EARLY_STOPPING_ROUNDS = 100


XGB_PARAMS = {

    "objective":
        "binary:logistic",

    "eval_metric":
        "logloss",

    "max_depth":
        3,

    "eta":
        0.03,

    "subsample":
        0.80,

    "colsample_bytree":
        0.80,

    "min_child_weight":
        5,

    "lambda":
        1.0,

    "alpha":
        0.0,

    "gamma":
        0.0,

    "seed":
        RANDOM_SEED,

    "device":
        "cpu",

    "tree_method":
        "hist",

    "verbosity":
        0,
}


# ==========================================================
# V3 FEATURES
# ==========================================================

V3_BASE_FEATURES = [

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
    V3_BASE_FEATURES
    +
    V3_EXTRA_FEATURES
)


# ==========================================================
# V4 FEATURES
#
# XGBoost uses only the 12 original features.
# ==========================================================

V4_FEATURES = [

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
# BASIC HELPERS
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

        "brier":
            brier_score(
                y_true,
                predictions
            ),

        "auc":
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
# V3 SCALING
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
# V3 MANUAL LOGISTIC REGRESSION
# ==========================================================

def train_logistic(
    X,
    y
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
        LOGISTIC_MAX_ITER
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
            LOGISTIC_L2
            * weights
        )


        gradient_intercept = (
            error.mean()
        )


        weights -= (
            LOGISTIC_LEARNING_RATE
            * gradient_weights
        )


        intercept -= (
            LOGISTIC_LEARNING_RATE
            * gradient_intercept
        )


    return (
        weights,
        intercept
    )


# ==========================================================
# LOAD DATA
# ==========================================================

print("=" * 105)
print("NESTED-STYLE OUTER CV — V3 ENHANCED LOGISTIC vs V4 XGBOOST")
print("=" * 105)


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
        "\nERROR: target column not found."
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
        "\nMissing required columns:"
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
# Conditions
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
# V3 ENGINEERED FEATURES
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
# VALID ROWS
# ==========================================================

all_features = list(
    dict.fromkeys(
        V3_FEATURES
        +
        V4_FEATURES
    )
)


valid_mask = (
    features[
        all_features
    ]
    .notna()
    .all(
        axis=1
    )
)


invalid_rows = int(
    (
        ~valid_mask
    )
    .sum()
)


if invalid_rows > 0:

    print(
        f"\nRows removed because of missing features: "
        f"{invalid_rows:,}"
    )


df = (
    df.loc[
        valid_mask
    ]
    .copy()
    .reset_index(
        drop=True
    )
)


features = (
    features.loc[
        valid_mask
    ]
    .copy()
    .reset_index(
        drop=True
    )
)


# ==========================================================
# BASIC SUMMARY
# ==========================================================

print("\n" + "=" * 105)
print("DATA SUMMARY")
print("=" * 105)


print(
    f"Shots: "
    f"{len(df):,}"
)


print(
    f"Goals: "
    f"{int(df['is_goal'].sum()):,}"
)


print(
    f"Matches: "
    f"{df['match_id'].nunique():,}"
)


print(
    f"V3 features: "
    f"{len(V3_FEATURES)}"
)


print(
    f"V4 features: "
    f"{len(V4_FEATURES)}"
)


# ==========================================================
# MATCH IDs
#
# IMPORTANT:
# Folds are created using ALL matches, not only
# non-penalty shots.
# ==========================================================

all_match_ids = (
    df[
        "match_id"
    ]
    .astype(str)
    .to_numpy()
)


unique_matches = np.array(
    sorted(
        pd.unique(
            all_match_ids
        )
    )
)


rng = np.random.default_rng(
    RANDOM_SEED
)


rng.shuffle(
    unique_matches
)


outer_fold_match_groups = np.array_split(
    unique_matches,
    OUTER_FOLDS
)


# ==========================================================
# OOF ARRAYS
# ==========================================================

y_all = (
    df[
        "is_goal"
    ]
    .astype(float)
    .to_numpy()
)


oof_v3 = np.zeros(
    len(df),
    dtype=float
)


oof_v4 = np.zeros(
    len(df),
    dtype=float
)


fold_results = []


wins = {

    "log_loss_v3":
        0,

    "log_loss_v4":
        0,

    "brier_v3":
        0,

    "brier_v4":
        0,

    "auc_v3":
        0,

    "auc_v4":
        0,
}


selected_boost_rounds = []


# ==========================================================
# OUTER CROSS-VALIDATION
# ==========================================================

print("\n" + "=" * 105)
print("5-FOLD OUTER VALIDATION")
print("=" * 105)


for (
    outer_fold,
    outer_validation_matches
) in enumerate(
    outer_fold_match_groups,
    start=1
):


    # ======================================================
    # OUTER SPLIT
    # ======================================================

    outer_validation_match_set = set(
        outer_validation_matches
    )


    outer_validation_mask = np.array(
        [
            match_id
            in outer_validation_match_set

            for match_id
            in all_match_ids
        ],
        dtype=bool
    )


    outer_training_mask = (
        ~outer_validation_mask
    )


    # ======================================================
    # OUTER TRAIN / VALID DATA
    # ======================================================

    outer_train_df = (
        df.loc[
            outer_training_mask
        ]
        .copy()
    )


    outer_valid_df = (
        df.loc[
            outer_validation_mask
        ]
        .copy()
    )


    outer_train_features = (
        features.loc[
            outer_training_mask
        ]
        .copy()
    )


    outer_valid_features = (
        features.loc[
            outer_validation_mask
        ]
        .copy()
    )


    # ======================================================
    # PENALTY xG
    #
    # STRICT:
    # Calculated ONLY from outer training data.
    # ======================================================

    outer_train_shot_type = (
        outer_train_df[
            "shot_type"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
    )


    outer_valid_shot_type = (
        outer_valid_df[
            "shot_type"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
    )


    train_penalty_mask = (
        outer_train_shot_type
        .eq(
            "Penalty"
        )
        .to_numpy()
    )


    valid_penalty_mask = (
        outer_valid_shot_type
        .eq(
            "Penalty"
        )
        .to_numpy()
    )


    train_penalty_count = int(
        train_penalty_mask.sum()
    )


    train_penalty_goals = int(
        outer_train_df[
            "is_goal"
        ]
        .to_numpy()[
            train_penalty_mask
        ]
        .sum()
    )


    if train_penalty_count == 0:

        raise SystemExit(
            "\nERROR: no training penalties "
            f"in outer fold {outer_fold}."
        )


    outer_penalty_xg = (
        train_penalty_goals
        /
        train_penalty_count
    )


    # ======================================================
    # NON-PENALTY MASKS
    # ======================================================

    train_non_penalty_mask = (
        ~train_penalty_mask
    )


    valid_non_penalty_mask = (
        ~valid_penalty_mask
    )


    # ======================================================
    # OUTER TARGETS
    # ======================================================

    y_outer_train = (
        outer_train_df[
            "is_goal"
        ]
        .astype(float)
        .to_numpy()
    )


    y_outer_valid = (
        outer_valid_df[
            "is_goal"
        ]
        .astype(float)
        .to_numpy()
    )


    y_outer_train_np = (
        y_outer_train[
            train_non_penalty_mask
        ]
    )


    # ======================================================
    # ======================================================
    # V3 — ENHANCED LOGISTIC
    # ======================================================
    # ======================================================

    X_v3_train_raw = (
        outer_train_features[
            V3_FEATURES
        ]
        .astype(float)
        .to_numpy()[
            train_non_penalty_mask
        ]
    )


    X_v3_valid_raw = (
        outer_valid_features[
            V3_FEATURES
        ]
        .astype(float)
        .to_numpy()[
            valid_non_penalty_mask
        ]
    )


    # ------------------------------------------------------
    # Scaling learned only on outer training
    # ------------------------------------------------------

    v3_mean, v3_std = fit_scaler(
        X_v3_train_raw
    )


    X_v3_train = transform_scaler(
        X_v3_train_raw,
        v3_mean,
        v3_std
    )


    X_v3_valid = transform_scaler(
        X_v3_valid_raw,
        v3_mean,
        v3_std
    )


    # ------------------------------------------------------
    # Train V3
    # ------------------------------------------------------

    v3_weights, v3_intercept = train_logistic(
        X_v3_train,
        y_outer_train_np
    )


    # ------------------------------------------------------
    # Predict outer validation
    # ------------------------------------------------------

    v3_non_penalty_predictions = sigmoid(
        X_v3_valid
        @ v3_weights
        +
        v3_intercept
    )


    v3_outer_predictions = np.zeros(
        len(
            outer_valid_df
        ),
        dtype=float
    )


    v3_outer_predictions[
        valid_non_penalty_mask
    ] = (
        v3_non_penalty_predictions
    )


    v3_outer_predictions[
        valid_penalty_mask
    ] = (
        outer_penalty_xg
    )


    # ======================================================
    # ======================================================
    # V4 — XGBOOST
    # ======================================================
    # ======================================================


    # ======================================================
    # INNER MATCH SPLIT
    #
    # ONLY matches from OUTER TRAINING are available here.
    # Outer validation remains completely untouched.
    # ======================================================

    outer_training_matches = np.array(
        sorted(
            outer_train_df[
                "match_id"
            ]
            .astype(str)
            .unique()
        )
    )


    inner_rng = np.random.default_rng(
        RANDOM_SEED
        +
        outer_fold
    )


    inner_rng.shuffle(
        outer_training_matches
    )


    inner_validation_count = max(
        1,
        int(
            round(
                len(
                    outer_training_matches
                )
                *
                INNER_VALIDATION_FRACTION
            )
        )
    )


    inner_validation_matches = (
        outer_training_matches[
            :inner_validation_count
        ]
    )


    inner_validation_match_set = set(
        inner_validation_matches
    )


    outer_train_match_ids = (
        outer_train_df[
            "match_id"
        ]
        .astype(str)
        .to_numpy()
    )


    inner_validation_mask = np.array(
        [
            match_id
            in inner_validation_match_set

            for match_id
            in outer_train_match_ids
        ],
        dtype=bool
    )


    inner_training_mask = (
        ~inner_validation_mask
    )


    # ------------------------------------------------------
    # Inner masks must also exclude penalties
    # ------------------------------------------------------

    inner_train_non_penalty_mask = (
        inner_training_mask
        &
        train_non_penalty_mask
    )


    inner_valid_non_penalty_mask = (
        inner_validation_mask
        &
        train_non_penalty_mask
    )


    # ======================================================
    # INNER XGBOOST DATA
    # ======================================================

    X_v4_outer_train_all = (
        outer_train_features[
            V4_FEATURES
        ]
        .astype(float)
        .to_numpy()
    )


    X_inner_train = (
        X_v4_outer_train_all[
            inner_train_non_penalty_mask
        ]
    )


    y_inner_train = (
        y_outer_train[
            inner_train_non_penalty_mask
        ]
    )


    X_inner_valid = (
        X_v4_outer_train_all[
            inner_valid_non_penalty_mask
        ]
    )


    y_inner_valid = (
        y_outer_train[
            inner_valid_non_penalty_mask
        ]
    )


    d_inner_train = xgb.DMatrix(
        X_inner_train,
        label=
            y_inner_train,
        feature_names=
            V4_FEATURES
    )


    d_inner_valid = xgb.DMatrix(
        X_inner_valid,
        label=
            y_inner_valid,
        feature_names=
            V4_FEATURES
    )


    # ======================================================
    # INNER EARLY STOPPING
    # ======================================================

    inner_booster = xgb.train(

        params=
            XGB_PARAMS,

        dtrain=
            d_inner_train,

        num_boost_round=
            MAX_BOOST_ROUNDS,

        evals=[
            (
                d_inner_train,
                "inner_train"
            ),
            (
                d_inner_valid,
                "inner_validation"
            ),
        ],

        early_stopping_rounds=
            EARLY_STOPPING_ROUNDS,

        verbose_eval=
            False,
    )


    best_boost_rounds = (
        int(
            inner_booster.best_iteration
        )
        + 1
    )


    selected_boost_rounds.append(
        best_boost_rounds
    )


    # ======================================================
    # RETRAIN V4 ON ALL OUTER TRAINING NON-PENALTY DATA
    #
    # Now inner validation rejoins training.
    # Number of rounds is already fixed.
    # ======================================================

    X_v4_outer_train = (
        X_v4_outer_train_all[
            train_non_penalty_mask
        ]
    )


    X_v4_outer_valid = (
        outer_valid_features[
            V4_FEATURES
        ]
        .astype(float)
        .to_numpy()[
            valid_non_penalty_mask
        ]
    )


    d_v4_outer_train = xgb.DMatrix(
        X_v4_outer_train,
        label=
            y_outer_train_np,
        feature_names=
            V4_FEATURES
    )


    d_v4_outer_valid = xgb.DMatrix(
        X_v4_outer_valid,
        feature_names=
            V4_FEATURES
    )


    outer_v4_booster = xgb.train(

        params=
            XGB_PARAMS,

        dtrain=
            d_v4_outer_train,

        num_boost_round=
            best_boost_rounds,

        evals=[
            (
                d_v4_outer_train,
                "outer_train"
            )
        ],

        verbose_eval=
            False,
    )


    # ======================================================
    # V4 OUTER VALIDATION PREDICTIONS
    # ======================================================

    v4_non_penalty_predictions = (
        outer_v4_booster.predict(
            d_v4_outer_valid
        )
    )


    v4_outer_predictions = np.zeros(
        len(
            outer_valid_df
        ),
        dtype=float
    )


    v4_outer_predictions[
        valid_non_penalty_mask
    ] = (
        v4_non_penalty_predictions
    )


    v4_outer_predictions[
        valid_penalty_mask
    ] = (
        outer_penalty_xg
    )


    # ======================================================
    # OUTER METRICS
    # ======================================================

    v3_metrics = calculate_metrics(
        y_outer_valid,
        v3_outer_predictions
    )


    v4_metrics = calculate_metrics(
        y_outer_valid,
        v4_outer_predictions
    )


    delta_log_loss = (
        v4_metrics[
            "log_loss"
        ]
        -
        v3_metrics[
            "log_loss"
        ]
    )


    delta_brier = (
        v4_metrics[
            "brier"
        ]
        -
        v3_metrics[
            "brier"
        ]
    )


    delta_auc = (
        v4_metrics[
            "auc"
        ]
        -
        v3_metrics[
            "auc"
        ]
    )


    # ======================================================
    # WINS
    # ======================================================

    if (
        v4_metrics[
            "log_loss"
        ]
        <
        v3_metrics[
            "log_loss"
        ]
    ):

        wins[
            "log_loss_v4"
        ] += 1

    else:

        wins[
            "log_loss_v3"
        ] += 1


    if (
        v4_metrics[
            "brier"
        ]
        <
        v3_metrics[
            "brier"
        ]
    ):

        wins[
            "brier_v4"
        ] += 1

    else:

        wins[
            "brier_v3"
        ] += 1


    if (
        v4_metrics[
            "auc"
        ]
        >
        v3_metrics[
            "auc"
        ]
    ):

        wins[
            "auc_v4"
        ] += 1

    else:

        wins[
            "auc_v3"
        ] += 1


    # ======================================================
    # STORE OOF
    # ======================================================

    outer_validation_indices = np.where(
        outer_validation_mask
    )[0]


    oof_v3[
        outer_validation_indices
    ] = (
        v3_outer_predictions
    )


    oof_v4[
        outer_validation_indices
    ] = (
        v4_outer_predictions
    )


    # ======================================================
    # STORE FOLD
    # ======================================================

    fold_results.append(
        {

            "fold":
                outer_fold,

            "outer_validation_matches":
                int(
                    len(
                        outer_validation_matches
                    )
                ),

            "outer_validation_shots":
                int(
                    len(
                        outer_valid_df
                    )
                ),

            "outer_validation_goals":
                int(
                    y_outer_valid.sum()
                ),

            "outer_training_penalties":
                train_penalty_count,

            "outer_penalty_xg":
                float(
                    outer_penalty_xg
                ),

            "inner_validation_matches":
                int(
                    len(
                        inner_validation_matches
                    )
                ),

            "selected_boost_rounds":
                int(
                    best_boost_rounds
                ),

            "v3":
                v3_metrics,

            "v4":
                v4_metrics,
        }
    )


    # ======================================================
    # PRINT FOLD
    # ======================================================

    print(
        f"\nOuter Fold {outer_fold}"
    )


    print(
        "-" * 105
    )


    print(
        f"Outer validation matches: "
        f"{len(outer_validation_matches):,}"
    )


    print(
        f"Outer validation shots:   "
        f"{len(outer_valid_df):,}"
    )


    print(
        f"Outer validation goals:   "
        f"{int(y_outer_valid.sum()):,}"
    )


    print(
        f"Inner validation matches: "
        f"{len(inner_validation_matches):,}"
    )


    print(
        f"Selected boost rounds:    "
        f"{best_boost_rounds:,}"
    )


    print(
        f"Outer-train penalty xG:   "
        f"{outer_penalty_xg:.6f}"
    )


    print(
        "\n"
        f"{'Metric':<20}"
        f"{'V3':>16}"
        f"{'V4':>16}"
        f"{'V4 - V3':>18}"
    )


    print(
        f"{'Log Loss':<20}"
        f"{v3_metrics['log_loss']:>16.6f}"
        f"{v4_metrics['log_loss']:>16.6f}"
        f"{delta_log_loss:>+18.6f}"
    )


    print(
        f"{'Brier':<20}"
        f"{v3_metrics['brier']:>16.6f}"
        f"{v4_metrics['brier']:>16.6f}"
        f"{delta_brier:>+18.6f}"
    )


    print(
        f"{'ROC-AUC':<20}"
        f"{v3_metrics['auc']:>16.6f}"
        f"{v4_metrics['auc']:>16.6f}"
        f"{delta_auc:>+18.6f}"
    )


    print(
        f"{'Predicted goals':<20}"
        f"{v3_metrics['predicted_goals']:>16.2f}"
        f"{v4_metrics['predicted_goals']:>16.2f}"
        f"{(
            v4_metrics['predicted_goals']
            -
            v3_metrics['predicted_goals']
        ):>+18.2f}"
    )


# ==========================================================
# OVERALL OOF METRICS
# ==========================================================

overall_v3 = calculate_metrics(
    y_all,
    oof_v3
)


overall_v4 = calculate_metrics(
    y_all,
    oof_v4
)


delta_log_loss = (
    overall_v4[
        "log_loss"
    ]
    -
    overall_v3[
        "log_loss"
    ]
)


delta_brier = (
    overall_v4[
        "brier"
    ]
    -
    overall_v3[
        "brier"
    ]
)


delta_auc = (
    overall_v4[
        "auc"
    ]
    -
    overall_v3[
        "auc"
    ]
)


# ==========================================================
# OVERALL
# ==========================================================

print("\n" + "=" * 105)
print("OVERALL OUTER OOF RESULTS")
print("=" * 105)


print(
    f"\n{'Metric':<22}"
    f"{'V3':>16}"
    f"{'V4':>16}"
    f"{'V4 - V3':>18}"
)


print(
    f"{'Log Loss':<22}"
    f"{overall_v3['log_loss']:>16.6f}"
    f"{overall_v4['log_loss']:>16.6f}"
    f"{delta_log_loss:>+18.6f}"
)


print(
    f"{'Brier':<22}"
    f"{overall_v3['brier']:>16.6f}"
    f"{overall_v4['brier']:>16.6f}"
    f"{delta_brier:>+18.6f}"
)


print(
    f"{'ROC-AUC':<22}"
    f"{overall_v3['auc']:>16.6f}"
    f"{overall_v4['auc']:>16.6f}"
    f"{delta_auc:>+18.6f}"
)


print(
    f"{'Predicted goals':<22}"
    f"{overall_v3['predicted_goals']:>16.2f}"
    f"{overall_v4['predicted_goals']:>16.2f}"
    f"{(
        overall_v4['predicted_goals']
        -
        overall_v3['predicted_goals']
    ):>+18.2f}"
)


print(
    f"\nActual goals: "
    f"{int(y_all.sum()):,}"
)


# ==========================================================
# FOLD WINS
# ==========================================================

print("\n" + "=" * 105)
print("OUTER FOLD WINS")
print("=" * 105)


print(
    f"\nLog Loss:"
    f"\n  V3 wins: "
    f"{wins['log_loss_v3']}/5"
    f"\n  V4 wins: "
    f"{wins['log_loss_v4']}/5"
)


print(
    f"\nBrier:"
    f"\n  V3 wins: "
    f"{wins['brier_v3']}/5"
    f"\n  V4 wins: "
    f"{wins['brier_v4']}/5"
)


print(
    f"\nROC-AUC:"
    f"\n  V3 wins: "
    f"{wins['auc_v3']}/5"
    f"\n  V4 wins: "
    f"{wins['auc_v4']}/5"
)


# ==========================================================
# BOOSTING ROUND SUMMARY
# ==========================================================

rounds_array = np.array(
    selected_boost_rounds,
    dtype=int
)


print("\n" + "=" * 105)
print("NESTED BOOSTING ROUND SUMMARY")
print("=" * 105)


for (
    fold_number,
    rounds
) in enumerate(
    selected_boost_rounds,
    start=1
):

    print(
        f"Outer Fold {fold_number}: "
        f"{rounds:,}"
    )


print(
    f"\nMean selected rounds: "
    f"{rounds_array.mean():.1f}"
)


print(
    f"Median selected rounds: "
    f"{int(np.median(rounds_array)):,}"
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


    return (
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


print("\n" + "=" * 105)
print("V3 OUTER-OOF CALIBRATION DECILES")
print("=" * 105)


v3_calibration = calibration_table(
    y_all,
    oof_v3
)


print(
    v3_calibration.to_string(

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


print("\n" + "=" * 105)
print("V4 OUTER-OOF CALIBRATION DECILES")
print("=" * 105)


v4_calibration = calibration_table(
    y_all,
    oof_v4
)


print(
    v4_calibration.to_string(

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
# FINAL SUMMARY
# ==========================================================

print("\n" + "=" * 105)
print("FINAL NESTED-STYLE COMPARISON")
print("=" * 105)


improvements = 0


if delta_log_loss < 0:

    improvements += 1


if delta_brier < 0:

    improvements += 1


if delta_auc > 0:

    improvements += 1


if improvements == 3:

    print(
        "\nV4 XGBoost improved Log Loss, "
        "Brier and ROC-AUC relative to V3 "
        "on untouched outer validation folds."
    )


elif improvements == 2:

    print(
        "\nV4 improved two of the three main metrics "
        "on untouched outer folds."
    )


elif improvements == 1:

    print(
        "\nV4 improved only one of the three main metrics "
        "on untouched outer folds."
    )


else:

    print(
        "\nV4 did not improve the three main metrics "
        "on untouched outer folds."
    )


print(
    "\nThe outer validation folds were never used "
    "for V4 early stopping."
)


print(
    "Penalty probability was calculated only "
    "from each outer-training fold."
)


print(
    "\nNo saved model files were modified."
)


print("\n" + "=" * 105)
print("NESTED-STYLE COMPARISON COMPLETE")
print("=" * 105)