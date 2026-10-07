from pathlib import Path
import json

import numpy as np
import pandas as pd
import xgboost as xgb


# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VALIDATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_blind_validation.csv"
)

V3_MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "xg_multicompetition_logistic_v3.json"
)

V4_MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "xg_multicompetition_xgboost_v4.json"
)

V4_METADATA_FILE = (
    PROJECT_ROOT
    / "models"
    / "xg_multicompetition_xgboost_v4_metadata.json"
)


# ==========================================================
# SETTINGS
# ==========================================================

EPSILON = 1e-15


# ==========================================================
# HELPERS
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

        "mean_xg":
            float(
                predictions.mean()
            ),
    }


def print_metrics(
    name,
    y_true,
    predictions
):

    metrics = calculate_metrics(
        y_true,
        predictions
    )

    print(
        f"\n{name}"
    )

    print(
        f"Log Loss:        "
        f"{metrics['log_loss']:.6f}"
    )

    print(
        f"Brier:           "
        f"{metrics['brier']:.6f}"
    )

    print(
        f"ROC-AUC:         "
        f"{metrics['auc']:.6f}"
    )

    print(
        f"Predicted goals: "
        f"{metrics['predicted_goals']:.2f}"
    )

    print(
        f"Mean xG:         "
        f"{metrics['mean_xg']:.4f}"
    )

    return metrics


# ==========================================================
# LOAD JSON
# ==========================================================

def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(
            file
        )


# ==========================================================
# START
# ==========================================================

print("=" * 100)
print("EXTERNAL VALIDATION — V3 ENHANCED LOGISTIC vs V4 XGBOOST")
print("=" * 100)

print(
    f"\nValidation data:\n"
    f"{VALIDATION_FILE}"
)

print(
    f"\nV3 model:\n"
    f"{V3_MODEL_FILE}"
)

print(
    f"\nV4 model:\n"
    f"{V4_MODEL_FILE}"
)

print(
    f"\nV4 metadata:\n"
    f"{V4_METADATA_FILE}"
)


# ==========================================================
# LOAD VALIDATION DATA
# ==========================================================

df = pd.read_csv(
    VALIDATION_FILE
)


print("\n" + "=" * 100)
print("VALIDATION DATA")
print("=" * 100)

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

    df = df.copy()

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
        "\nERROR: neither is_goal nor shot_outcome exists."
    )


print(
    f"Goals detected: "
    f"{int(df['is_goal'].sum()):,}"
)


# ==========================================================
# LOAD V3
# ==========================================================

v3_raw = load_json(
    V3_MODEL_FILE
)


if (
    "non_penalty_model"
    not in v3_raw
):

    raise SystemExit(
        "\nERROR: V3 non_penalty_model not found."
    )


v3_non_penalty = (
    v3_raw[
        "non_penalty_model"
    ]
)


V3_FEATURES = list(
    v3_non_penalty[
        "features"
    ]
)


V3_COEFFICIENTS = dict(
    v3_non_penalty[
        "coefficients"
    ]
)


V3_INTERCEPT = float(
    v3_non_penalty[
        "intercept"
    ]
)


V3_PENALTY_XG = float(
    v3_raw[
        "penalty_rule"
    ][
        "xg"
    ]
)


# ==========================================================
# LOAD V4 METADATA
# ==========================================================

v4_metadata = load_json(
    V4_METADATA_FILE
)


V4_FEATURES = list(
    v4_metadata[
        "feature_columns"
    ]
)


V4_PENALTY_XG = float(
    v4_metadata[
        "penalty_rule"
    ][
        "xg"
    ]
)


# ==========================================================
# LOAD V4 BOOSTER
# ==========================================================

v4_booster = xgb.Booster()


v4_booster.load_model(
    V4_MODEL_FILE
)


# ==========================================================
# MODEL SUMMARY
# ==========================================================

print("\n" + "=" * 100)
print("MODELS")
print("=" * 100)


print(
    f"\nV3:"
    f"\n{v3_raw.get('model_name')}"
)


print(
    f"\nV4:"
    f"\n{v4_metadata.get('model_name')}"
)


print(
    f"\nV3 features: "
    f"{len(V3_FEATURES)}"
)


for feature in V3_FEATURES:

    print(
        f"  {feature}"
    )


print(
    f"\nV4 features: "
    f"{len(V4_FEATURES)}"
)


for feature in V4_FEATURES:

    print(
        f"  {feature}"
    )


print(
    f"\nV3 penalty xG: "
    f"{V3_PENALTY_XG:.6f}"
)


print(
    f"V4 penalty xG: "
    f"{V4_PENALTY_XG:.6f}"
)


# ==========================================================
# REQUIRED COLUMNS
# ==========================================================

required_columns = [

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
        "\nValidation stopped."
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
# CHECK FEATURES
# ==========================================================

all_needed_features = list(
    dict.fromkeys(
        V3_FEATURES
        +
        V4_FEATURES
    )
)


missing_model_features = [

    feature
    for feature in all_needed_features

    if feature not in features.columns
]


if missing_model_features:

    print(
        "\nMissing model features:"
    )

    for feature in missing_model_features:

        print(
            f"  {feature}"
        )

    raise SystemExit(
        "\nValidation stopped."
    )


# ==========================================================
# VALID ROWS
# ==========================================================

valid_mask = (
    features[
        all_needed_features
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
# PENALTY MASK
# ==========================================================

shot_type = (
    df[
        "shot_type"
    ]
    .fillna("")
    .astype(str)
    .str.strip()
)


is_penalty = (
    shot_type
    .eq(
        "Penalty"
    )
)


non_penalty_mask = (
    ~is_penalty
)


# ==========================================================
# TRUE TARGET
# ==========================================================

y_true = (
    df[
        "is_goal"
    ]
    .astype(float)
    .to_numpy()
)


# ==========================================================
# V3 PREDICTIONS
# ==========================================================

v3_predictions = np.zeros(
    len(df),
    dtype=float
)


v3_linear_score = np.full(
    int(
        non_penalty_mask.sum()
    ),
    V3_INTERCEPT,
    dtype=float
)


for feature_name in V3_FEATURES:

    coefficient = float(
        V3_COEFFICIENTS[
            feature_name
        ]
    )


    feature_values = (
        features
        .loc[
            non_penalty_mask,
            feature_name
        ]
        .astype(float)
        .to_numpy()
    )


    v3_linear_score += (
        coefficient
        * feature_values
    )


v3_predictions[
    non_penalty_mask.to_numpy()
] = sigmoid(
    v3_linear_score
)


v3_predictions[
    is_penalty.to_numpy()
] = (
    V3_PENALTY_XG
)


# ==========================================================
# V4 PREDICTIONS
# ==========================================================

v4_predictions = np.zeros(
    len(df),
    dtype=float
)


X_v4_non_penalty = (
    features
    .loc[
        non_penalty_mask,
        V4_FEATURES
    ]
    .astype(float)
    .to_numpy()
)


dvalidation_v4 = xgb.DMatrix(
    X_v4_non_penalty,
    feature_names=V4_FEATURES
)


v4_non_penalty_predictions = (
    v4_booster.predict(
        dvalidation_v4
    )
)


v4_predictions[
    non_penalty_mask.to_numpy()
] = (
    v4_non_penalty_predictions
)


v4_predictions[
    is_penalty.to_numpy()
] = (
    V4_PENALTY_XG
)


# ==========================================================
# STATSBOMB
# ==========================================================

statsbomb_column = None


for candidate in [

    "shot_statsbomb_xg",
    "statsbomb_xg",

]:

    if candidate in df.columns:

        statsbomb_column = (
            candidate
        )

        break


if statsbomb_column is not None:

    statsbomb_predictions = (
        pd.to_numeric(
            df[
                statsbomb_column
            ],
            errors="coerce"
        )
        .to_numpy()
    )

else:

    statsbomb_predictions = None


# ==========================================================
# VALIDATION SUMMARY
# ==========================================================

print("\n" + "=" * 100)
print("VALIDATION SUMMARY")
print("=" * 100)


print(
    f"Usable shots: "
    f"{len(df):,}"
)


print(
    f"Goals: "
    f"{int(y_true.sum()):,}"
)


print(
    f"Goal rate: "
    f"{y_true.mean():.4f}"
)


print(
    f"Penalties: "
    f"{int(is_penalty.sum()):,}"
)


# ==========================================================
# OVERALL VALIDATION
# ==========================================================

print("\n" + "=" * 100)
print("OVERALL EXTERNAL VALIDATION")
print("=" * 100)


v3_metrics = print_metrics(
    "V3 Enhanced Logistic",
    y_true,
    v3_predictions
)


v4_metrics = print_metrics(
    "V4 XGBoost",
    y_true,
    v4_predictions
)


statsbomb_metrics = None


if statsbomb_predictions is not None:

    valid_sb = (
        ~np.isnan(
            statsbomb_predictions
        )
    )

    if valid_sb.all():

        statsbomb_metrics = print_metrics(
            "StatsBomb",
            y_true,
            statsbomb_predictions
        )

    else:

        print(
            "\nStatsBomb xG contains missing values."
        )


# ==========================================================
# V4 CHANGE RELATIVE TO V3
# ==========================================================

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


delta_goals = (
    v4_metrics[
        "predicted_goals"
    ]
    -
    v3_metrics[
        "predicted_goals"
    ]
)


print("\n" + "=" * 100)
print("V4 CHANGE RELATIVE TO V3")
print("=" * 100)


print(
    f"Log Loss change: "
    f"{delta_log_loss:+.6f}"
)


print(
    f"Brier change: "
    f"{delta_brier:+.6f}"
)


print(
    f"ROC-AUC change: "
    f"{delta_auc:+.6f}"
)


print(
    f"Predicted goals change: "
    f"{delta_goals:+.2f}"
)


# ==========================================================
# TOTAL GOAL CALIBRATION
# ==========================================================

actual_goals = float(
    y_true.sum()
)


v3_goal_error = (
    v3_metrics[
        "predicted_goals"
    ]
    -
    actual_goals
)


v4_goal_error = (
    v4_metrics[
        "predicted_goals"
    ]
    -
    actual_goals
)


print("\n" + "=" * 100)
print("CALIBRATION — TOTAL GOALS")
print("=" * 100)


print(
    f"Actual goals: "
    f"{actual_goals:.0f}"
)


print(
    f"\nV3 predicted: "
    f"{v3_metrics['predicted_goals']:.2f}"
)


print(
    f"V3 error: "
    f"{v3_goal_error:+.2f}"
)


print(
    f"\nV4 predicted: "
    f"{v4_metrics['predicted_goals']:.2f}"
)


print(
    f"V4 error: "
    f"{v4_goal_error:+.2f}"
)


if statsbomb_metrics is not None:

    statsbomb_error = (
        statsbomb_metrics[
            "predicted_goals"
        ]
        -
        actual_goals
    )

    print(
        f"\nStatsBomb predicted: "
        f"{statsbomb_metrics['predicted_goals']:.2f}"
    )

    print(
        f"StatsBomb error: "
        f"{statsbomb_error:+.2f}"
    )


# ==========================================================
# PLAY PATTERN COMPARISON
# ==========================================================

print("\n" + "=" * 100)
print("V3 vs V4 BY SELECTED PLAY PATTERN")
print("=" * 100)


patterns = {

    "Regular Play":
        features[
            "play_regular"
        ]
        .eq(1)
        .to_numpy(),

    "Counterattack":
        features[
            "play_counter"
        ]
        .eq(1)
        .to_numpy(),

    "Corner":
        features[
            "play_corner"
        ]
        .eq(1)
        .to_numpy(),
}


for (
    pattern_name,
    pattern_mask
) in patterns.items():


    n_shots = int(
        pattern_mask.sum()
    )


    if n_shots == 0:

        continue


    subset_y = (
        y_true[
            pattern_mask
        ]
    )


    subset_v3 = (
        v3_predictions[
            pattern_mask
        ]
    )


    subset_v4 = (
        v4_predictions[
            pattern_mask
        ]
    )


    print(
        f"\n{pattern_name}"
    )


    print(
        f"Shots: "
        f"{n_shots:,}"
    )


    print(
        f"Goals: "
        f"{int(subset_y.sum()):,}"
    )


    print(
        f"Actual rate: "
        f"{subset_y.mean():.4f}"
    )


    print(
        f"V3 mean xG: "
        f"{subset_v3.mean():.4f}"
    )


    print(
        f"V4 mean xG: "
        f"{subset_v4.mean():.4f}"
    )


    print(
        f"V3 Log Loss: "
        f"{log_loss(subset_y, subset_v3):.6f}"
    )


    print(
        f"V4 Log Loss: "
        f"{log_loss(subset_y, subset_v4):.6f}"
    )


    print(
        f"V3 Brier: "
        f"{brier_score(subset_y, subset_v3):.6f}"
    )


    print(
        f"V4 Brier: "
        f"{brier_score(subset_y, subset_v4):.6f}"
    )


    print(
        f"V3 total xG: "
        f"{subset_v3.sum():.2f}"
    )


    print(
        f"V4 total xG: "
        f"{subset_v4.sum():.2f}"
    )


# ==========================================================
# HEADERS
# ==========================================================

header_mask = (
    features[
        "body_head"
    ]
    .eq(1)
    .to_numpy()
)


if int(
    header_mask.sum()
) > 0:


    header_y = (
        y_true[
            header_mask
        ]
    )


    header_v3 = (
        v3_predictions[
            header_mask
        ]
    )


    header_v4 = (
        v4_predictions[
            header_mask
        ]
    )


    print("\n" + "=" * 100)
    print("HEADERS")
    print("=" * 100)


    print(
        f"Shots: "
        f"{int(header_mask.sum()):,}"
    )


    print(
        f"Goals: "
        f"{int(header_y.sum()):,}"
    )


    print(
        f"Actual rate: "
        f"{header_y.mean():.4f}"
    )


    print(
        f"V3 mean xG: "
        f"{header_v3.mean():.4f}"
    )


    print(
        f"V4 mean xG: "
        f"{header_v4.mean():.4f}"
    )


    print(
        f"V3 Log Loss: "
        f"{log_loss(header_y, header_v3):.6f}"
    )


    print(
        f"V4 Log Loss: "
        f"{log_loss(header_y, header_v4):.6f}"
    )


    print(
        f"V3 Brier: "
        f"{brier_score(header_y, header_v3):.6f}"
    )


    print(
        f"V4 Brier: "
        f"{brier_score(header_y, header_v4):.6f}"
    )


# ==========================================================
# ONE-ON-ONES
# ==========================================================

one_on_one_mask = (
    features[
        "shot_one_on_one"
    ]
    .eq(1)
    .to_numpy()
)


if int(
    one_on_one_mask.sum()
) > 0:


    subset_y = (
        y_true[
            one_on_one_mask
        ]
    )


    subset_v3 = (
        v3_predictions[
            one_on_one_mask
        ]
    )


    subset_v4 = (
        v4_predictions[
            one_on_one_mask
        ]
    )


    print("\n" + "=" * 100)
    print("ONE-ON-ONES")
    print("=" * 100)


    print(
        f"Shots: "
        f"{int(one_on_one_mask.sum()):,}"
    )


    print(
        f"Goals: "
        f"{int(subset_y.sum()):,}"
    )


    print(
        f"Actual rate: "
        f"{subset_y.mean():.4f}"
    )


    print(
        f"V3 mean xG: "
        f"{subset_v3.mean():.4f}"
    )


    print(
        f"V4 mean xG: "
        f"{subset_v4.mean():.4f}"
    )


    print(
        f"V3 Log Loss: "
        f"{log_loss(subset_y, subset_v3):.6f}"
    )


    print(
        f"V4 Log Loss: "
        f"{log_loss(subset_y, subset_v4):.6f}"
    )


    print(
        f"V3 Brier: "
        f"{brier_score(subset_y, subset_v3):.6f}"
    )


    print(
        f"V4 Brier: "
        f"{brier_score(subset_y, subset_v4):.6f}"
    )


# ==========================================================
# OPEN GOALS
# ==========================================================

open_goal_mask = (
    features[
        "shot_open_goal"
    ]
    .eq(1)
    .to_numpy()
)


if int(
    open_goal_mask.sum()
) > 0:


    subset_y = (
        y_true[
            open_goal_mask
        ]
    )


    subset_v3 = (
        v3_predictions[
            open_goal_mask
        ]
    )


    subset_v4 = (
        v4_predictions[
            open_goal_mask
        ]
    )


    print("\n" + "=" * 100)
    print("OPEN GOALS")
    print("=" * 100)


    print(
        f"Shots: "
        f"{int(open_goal_mask.sum()):,}"
    )


    print(
        f"Goals: "
        f"{int(subset_y.sum()):,}"
    )


    print(
        f"Actual rate: "
        f"{subset_y.mean():.4f}"
    )


    print(
        f"V3 mean xG: "
        f"{subset_v3.mean():.4f}"
    )


    print(
        f"V4 mean xG: "
        f"{subset_v4.mean():.4f}"
    )


# ==========================================================
# FREE KICKS
# ==========================================================

free_kick_mask = (
    features[
        "type_free_kick"
    ]
    .eq(1)
    .to_numpy()
)


if int(
    free_kick_mask.sum()
) > 0:


    subset_y = (
        y_true[
            free_kick_mask
        ]
    )


    subset_v3 = (
        v3_predictions[
            free_kick_mask
        ]
    )


    subset_v4 = (
        v4_predictions[
            free_kick_mask
        ]
    )


    print("\n" + "=" * 100)
    print("FREE KICKS")
    print("=" * 100)


    print(
        f"Shots: "
        f"{int(free_kick_mask.sum()):,}"
    )


    print(
        f"Goals: "
        f"{int(subset_y.sum()):,}"
    )


    print(
        f"Actual rate: "
        f"{subset_y.mean():.4f}"
    )


    print(
        f"V3 mean xG: "
        f"{subset_v3.mean():.4f}"
    )


    print(
        f"V4 mean xG: "
        f"{subset_v4.mean():.4f}"
    )


# ==========================================================
# CALIBRATION TABLE
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


# ==========================================================
# V3 CALIBRATION
# ==========================================================

print("\n" + "=" * 100)
print("V3 CALIBRATION DECILES")
print("=" * 100)


v3_calibration = calibration_table(
    y_true,
    v3_predictions
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


# ==========================================================
# V4 CALIBRATION
# ==========================================================

print("\n" + "=" * 100)
print("V4 CALIBRATION DECILES")
print("=" * 100)


v4_calibration = calibration_table(
    y_true,
    v4_predictions
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
# FINAL DECISION
# ==========================================================

print("\n" + "=" * 100)
print("FINAL EXTERNAL VALIDATION SUMMARY")
print("=" * 100)


main_improvements = 0


if delta_log_loss < 0:

    main_improvements += 1


if delta_brier < 0:

    main_improvements += 1


if delta_auc > 0:

    main_improvements += 1


if main_improvements == 3:

    print(
        "\nV4 XGBoost improved all three main metrics "
        "relative to V3."
    )


elif main_improvements == 2:

    print(
        "\nV4 XGBoost improved two of the three main metrics "
        "relative to V3."
    )


elif main_improvements == 1:

    print(
        "\nV4 XGBoost improved only one of the three "
        "main metrics relative to V3."
    )


else:

    print(
        "\nV4 XGBoost did not improve the main metrics "
        "relative to V3."
    )


if (
    abs(
        v4_goal_error
    )
    <
    abs(
        v3_goal_error
    )
):

    print(
        "V4 also improved total-goal calibration."
    )

else:

    print(
        "V4 did not improve total-goal calibration."
    )


print(
    "\nNo model files were modified."
)


print("\n" + "=" * 100)
print("EXTERNAL VALIDATION COMPLETE")
print("=" * 100)