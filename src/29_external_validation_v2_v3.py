from pathlib import Path
import json

import numpy as np
import pandas as pd


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

V2_MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "xg_multicompetition_logistic_v2.json"
)

V3_MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "xg_multicompetition_logistic_v3.json"
)


# ==========================================================
# SETTINGS
# ==========================================================

EPSILON = 1e-15


# ==========================================================
# BASIC HELPERS
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
            * np.log(probabilities)
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
# LOAD MODEL
# ==========================================================

def load_model(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(
            file
        )


def normalize_model(
    model,
    label
):

    if (
        "non_penalty_model" not in model
        or
        not isinstance(
            model["non_penalty_model"],
            dict
        )
    ):

        raise ValueError(
            f"{label}: non_penalty_model not found."
        )


    non_penalty_model = (
        model[
            "non_penalty_model"
        ]
    )


    required_keys = [
        "features",
        "intercept",
        "coefficients",
    ]


    missing_keys = [
        key
        for key in required_keys
        if key not in non_penalty_model
    ]


    if missing_keys:

        raise ValueError(
            f"{label}: missing keys: "
            + ", ".join(
                missing_keys
            )
        )


    if (
        "penalty_rule" not in model
        or
        "xg" not in model[
            "penalty_rule"
        ]
    ):

        raise ValueError(
            f"{label}: penalty_rule.xg not found."
        )


    return {

        "model_name":
            model.get(
                "model_name",
                label
            ),

        "model_type":
            model.get(
                "model_type",
                ""
            ),

        "features":
            list(
                non_penalty_model[
                    "features"
                ]
            ),

        "intercept":
            float(
                non_penalty_model[
                    "intercept"
                ]
            ),

        "coefficients":
            dict(
                non_penalty_model[
                    "coefficients"
                ]
            ),

        "penalty_xg":
            float(
                model[
                    "penalty_rule"
                ][
                    "xg"
                ]
            ),
    }


# ==========================================================
# START
# ==========================================================

print("=" * 95)
print("EXTERNAL VALIDATION — V2 vs V3")
print("=" * 95)

print(
    f"\nValidation data:\n"
    f"{VALIDATION_FILE}"
)

print(
    f"\nV2 model:\n"
    f"{V2_MODEL_FILE}"
)

print(
    f"\nV3 model:\n"
    f"{V3_MODEL_FILE}"
)


# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_csv(
    VALIDATION_FILE
)


print("\n" + "=" * 95)
print("VALIDATION DATA")
print("=" * 95)

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
        "\nERROR: neither is_goal nor shot_outcome exists."
    )


print(
    f"Goals detected: "
    f"{int(df['is_goal'].sum()):,}"
)


# ==========================================================
# LOAD MODELS
# ==========================================================

v2_raw = load_model(
    V2_MODEL_FILE
)

v3_raw = load_model(
    V3_MODEL_FILE
)


v2_model = normalize_model(
    v2_raw,
    "V2"
)

v3_model = normalize_model(
    v3_raw,
    "V3"
)


print("\n" + "=" * 95)
print("MODELS")
print("=" * 95)

print(
    f"V2: "
    f"{v2_model['model_name']}"
)

print(
    f"V3: "
    f"{v3_model['model_name']}"
)


print(
    f"\nV2 features: "
    f"{len(v2_model['features'])}"
)

for feature in v2_model[
    "features"
]:
    print(
        f"  {feature}"
    )


print(
    f"\nV3 features: "
    f"{len(v3_model['features'])}"
)

for feature in v3_model[
    "features"
]:
    print(
        f"  {feature}"
    )


# ==========================================================
# REQUIRED RAW COLUMNS
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

    print("\nMissing required columns:")

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

all_needed_features = list(
    dict.fromkeys(
        v2_model[
            "features"
        ]
        +
        v3_model[
            "features"
        ]
    )
)


missing_model_features = [
    feature
    for feature in all_needed_features
    if feature not in features.columns
]


if missing_model_features:

    print("\nMissing model features:")

    for feature in missing_model_features:

        print(
            f"  {feature}"
        )

    raise SystemExit(
        "\nValidation stopped."
    )


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
# PENALTIES
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


# ==========================================================
# PREDICTION FUNCTION
# ==========================================================

def predict_model(
    model,
    feature_df,
    penalty_mask
):

    predictions = np.zeros(
        len(feature_df),
        dtype=float
    )


    non_penalty_mask = (
        ~penalty_mask
    )


    linear_score = np.full(
        int(
            non_penalty_mask.sum()
        ),
        model[
            "intercept"
        ],
        dtype=float
    )


    for feature_name in model[
        "features"
    ]:

        coefficient = float(
            model[
                "coefficients"
            ][
                feature_name
            ]
        )


        feature_values = (
            feature_df
            .loc[
                non_penalty_mask,
                feature_name
            ]
            .astype(float)
            .to_numpy()
        )


        linear_score += (
            coefficient
            * feature_values
        )


    predictions[
        non_penalty_mask.to_numpy()
    ] = sigmoid(
        linear_score
    )


    predictions[
        penalty_mask.to_numpy()
    ] = (
        model[
            "penalty_xg"
        ]
    )


    return predictions


# ==========================================================
# PREDICTIONS
# ==========================================================

y_true = (
    df[
        "is_goal"
    ]
    .astype(float)
    .to_numpy()
)


v2_predictions = predict_model(
    v2_model,
    features,
    is_penalty
)


v3_predictions = predict_model(
    v3_model,
    features,
    is_penalty
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
# METRICS
# ==========================================================

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

        "mean_xg":
            float(
                predictions.mean()
            ),
    }


def print_metrics(
    name,
    y,
    predictions
):

    metrics = calculate_metrics(
        y,
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
# SUMMARY
# ==========================================================

print("\n" + "=" * 95)
print("VALIDATION SUMMARY")
print("=" * 95)


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
# OVERALL
# ==========================================================

print("\n" + "=" * 95)
print("OVERALL EXTERNAL VALIDATION")
print("=" * 95)


v2_metrics = print_metrics(
    "V2",
    y_true,
    v2_predictions
)


v3_metrics = print_metrics(
    "V3",
    y_true,
    v3_predictions
)


statsbomb_metrics = None


if statsbomb_predictions is not None:

    valid_sb_mask = (
        ~np.isnan(
            statsbomb_predictions
        )
    )


    if valid_sb_mask.all():

        statsbomb_metrics = print_metrics(
            "StatsBomb",
            y_true,
            statsbomb_predictions
        )


    else:

        print(
            "\nStatsBomb xG has missing values."
        )


# ==========================================================
# V3 CHANGE RELATIVE TO V2
# ==========================================================

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


delta_goals = (
    v3_metrics[
        "predicted_goals"
    ]
    -
    v2_metrics[
        "predicted_goals"
    ]
)


print("\n" + "=" * 95)
print("V3 CHANGE RELATIVE TO V2")
print("=" * 95)


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
# ERROR VS ACTUAL GOALS
# ==========================================================

actual_goals = float(
    y_true.sum()
)


v2_goal_error = (
    v2_metrics[
        "predicted_goals"
    ]
    -
    actual_goals
)


v3_goal_error = (
    v3_metrics[
        "predicted_goals"
    ]
    -
    actual_goals
)


print("\n" + "=" * 95)
print("CALIBRATION — TOTAL GOALS")
print("=" * 95)


print(
    f"Actual goals: "
    f"{actual_goals:.0f}"
)


print(
    f"V2 predicted: "
    f"{v2_metrics['predicted_goals']:.2f}"
)


print(
    f"V2 error: "
    f"{v2_goal_error:+.2f}"
)


print(
    f"\nV3 predicted: "
    f"{v3_metrics['predicted_goals']:.2f}"
)


print(
    f"V3 error: "
    f"{v3_goal_error:+.2f}"
)


# ==========================================================
# BY SELECTED PLAY PATTERN
# ==========================================================

print("\n" + "=" * 95)
print("V2 vs V3 BY SELECTED PLAY PATTERN")
print("=" * 95)


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


    pattern_y = (
        y_true[
            pattern_mask
        ]
    )


    pattern_v2 = (
        v2_predictions[
            pattern_mask
        ]
    )


    pattern_v3 = (
        v3_predictions[
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
        f"{int(pattern_y.sum()):,}"
    )


    print(
        f"Actual rate: "
        f"{pattern_y.mean():.4f}"
    )


    print(
        f"V2 mean xG: "
        f"{pattern_v2.mean():.4f}"
    )


    print(
        f"V3 mean xG: "
        f"{pattern_v3.mean():.4f}"
    )


    print(
        f"V2 Log Loss: "
        f"{log_loss(pattern_y, pattern_v2):.6f}"
    )


    print(
        f"V3 Log Loss: "
        f"{log_loss(pattern_y, pattern_v3):.6f}"
    )


    print(
        f"V2 Brier: "
        f"{brier_score(pattern_y, pattern_v2):.6f}"
    )


    print(
        f"V3 Brier: "
        f"{brier_score(pattern_y, pattern_v3):.6f}"
    )


    print(
        f"V2 total xG: "
        f"{pattern_v2.sum():.2f}"
    )


    print(
        f"V3 total xG: "
        f"{pattern_v3.sum():.2f}"
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


    header_v2 = (
        v2_predictions[
            header_mask
        ]
    )


    header_v3 = (
        v3_predictions[
            header_mask
        ]
    )


    print("\n" + "=" * 95)
    print("HEADERS")
    print("=" * 95)


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
        f"V2 mean xG: "
        f"{header_v2.mean():.4f}"
    )


    print(
        f"V3 mean xG: "
        f"{header_v3.mean():.4f}"
    )


    print(
        f"V2 Log Loss: "
        f"{log_loss(header_y, header_v2):.6f}"
    )


    print(
        f"V3 Log Loss: "
        f"{log_loss(header_y, header_v3):.6f}"
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

    ooo_y = (
        y_true[
            one_on_one_mask
        ]
    )


    ooo_v2 = (
        v2_predictions[
            one_on_one_mask
        ]
    )


    ooo_v3 = (
        v3_predictions[
            one_on_one_mask
        ]
    )


    print("\n" + "=" * 95)
    print("ONE-ON-ONES")
    print("=" * 95)


    print(
        f"Shots: "
        f"{int(one_on_one_mask.sum()):,}"
    )


    print(
        f"Goals: "
        f"{int(ooo_y.sum()):,}"
    )


    print(
        f"Actual rate: "
        f"{ooo_y.mean():.4f}"
    )


    print(
        f"V2 mean xG: "
        f"{ooo_v2.mean():.4f}"
    )


    print(
        f"V3 mean xG: "
        f"{ooo_v3.mean():.4f}"
    )


    print(
        f"V2 Log Loss: "
        f"{log_loss(ooo_y, ooo_v2):.6f}"
    )


    print(
        f"V3 Log Loss: "
        f"{log_loss(ooo_y, ooo_v3):.6f}"
    )


# ==========================================================
# CALIBRATION DECILES
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
print("V2 CALIBRATION DECILES")
print("=" * 95)


v2_calibration = calibration_table(
    y_true,
    v2_predictions
)


print(
    v2_calibration.to_string(
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


print("\n" + "=" * 95)
print("V3 CALIBRATION DECILES")
print("=" * 95)


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
# FINAL DECISION
# ==========================================================

print("\n" + "=" * 95)
print("FINAL EXTERNAL VALIDATION SUMMARY")
print("=" * 95)


improvements = 0


if delta_log_loss < 0:

    improvements += 1


if delta_brier < 0:

    improvements += 1


if delta_auc > 0:

    improvements += 1


if improvements == 3:

    print(
        "\nV3 improved all three main metrics "
        "relative to V2 on external validation."
    )


elif improvements == 2:

    print(
        "\nV3 improved two of the three main metrics "
        "relative to V2 on external validation."
    )


elif improvements == 1:

    print(
        "\nV3 improved only one of the three main metrics "
        "relative to V2 on external validation."
    )


else:

    print(
        "\nV3 did not improve the main metrics "
        "relative to V2 on external validation."
    )


if (
    abs(
        v3_goal_error
    )
    <
    abs(
        v2_goal_error
    )
):

    print(
        "V3 also improved total-goal calibration."
    )


else:

    print(
        "V3 did not improve total-goal calibration."
    )


print(
    "\nNo model files were modified."
)


print("\n" + "=" * 95)
print("EXTERNAL VALIDATION COMPLETE")
print("=" * 95)