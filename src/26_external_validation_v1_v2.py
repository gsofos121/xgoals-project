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

V1_MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "xg_multicompetition_logistic.json"
)

V2_MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "xg_multicompetition_logistic_v2.json"
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


# ==========================================================
# NORMALIZE MODEL STRUCTURE
# ==========================================================

def normalize_model(
    model,
    label
):
    """
    Supports the common nested structure used by V1 and V2:

    penalty_rule
        xg

    non_penalty_model
        features
        intercept
        coefficients

    Also keeps a flat-format fallback in case an older V2
    file is ever used.
    """

    # ------------------------------------------------------
    # NESTED FORMAT
    # ------------------------------------------------------

    if (
        "non_penalty_model" in model
        and
        isinstance(
            model["non_penalty_model"],
            dict
        )
    ):
        non_penalty_model = (
            model[
                "non_penalty_model"
            ]
        )

        required_nested_keys = [
            "features",
            "intercept",
            "coefficients",
        ]

        missing_nested_keys = [
            key
            for key in required_nested_keys
            if key not in non_penalty_model
        ]

        if missing_nested_keys:
            raise ValueError(
                f"{label}: missing nested keys: "
                + ", ".join(
                    missing_nested_keys
                )
            )

        if (
            "penalty_rule" not in model
            or
            not isinstance(
                model["penalty_rule"],
                dict
            )
            or
            "xg"
            not in model["penalty_rule"]
        ):
            raise ValueError(
                f"{label}: penalty_rule.xg not found."
            )

        features = list(
            non_penalty_model[
                "features"
            ]
        )

        coefficients = dict(
            non_penalty_model[
                "coefficients"
            ]
        )

        intercept = float(
            non_penalty_model[
                "intercept"
            ]
        )

        penalty_xg = float(
            model[
                "penalty_rule"
            ][
                "xg"
            ]
        )

        return {
            "model_name":
                model.get(
                    "model_name",
                    label
                ),

            "features":
                features,

            "coefficients":
                coefficients,

            "intercept":
                intercept,

            "penalty_xg":
                penalty_xg,
        }

    # ------------------------------------------------------
    # FLAT FORMAT FALLBACK
    # ------------------------------------------------------

    if (
        "features" in model
        and
        "coefficients" in model
        and
        "intercept" in model
    ):
        if "penalty_xg" in model:
            penalty_xg = float(
                model[
                    "penalty_xg"
                ]
            )

        elif "fixed_penalty_xg" in model:
            penalty_xg = float(
                model[
                    "fixed_penalty_xg"
                ]
            )

        else:
            raise ValueError(
                f"{label}: penalty xG not found."
            )

        return {
            "model_name":
                model.get(
                    "model_name",
                    label
                ),

            "features":
                list(
                    model[
                        "features"
                    ]
                ),

            "coefficients":
                dict(
                    model[
                        "coefficients"
                    ]
                ),

            "intercept":
                float(
                    model[
                        "intercept"
                    ]
                ),

            "penalty_xg":
                penalty_xg,
        }

    raise ValueError(
        f"{label}: unknown model JSON structure."
    )


# ==========================================================
# START
# ==========================================================

print("=" * 90)
print("EXTERNAL VALIDATION — V1 vs V2")
print("=" * 90)

print(
    f"\nValidation data:\n"
    f"{VALIDATION_FILE}"
)

print(
    f"\nV1 model:\n"
    f"{V1_MODEL_FILE}"
)

print(
    f"\nV2 model:\n"
    f"{V2_MODEL_FILE}"
)


# ==========================================================
# LOAD VALIDATION DATA
# ==========================================================

df = pd.read_csv(
    VALIDATION_FILE
)

print("\n" + "=" * 90)
print("VALIDATION DATA")
print("=" * 90)

print(
    f"Rows loaded: "
    f"{len(df):,}"
)

print(
    f"Columns: "
    f"{len(df.columns):,}"
)


# ==========================================================
# LOAD + NORMALIZE MODELS
# ==========================================================

v1_raw = load_model(
    V1_MODEL_FILE
)

v2_raw = load_model(
    V2_MODEL_FILE
)

v1_model = normalize_model(
    v1_raw,
    "V1"
)

v2_model = normalize_model(
    v2_raw,
    "V2"
)


print("\n" + "=" * 90)
print("MODELS")
print("=" * 90)

print(
    f"V1: "
    f"{v1_model['model_name']}"
)

print(
    f"V2: "
    f"{v2_model['model_name']}"
)


print("\nV1 features:")

for feature in v1_model[
    "features"
]:
    print(
        f"  {feature}"
    )


print("\nV2 features:")

for feature in v2_model[
    "features"
]:
    print(
        f"  {feature}"
    )


print(
    f"\nV1 penalty xG: "
    f"{v1_model['penalty_xg']:.4f}"
)

print(
    f"V2 penalty xG: "
    f"{v2_model['penalty_xg']:.4f}"
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

    df = df.copy()

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
    print("\n" + "=" * 90)
    print("MISSING REQUIRED COLUMNS")
    print("=" * 90)

    for column in missing_columns:
        print(
            column
        )

    print("\nAvailable columns:\n")

    for column in df.columns:
        print(
            column
        )

    raise SystemExit(
        "\nValidation stopped."
    )


# ==========================================================
# BUILD FEATURE TABLE
# ==========================================================

feature_data = {}


# ----------------------------------------------------------
# GEOMETRY
# ----------------------------------------------------------

feature_data[
    "distance_m"
] = pd.to_numeric(
    df[
        "distance_m"
    ],
    errors="coerce"
)

feature_data[
    "angle_deg"
] = pd.to_numeric(
    df[
        "angle_deg"
    ],
    errors="coerce"
)


# ----------------------------------------------------------
# SHOT CONDITIONS
# ----------------------------------------------------------

feature_data[
    "under_pressure"
] = bool_to_int(
    df[
        "under_pressure"
    ]
)

feature_data[
    "shot_first_time"
] = bool_to_int(
    df[
        "shot_first_time"
    ]
)

feature_data[
    "shot_one_on_one"
] = bool_to_int(
    df[
        "shot_one_on_one"
    ]
)

feature_data[
    "shot_open_goal"
] = bool_to_int(
    df[
        "shot_open_goal"
    ]
)


# ----------------------------------------------------------
# BODY PART
# ----------------------------------------------------------

body_part = (
    df[
        "shot_body_part"
    ]
    .fillna("")
    .astype(str)
    .str.strip()
)

feature_data[
    "body_left_foot"
] = (
    body_part
    .eq(
        "Left Foot"
    )
    .astype(int)
)

feature_data[
    "body_head"
] = (
    body_part
    .eq(
        "Head"
    )
    .astype(int)
)


# ----------------------------------------------------------
# SHOT TYPE
# ----------------------------------------------------------

shot_type_raw = (
    df[
        "shot_type"
    ]
    .fillna("")
    .astype(str)
    .str.strip()
)

feature_data[
    "type_free_kick"
] = (
    shot_type_raw
    .eq(
        "Free Kick"
    )
    .astype(int)
)


# ----------------------------------------------------------
# PLAY PATTERN — V2
# ----------------------------------------------------------

play_pattern = (
    df[
        "play_pattern"
    ]
    .fillna("")
    .astype(str)
    .str.strip()
)

feature_data[
    "play_regular"
] = (
    play_pattern
    .eq(
        "Regular Play"
    )
    .astype(int)
)

feature_data[
    "play_counter"
] = (
    play_pattern
    .eq(
        "From Counter"
    )
    .astype(int)
)

feature_data[
    "play_corner"
] = (
    play_pattern
    .eq(
        "From Corner"
    )
    .astype(int)
)


features = pd.DataFrame(
    feature_data,
    index=df.index
)


# ==========================================================
# CHECK MODEL FEATURES
# ==========================================================

all_needed_features = list(
    dict.fromkeys(
        v1_model[
            "features"
        ]
        +
        v2_model[
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
    print("\n" + "=" * 90)
    print("MISSING MODEL FEATURES")
    print("=" * 90)

    for feature in missing_model_features:
        print(
            feature
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

        values = (
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
            * values
        )

    predictions[
        non_penalty_mask.to_numpy()
    ] = sigmoid(
        linear_score
    )

    predictions[
        penalty_mask.to_numpy()
    ] = model[
        "penalty_xg"
    ]

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

v1_predictions = predict_model(
    v1_model,
    features,
    is_penalty
)

v2_predictions = predict_model(
    v2_model,
    features,
    is_penalty
)


# ==========================================================
# STATSBOMB xG
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

print("\n" + "=" * 90)
print("VALIDATION SUMMARY")
print("=" * 90)

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
# PLAY PATTERN COUNTS
# ==========================================================

print("\n" + "=" * 90)
print("VALIDATION PLAY-PATTERN COUNTS")
print("=" * 90)

print(
    f"Regular Play: "
    f"{int(features['play_regular'].sum()):,}"
)

print(
    f"Counterattack: "
    f"{int(features['play_counter'].sum()):,}"
)

print(
    f"Corner: "
    f"{int(features['play_corner'].sum()):,}"
)


# ==========================================================
# METRIC HELPERS
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
    }


def print_metrics(
    name,
    y,
    predictions
):
    metrics = (
        calculate_metrics(
            y,
            predictions
        )
    )

    print(
        f"\n{name}"
    )

    print(
        f"Log Loss:        "
        f"{metrics['log_loss']:.4f}"
    )

    print(
        f"Brier:           "
        f"{metrics['brier']:.4f}"
    )

    print(
        f"ROC-AUC:         "
        f"{metrics['auc']:.4f}"
    )

    print(
        f"Predicted goals: "
        f"{metrics['predicted_goals']:.1f}"
    )

    return metrics


# ==========================================================
# OVERALL VALIDATION
# ==========================================================

print("\n" + "=" * 90)
print("OVERALL EXTERNAL VALIDATION")
print("=" * 90)


v1_metrics = print_metrics(
    "V1",
    y_true,
    v1_predictions
)

v2_metrics = print_metrics(
    "V2",
    y_true,
    v2_predictions
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
            "\nStatsBomb xG contains missing values, "
            "so overall StatsBomb metrics were skipped."
        )

else:
    print(
        "\nStatsBomb xG column not found."
    )


# ==========================================================
# V2 CHANGE RELATIVE TO V1
# ==========================================================

delta_log_loss = (
    v2_metrics[
        "log_loss"
    ]
    -
    v1_metrics[
        "log_loss"
    ]
)

delta_brier = (
    v2_metrics[
        "brier"
    ]
    -
    v1_metrics[
        "brier"
    ]
)

delta_auc = (
    v2_metrics[
        "auc"
    ]
    -
    v1_metrics[
        "auc"
    ]
)

delta_goals = (
    v2_metrics[
        "predicted_goals"
    ]
    -
    v1_metrics[
        "predicted_goals"
    ]
)


print("\n" + "=" * 90)
print("V2 CHANGE RELATIVE TO V1")
print("=" * 90)

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
# RESULTS BY COMPETITION
# ==========================================================

competition_column = None

for candidate in [
    "competition_name",
    "competition",
]:
    if candidate in df.columns:
        competition_column = (
            candidate
        )

        break


if competition_column is not None:
    print("\n" + "=" * 90)
    print("RESULTS BY COMPETITION")
    print("=" * 90)

    competitions = (
        df[
            competition_column
        ]
        .dropna()
        .unique()
    )

    for competition_name in sorted(
        competitions
    ):
        competition_mask = (
            df[
                competition_column
            ]
            .eq(
                competition_name
            )
            .to_numpy()
        )

        competition_y = (
            y_true[
                competition_mask
            ]
        )

        competition_v1 = (
            v1_predictions[
                competition_mask
            ]
        )

        competition_v2 = (
            v2_predictions[
                competition_mask
            ]
        )

        print(
            f"\n{competition_name}"
        )

        print(
            f"Shots: "
            f"{len(competition_y):,}"
        )

        print(
            f"Goals: "
            f"{int(competition_y.sum()):,}"
        )

        print_metrics(
            "V1",
            competition_y,
            competition_v1
        )

        print_metrics(
            "V2",
            competition_y,
            competition_v2
        )


# ==========================================================
# V1 vs V2 BY SELECTED PLAY PATTERN
# ==========================================================

print("\n" + "=" * 90)
print("V1 vs V2 BY SELECTED PLAY PATTERN")
print("=" * 90)

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

    pattern_v1 = (
        v1_predictions[
            pattern_mask
        ]
    )

    pattern_v2 = (
        v2_predictions[
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
        f"Actual goal rate: "
        f"{pattern_y.mean():.4f}"
    )

    print(
        f"V1 mean xG: "
        f"{pattern_v1.mean():.4f}"
    )

    print(
        f"V2 mean xG: "
        f"{pattern_v2.mean():.4f}"
    )

    print(
        f"V1 Log Loss: "
        f"{log_loss(pattern_y, pattern_v1):.4f}"
    )

    print(
        f"V2 Log Loss: "
        f"{log_loss(pattern_y, pattern_v2):.4f}"
    )

    print(
        f"V1 Brier: "
        f"{brier_score(pattern_y, pattern_v1):.4f}"
    )

    print(
        f"V2 Brier: "
        f"{brier_score(pattern_y, pattern_v2):.4f}"
    )

    print(
        f"V1 total xG: "
        f"{pattern_v1.sum():.1f}"
    )

    print(
        f"V2 total xG: "
        f"{pattern_v2.sum():.1f}"
    )


# ==========================================================
# CALIBRATION DECILES
# ==========================================================

def calibration_table(
    name,
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

    print(
        f"\n{name}"
    )

    print(
        summary.to_string(
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


print("\n" + "=" * 90)
print("CALIBRATION DECILES")
print("=" * 90)

calibration_table(
    "V1",
    y_true,
    v1_predictions
)

calibration_table(
    "V2",
    y_true,
    v2_predictions
)


# ==========================================================
# PENALTY SUMMARY
# ==========================================================

if int(
    is_penalty.sum()
) > 0:
    penalty_mask_np = (
        is_penalty
        .to_numpy()
    )

    penalty_y = (
        y_true[
            penalty_mask_np
        ]
    )

    penalty_v1 = (
        v1_predictions[
            penalty_mask_np
        ]
    )

    penalty_v2 = (
        v2_predictions[
            penalty_mask_np
        ]
    )

    print("\n" + "=" * 90)
    print("PENALTY SUMMARY")
    print("=" * 90)

    print(
        f"Penalty shots: "
        f"{len(penalty_y):,}"
    )

    print(
        f"Penalty goals: "
        f"{int(penalty_y.sum()):,}"
    )

    print(
        f"Observed conversion: "
        f"{penalty_y.mean():.4f}"
    )

    print(
        f"V1 penalty xG: "
        f"{penalty_v1.mean():.4f}"
    )

    print(
        f"V2 penalty xG: "
        f"{penalty_v2.mean():.4f}"
    )


# ==========================================================
# FINAL SUMMARY
# ==========================================================

print("\n" + "=" * 90)
print("FINAL SUMMARY")
print("=" * 90)


if (
    delta_log_loss < 0
    and
    delta_brier < 0
    and
    delta_auc > 0
):
    print(
        "\nV2 improved all three main metrics "
        "relative to V1 on external validation."
    )

elif (
    delta_log_loss < 0
    and
    delta_brier < 0
):
    print(
        "\nV2 improved probability accuracy "
        "relative to V1, but ROC-AUC did not improve."
    )

elif (
    delta_auc > 0
    and
    (
        delta_log_loss >= 0
        or
        delta_brier >= 0
    )
):
    print(
        "\nV2 improved discrimination, "
        "but probability accuracy did not clearly improve."
    )

else:
    print(
        "\nV2 did not clearly outperform V1 "
        "on external validation."
    )


print(
    "\nNo saved model files were modified."
)

print(
    "\nV1 remains:"
)

print(
    V1_MODEL_FILE
)

print(
    "\nV2 remains:"
)

print(
    V2_MODEL_FILE
)


print("\n" + "=" * 90)
print("EXTERNAL VALIDATION COMPLETE")
print("=" * 90)