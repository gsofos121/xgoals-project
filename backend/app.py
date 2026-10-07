from pathlib import Path
import json
import math

import numpy as np

from flask import Flask, jsonify, request
from flask_cors import CORS


# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FRONTEND_DIR = (
    PROJECT_ROOT
    / "frontend"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "xg_multicompetition_logistic_v3.json"
)


# ==========================================================
# PITCH CONSTANTS
# ==========================================================

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0

GOAL_X = 105.0
GOAL_CENTER_Y = 34.0
GOAL_WIDTH = 7.32

LEFT_POST_Y = (
    GOAL_CENTER_Y
    - GOAL_WIDTH / 2
)

RIGHT_POST_Y = (
    GOAL_CENTER_Y
    + GOAL_WIDTH / 2
)


# ==========================================================
# LOAD V3 MODEL
# ==========================================================

if not MODEL_FILE.exists():

    raise FileNotFoundError(
        f"V3 model file not found: {MODEL_FILE}"
    )


with open(
    MODEL_FILE,
    "r",
    encoding="utf-8"
) as file:

    MODEL = json.load(file)


MODEL_NAME = MODEL.get(
    "model_name",
    "multi_competition_xg_v3_enhanced_logistic"
)


MODEL_TYPE = MODEL.get(
    "model_type",
    "enhanced_logistic_regression"
)


# ==========================================================
# NON-PENALTY MODEL
# ==========================================================

NON_PENALTY_MODEL = MODEL[
    "non_penalty_model"
]


MODEL_FEATURES = list(
    NON_PENALTY_MODEL[
        "features"
    ]
)


MODEL_INTERCEPT = float(
    NON_PENALTY_MODEL[
        "intercept"
    ]
)


MODEL_COEFFICIENTS = {

    feature_name:
        float(coefficient)

    for (
        feature_name,
        coefficient
    ) in NON_PENALTY_MODEL[
        "coefficients"
    ].items()
}


# ==========================================================
# PENALTY RULE
# ==========================================================

PENALTY_XG = float(
    MODEL[
        "penalty_rule"
    ][
        "xg"
    ]
)


# ==========================================================
# FLASK
#
# static_folder points to frontend/
#
# This means:
#
# /             -> index.html
# /styles.css   -> frontend/styles.css
# /app.js       -> frontend/app.js
#
# /predict and /health remain backend routes.
# ==========================================================

app = Flask(
    __name__,
    static_folder=str(
        FRONTEND_DIR
    ),
    static_url_path=""
)

CORS(app)


# ==========================================================
# HELPERS
# ==========================================================

def safe_float(
    value,
    default=0.0
):

    try:

        return float(value)

    except (
        TypeError,
        ValueError
    ):

        return float(default)


def safe_bool(
    value
):

    if isinstance(
        value,
        bool
    ):

        return int(value)


    if isinstance(
        value,
        (
            int,
            float
        )
    ):

        return int(
            value != 0
        )


    value = str(
        value
    ).strip().lower()


    return int(
        value
        in {
            "true",
            "1",
            "yes",
            "on"
        }
    )


def sigmoid(
    value
):

    value = np.clip(
        value,
        -35,
        35
    )


    return float(
        1.0
        /
        (
            1.0
            + np.exp(
                -value
            )
        )
    )


# ==========================================================
# DISTANCE
# ==========================================================

def calculate_distance(
    x,
    y
):

    dx = (
        GOAL_X
        - x
    )

    dy = (
        GOAL_CENTER_Y
        - y
    )


    return math.sqrt(
        dx ** 2
        +
        dy ** 2
    )


# ==========================================================
# GOAL ANGLE
# ==========================================================

def calculate_goal_angle(
    x,
    y
):

    vector_left = (
        GOAL_X - x,
        LEFT_POST_Y - y
    )

    vector_right = (
        GOAL_X - x,
        RIGHT_POST_Y - y
    )


    dot_product = (
        vector_left[0]
        * vector_right[0]
        +
        vector_left[1]
        * vector_right[1]
    )


    magnitude_left = math.sqrt(
        vector_left[0] ** 2
        +
        vector_left[1] ** 2
    )


    magnitude_right = math.sqrt(
        vector_right[0] ** 2
        +
        vector_right[1] ** 2
    )


    denominator = (
        magnitude_left
        *
        magnitude_right
    )


    if denominator == 0:

        return 180.0


    cosine_angle = (
        dot_product
        /
        denominator
    )


    cosine_angle = max(
        -1.0,
        min(
            1.0,
            cosine_angle
        )
    )


    angle_radians = math.acos(
        cosine_angle
    )


    return math.degrees(
        angle_radians
    )


# ==========================================================
# FEATURE BUILDER
# ==========================================================

def build_features(
    payload
):

    # ------------------------------------------------------
    # Position
    # ------------------------------------------------------

    x = safe_float(
        payload.get(
            "x"
        )
    )

    y = safe_float(
        payload.get(
            "y"
        )
    )


    # ------------------------------------------------------
    # Geometry
    # ------------------------------------------------------

    distance_m = calculate_distance(
        x,
        y
    )

    angle_deg = calculate_goal_angle(
        x,
        y
    )


    # ------------------------------------------------------
    # Shot conditions
    # ------------------------------------------------------

    under_pressure = safe_bool(
        payload.get(
            "under_pressure",
            False
        )
    )


    shot_first_time = safe_bool(
        payload.get(
            "shot_first_time",
            False
        )
    )


    shot_one_on_one = safe_bool(
        payload.get(
            "shot_one_on_one",
            False
        )
    )


    shot_open_goal = safe_bool(
        payload.get(
            "shot_open_goal",
            False
        )
    )


    # ------------------------------------------------------
    # Body part
    # ------------------------------------------------------

    body_part = str(
        payload.get(
            "body_part",
            "Right Foot"
        )
    ).strip()


    body_left_foot = int(
        body_part
        ==
        "Left Foot"
    )


    body_head = int(
        body_part
        ==
        "Head"
    )


    # ------------------------------------------------------
    # Shot type
    # ------------------------------------------------------

    shot_type = str(
        payload.get(
            "shot_type",
            "Open Play"
        )
    ).strip()


    type_free_kick = int(
        shot_type
        ==
        "Free Kick"
    )


    # ------------------------------------------------------
    # Chance origin
    # ------------------------------------------------------

    chance_origin = str(
        payload.get(
            "chance_origin",
            "Regular Play"
        )
    ).strip()


    play_regular = 0
    play_counter = 0
    play_corner = 0


    if shot_type == "Open Play":

        if (
            chance_origin
            ==
            "Regular Play"
        ):

            play_regular = 1


        elif (
            chance_origin
            ==
            "Counterattack"
        ):

            play_counter = 1


        elif (
            chance_origin
            ==
            "Corner"
        ):

            play_corner = 1


    # ======================================================
    # 12 BASE FEATURES
    # ======================================================

    features = {

        "distance_m":
            distance_m,

        "angle_deg":
            angle_deg,

        "under_pressure":
            under_pressure,

        "shot_first_time":
            shot_first_time,

        "shot_one_on_one":
            shot_one_on_one,

        "shot_open_goal":
            shot_open_goal,

        "body_left_foot":
            body_left_foot,

        "body_head":
            body_head,

        "type_free_kick":
            type_free_kick,

        "play_regular":
            play_regular,

        "play_counter":
            play_counter,

        "play_corner":
            play_corner,
    }


    # ======================================================
    # 6 V3 ENGINEERED FEATURES
    # ======================================================

    features[
        "distance_sq"
    ] = (
        distance_m ** 2
    )


    features[
        "angle_sq"
    ] = (
        angle_deg ** 2
    )


    features[
        "distance_angle"
    ] = (
        distance_m
        *
        angle_deg
    )


    features[
        "distance_head"
    ] = (
        distance_m
        *
        body_head
    )


    features[
        "counter_one_on_one"
    ] = (
        play_counter
        *
        shot_one_on_one
    )


    features[
        "corner_head"
    ] = (
        play_corner
        *
        body_head
    )


    return (
        features,
        shot_type
    )


# ==========================================================
# V3 LOGISTIC PREDICTION
# ==========================================================

def predict_non_penalty(
    features
):

    linear_score = (
        MODEL_INTERCEPT
    )


    for feature_name in MODEL_FEATURES:

        if (
            feature_name
            not in MODEL_COEFFICIENTS
        ):

            raise KeyError(
                f"Missing coefficient for feature: "
                f"{feature_name}"
            )


        feature_value = float(
            features.get(
                feature_name,
                0.0
            )
        )


        coefficient = (
            MODEL_COEFFICIENTS[
                feature_name
            ]
        )


        linear_score += (
            coefficient
            *
            feature_value
        )


    return sigmoid(
        linear_score
    )


# ==========================================================
# FRONTEND HOME
# ==========================================================

@app.route(
    "/",
    methods=[
        "GET"
    ]
)
def frontend_home():

    return app.send_static_file(
        "index.html"
    )


# ==========================================================
# HEALTH
# ==========================================================

@app.route(
    "/health",
    methods=[
        "GET"
    ]
)
def health():

    return jsonify(
        {

            "status":
                "healthy",

            "model_loaded":
                True,

            "model_name":
                MODEL_NAME,

            "model_type":
                "Enhanced Logistic Regression V3",

            "feature_count":
                len(
                    MODEL_FEATURES
                ),

            "penalty_xg":
                PENALTY_XG,
        }
    )


# ==========================================================
# MODEL INFO
# ==========================================================

@app.route(
    "/model-info",
    methods=[
        "GET"
    ]
)
def model_info():

    return jsonify(
        {

            "model_name":
                MODEL_NAME,

            "model_type":
                "Enhanced Logistic Regression V3",

            "features":
                MODEL_FEATURES,

            "feature_count":
                len(
                    MODEL_FEATURES
                ),

            "penalty_xg":
                PENALTY_XG,
        }
    )


# ==========================================================
# PREDICT
# ==========================================================

@app.route(
    "/predict",
    methods=[
        "POST"
    ]
)
def predict():

    try:

        payload = (
            request.get_json(
                silent=True
            )
            or
            {}
        )


        features, shot_type = build_features(
            payload
        )


        # ==================================================
        # PENALTY
        # ==================================================

        if (
            shot_type
            ==
            "Penalty"
        ):

            xg = (
                PENALTY_XG
            )


            prediction_source = (
                "fixed_penalty_probability"
            )


        # ==================================================
        # NON-PENALTY
        # ==================================================

        else:

            xg = predict_non_penalty(
                features
            )


            prediction_source = (
                "enhanced_logistic_v3"
            )


        # --------------------------------------------------
        # Safety clamp
        # --------------------------------------------------

        xg = max(
            0.0,
            min(
                1.0,
                float(
                    xg
                )
            )
        )


        return jsonify(
            {

                "success":
                    True,

                "model":
                    MODEL_NAME,

                "model_type":
                    "Enhanced Logistic Regression V3",

                "prediction_source":
                    prediction_source,

                "xg":
                    xg,

                "xg_percent":
                    xg * 100,

                "distance_m":
                    features[
                        "distance_m"
                    ],

                "angle_deg":
                    features[
                        "angle_deg"
                    ],

                "features":
                    {

                        feature_name:
                            features.get(
                                feature_name,
                                0.0
                            )

                        for feature_name
                        in MODEL_FEATURES
                    },
            }
        )


    except Exception as error:

        print(
            "Prediction error:",
            repr(
                error
            )
        )


        return jsonify(
            {

                "success":
                    False,

                "error":
                    str(
                        error
                    ),
            }
        ), 500


# ==========================================================
# LOCAL SERVER
# ==========================================================

if __name__ == "__main__":

    print("=" * 80)

    print(
        "xG BACKEND + FRONTEND"
    )

    print("=" * 80)

    print(
        f"Model: "
        f"{MODEL_NAME}"
    )

    print(
        "Type: "
        "Enhanced Logistic Regression V3"
    )

    print(
        f"Features: "
        f"{len(MODEL_FEATURES)}"
    )

    print(
        f"Penalty xG: "
        f"{PENALTY_XG:.6f}"
    )

    print(
        f"Frontend directory: "
        f"{FRONTEND_DIR}"
    )

    print(
        f"Model file: "
        f"{MODEL_FILE}"
    )

    print("=" * 80)


    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )