// ==========================================================
// API
// ==========================================================

const API_URL =
    "/predict";


// ==========================================================
// PITCH CONSTANTS
// ==========================================================

const PITCH_LENGTH_M = 105;
const PITCH_WIDTH_M = 68;

const ATTACKING_HALF_START_X = 52.5;

const GOAL_X_M = 105;
const GOAL_CENTER_Y_M = 34;
const GOAL_WIDTH_M = 7.32;

const PENALTY_X_M = 94;
const PENALTY_Y_M = 34;

const LEFT_POST_Y_M =
    GOAL_CENTER_Y_M
    - GOAL_WIDTH_M / 2;

const RIGHT_POST_Y_M =
    GOAL_CENTER_Y_M
    + GOAL_WIDTH_M / 2;

const SVG_SCALE = 10;


// ==========================================================
// HTML ELEMENTS
// ==========================================================

const pitch =
    document.getElementById(
        "pitch"
    );

const shotMarker =
    document.getElementById(
        "shot-marker"
    );

const angleArea =
    document.getElementById(
        "angle-area"
    );

const angleLineLeft =
    document.getElementById(
        "angle-line-left"
    );

const angleLineRight =
    document.getElementById(
        "angle-line-right"
    );


const xValue =
    document.getElementById(
        "x-value"
    );

const yValue =
    document.getElementById(
        "y-value"
    );

const distanceValue =
    document.getElementById(
        "distance-value"
    );

const angleValue =
    document.getElementById(
        "angle-value"
    );


const underPressure =
    document.getElementById(
        "under-pressure"
    );

const firstTime =
    document.getElementById(
        "first-time"
    );

const oneOnOne =
    document.getElementById(
        "one-on-one"
    );

const openGoal =
    document.getElementById(
        "open-goal"
    );


const predictButton =
    document.getElementById(
        "predict-button"
    );

const errorMessage =
    document.getElementById(
        "error-message"
    );


const resultCard =
    document.getElementById(
        "result-card"
    );

const xgValue =
    document.getElementById(
        "xg-value"
    );

const probabilityValue =
    document.getElementById(
        "probability-value"
    );

const resultDistance =
    document.getElementById(
        "result-distance"
    );

const resultAngle =
    document.getElementById(
        "result-angle"
    );

const predictionType =
    document.getElementById(
        "prediction-type"
    );


const bodyPartInputs =
    document.querySelectorAll(
        'input[name="body-part"]'
    );

const shotTypeInputs =
    document.querySelectorAll(
        'input[name="shot-type"]'
    );

const chanceOriginInputs =
    document.querySelectorAll(
        'input[name="chance-origin"]'
    );


// ==========================================================
// CURRENT SHOT
// ==========================================================

let selectedShot = null;


// ==========================================================
// RADIO HELPERS
// ==========================================================

function getRadioValue(
    name
) {

    const selected =
        document.querySelector(
            `input[name="${name}"]:checked`
        );


    return selected
        ? selected.value
        : null;
}


function setRadioValue(
    name,
    value
) {

    const inputs =
        document.querySelectorAll(
            `input[name="${name}"]`
        );


    const input =
        Array.from(
            inputs
        )
        .find(
            item =>
                item.value
                ===
                value
        );


    if (input) {

        input.checked =
            true;
    }
}


function setRadioOptionDisabled(
    name,
    value,
    disabled
) {

    const input =
        Array.from(
            document.querySelectorAll(
                `input[name="${name}"]`
            )
        )
        .find(
            item =>
                item.value
                ===
                value
        );


    if (!input) {

        return;
    }


    input.disabled =
        disabled;


    const wrapper =
        input.closest(
            ".radio-option"
        );


    if (wrapper) {

        wrapper.classList.toggle(
            "disabled-option",
            disabled
        );
    }
}


function setWholeRadioGroupDisabled(
    name,
    disabled
) {

    const inputs =
        document.querySelectorAll(
            `input[name="${name}"]`
        );


    inputs.forEach(
        input => {

            input.disabled =
                disabled;


            const wrapper =
                input.closest(
                    ".radio-option"
                );


            if (wrapper) {

                wrapper.classList.toggle(
                    "disabled-option",
                    disabled
                );
            }
        }
    );
}


// ==========================================================
// CHECKBOX HELPERS
// ==========================================================

function setCheckboxState(
    checkbox,
    disabled,
    checked = false
) {

    checkbox.disabled =
        disabled;


    if (disabled) {

        checkbox.checked =
            checked;
    }


    const row =
        checkbox.closest(
            ".checkbox-row, .context-option"
        );


    if (row) {

        row.classList.toggle(
            "disabled-option",
            disabled
        );
    }
}


function enableAllCheckboxes() {

    setCheckboxState(
        underPressure,
        false
    );

    setCheckboxState(
        firstTime,
        false
    );

    setCheckboxState(
        oneOnOne,
        false
    );

    setCheckboxState(
        openGoal,
        false
    );
}


// ==========================================================
// GEOMETRY
// ==========================================================

function calculateDistance(
    x,
    y
) {

    const dx =
        GOAL_X_M
        - x;


    const dy =
        GOAL_CENTER_Y_M
        - y;


    return Math.sqrt(
        dx ** 2
        +
        dy ** 2
    );
}


function calculateAngle(
    x,
    y
) {

    const v1x =
        GOAL_X_M
        - x;


    const v1y =
        LEFT_POST_Y_M
        - y;


    const v2x =
        GOAL_X_M
        - x;


    const v2y =
        RIGHT_POST_Y_M
        - y;


    const dot =
        v1x * v2x
        +
        v1y * v2y;


    const magnitude1 =
        Math.sqrt(
            v1x ** 2
            +
            v1y ** 2
        );


    const magnitude2 =
        Math.sqrt(
            v2x ** 2
            +
            v2y ** 2
        );


    if (
        magnitude1 === 0
        ||
        magnitude2 === 0
    ) {

        return 180;
    }


    let cosine =
        dot
        /
        (
            magnitude1
            *
            magnitude2
        );


    cosine =
        Math.max(
            -1,
            Math.min(
                1,
                cosine
            )
        );


    return (
        Math.acos(
            cosine
        )
        *
        180
    ) / Math.PI;
}


// ==========================================================
// SVG COORDINATES
// ==========================================================

function getSVGCoordinates(
    event
) {

    const point =
        pitch.createSVGPoint();


    point.x =
        event.clientX;


    point.y =
        event.clientY;


    const screenCTM =
        pitch.getScreenCTM();


    if (!screenCTM) {

        return null;
    }


    return point.matrixTransform(
        screenCTM.inverse()
    );
}


// ==========================================================
// SVG -> REAL PITCH
//
// Top = goal
// Bottom = halfway line
// ==========================================================

function svgToPitchCoordinates(
    svgX,
    svgY
) {

    let y =
        svgX
        /
        SVG_SCALE;


    let x =
        GOAL_X_M
        -
        (
            svgY
            /
            SVG_SCALE
        );


    x =
        Math.max(
            ATTACKING_HALF_START_X,
            Math.min(
                GOAL_X_M,
                x
            )
        );


    y =
        Math.max(
            0,
            Math.min(
                PITCH_WIDTH_M,
                y
            )
        );


    return {
        x,
        y
    };
}


// ==========================================================
// REAL PITCH -> SVG
// ==========================================================

function pitchToSVGCoordinates(
    x,
    y
) {

    return {

        svgX:
            y
            *
            SVG_SCALE,

        svgY:
            (
                GOAL_X_M
                -
                x
            )
            *
            SVG_SCALE
    };
}


// ==========================================================
// VISUAL GOAL ANGLE
// ==========================================================

function updateVisualAngle(
    svgX,
    svgY
) {

    const leftPostX =
        LEFT_POST_Y_M
        *
        SVG_SCALE;


    const rightPostX =
        RIGHT_POST_Y_M
        *
        SVG_SCALE;


    const goalLineY =
        0;


    angleArea.setAttribute(
        "points",
        `${svgX},${svgY} ${leftPostX},${goalLineY} ${rightPostX},${goalLineY}`
    );


    angleLineLeft.setAttribute(
        "x1",
        svgX
    );


    angleLineLeft.setAttribute(
        "y1",
        svgY
    );


    angleLineLeft.setAttribute(
        "x2",
        leftPostX
    );


    angleLineLeft.setAttribute(
        "y2",
        goalLineY
    );


    angleLineRight.setAttribute(
        "x1",
        svgX
    );


    angleLineRight.setAttribute(
        "y1",
        svgY
    );


    angleLineRight.setAttribute(
        "x2",
        rightPostX
    );


    angleLineRight.setAttribute(
        "y2",
        goalLineY
    );


    angleArea.classList.remove(
        "hidden"
    );


    angleLineLeft.classList.remove(
        "hidden"
    );


    angleLineRight.classList.remove(
        "hidden"
    );
}


// ==========================================================
// SHOT MARKER
// ==========================================================

function updateShotMarker(
    svgX,
    svgY
) {

    shotMarker.setAttribute(
        "cx",
        svgX
    );


    shotMarker.setAttribute(
        "cy",
        svgY
    );


    shotMarker.classList.remove(
        "hidden"
    );


    updateVisualAngle(
        svgX,
        svgY
    );
}


function moveMarkerToPitchPosition(
    x,
    y
) {

    const coordinates =
        pitchToSVGCoordinates(
            x,
            y
        );


    updateShotMarker(
        coordinates.svgX,
        coordinates.svgY
    );


    updatePositionInfo(
        x,
        y
    );
}


// ==========================================================
// POSITION INFO
// ==========================================================

function updatePositionInfo(
    x,
    y
) {

    const distance =
        calculateDistance(
            x,
            y
        );


    const angle =
        calculateAngle(
            x,
            y
        );


    xValue.textContent =
        x.toFixed(
            2
        );


    yValue.textContent =
        y.toFixed(
            2
        );


    distanceValue.textContent =
        distance.toFixed(
            2
        );


    angleValue.textContent =
        angle.toFixed(
            2
        );


    selectedShot = {
        x,
        y,
        distance,
        angle
    };


    predictButton.disabled =
        false;
}


// ==========================================================
// RESULT HELPERS
// ==========================================================

function clearResult() {

    resultCard.classList.add(
        "hidden"
    );


    errorMessage.classList.add(
        "hidden"
    );
}


function showError(
    message
) {

    errorMessage.textContent =
        message;


    errorMessage.classList.remove(
        "hidden"
    );


    resultCard.classList.add(
        "hidden"
    );
}


function showResult(
    result
) {

    errorMessage.classList.add(
        "hidden"
    );


    // ======================================================
    // xG VALUE
    // ======================================================

    xgValue.textContent =
        Number(
            result.xg
        )
        .toFixed(
            3
        );


    // ======================================================
    // GOAL PROBABILITY
    // ======================================================

    const probabilityPercent =
        result.xg_percent
        ??
        (
            Number(
                result.xg
            )
            *
            100
        );


    probabilityValue.textContent =
        `${Number(
            probabilityPercent
        ).toFixed(1)}% chance of goal`;


    // ======================================================
    // GEOMETRY
    // ======================================================

    resultDistance.textContent =
        `${Number(
            result.distance_m
        ).toFixed(2)} m`;


    resultAngle.textContent =
        `${Number(
            result.angle_deg
        ).toFixed(2)}°`;


    // ======================================================
    // PREDICTION SOURCE
    // ======================================================

    const source =
        result.prediction_source
        ??
        "";


    if (
        source
        ===
        "fixed_penalty_probability"
    ) {

        predictionType.textContent =
            "Fixed penalty xG";
    }


    else {

        predictionType.textContent =
            "Pre-shot Enhanced Logistic V3";
    }


    resultCard.classList.remove(
        "hidden"
    );
}


// ==========================================================
// CONTEXT COMPATIBILITY
// ==========================================================

function applyContextCompatibility() {

    if (
        getRadioValue(
            "shot-type"
        )
        !==
        "Open Play"
    ) {

        return;
    }


    // ------------------------------------------------------
    // Reset availability first
    // ------------------------------------------------------

    setCheckboxState(
        underPressure,
        false
    );


    setCheckboxState(
        oneOnOne,
        false
    );


    setCheckboxState(
        openGoal,
        false
    );


    // ------------------------------------------------------
    // Open Goal
    //
    // Incompatible with:
    // Under Pressure
    // One-on-One
    // ------------------------------------------------------

    if (
        openGoal.checked
    ) {

        underPressure.checked =
            false;


        oneOnOne.checked =
            false;


        setCheckboxState(
            underPressure,
            true,
            false
        );


        setCheckboxState(
            oneOnOne,
            true,
            false
        );


        return;
    }


    // ------------------------------------------------------
    // One-on-One
    // ------------------------------------------------------

    if (
        oneOnOne.checked
    ) {

        openGoal.checked =
            false;


        setCheckboxState(
            openGoal,
            true,
            false
        );
    }


    // ------------------------------------------------------
    // Under Pressure
    // ------------------------------------------------------

    if (
        underPressure.checked
    ) {

        openGoal.checked =
            false;


        setCheckboxState(
            openGoal,
            true,
            false
        );
    }
}


// ==========================================================
// SHOT TYPE RULES
// ==========================================================

function applyShotTypeRules() {

    const type =
        getRadioValue(
            "shot-type"
        );


    clearResult();


    // ======================================================
    // GENERIC RESET
    // ======================================================

    pitch.classList.remove(
        "pitch-locked"
    );


    setWholeRadioGroupDisabled(
        "body-part",
        false
    );


    setRadioOptionDisabled(
        "body-part",
        "Head",
        false
    );


    setWholeRadioGroupDisabled(
        "chance-origin",
        false
    );


    enableAllCheckboxes();


    // ======================================================
    // PENALTY
    // ======================================================

    if (
        type
        ===
        "Penalty"
    ) {

        moveMarkerToPitchPosition(
            PENALTY_X_M,
            PENALTY_Y_M
        );


        setRadioValue(
            "body-part",
            "Right Foot"
        );


        setWholeRadioGroupDisabled(
            "body-part",
            true
        );


        setWholeRadioGroupDisabled(
            "chance-origin",
            true
        );


        setCheckboxState(
            underPressure,
            true,
            false
        );


        setCheckboxState(
            firstTime,
            true,
            false
        );


        setCheckboxState(
            oneOnOne,
            true,
            false
        );


        setCheckboxState(
            openGoal,
            true,
            false
        );


        pitch.classList.add(
            "pitch-locked"
        );


        return;
    }


    // ======================================================
    // FREE KICK
    // ======================================================

    if (
        type
        ===
        "Free Kick"
    ) {

        setRadioOptionDisabled(
            "body-part",
            "Head",
            true
        );


        if (
            getRadioValue(
                "body-part"
            )
            ===
            "Head"
        ) {

            setRadioValue(
                "body-part",
                "Right Foot"
            );
        }


        setWholeRadioGroupDisabled(
            "chance-origin",
            true
        );


        setCheckboxState(
            underPressure,
            true,
            false
        );


        setCheckboxState(
            firstTime,
            true,
            false
        );


        setCheckboxState(
            oneOnOne,
            true,
            false
        );


        setCheckboxState(
            openGoal,
            true,
            false
        );


        return;
    }


    // ======================================================
    // OPEN PLAY
    // ======================================================

    setRadioOptionDisabled(
        "body-part",
        "Head",
        false
    );


    setWholeRadioGroupDisabled(
        "chance-origin",
        false
    );


    applyContextCompatibility();
}


// ==========================================================
// PITCH CLICK
// ==========================================================

pitch.addEventListener(
    "click",
    function (
        event
    ) {

        if (
            getRadioValue(
                "shot-type"
            )
            ===
            "Penalty"
        ) {

            showError(
                "Penalty location is fixed at the penalty spot."
            );


            return;
        }


        const point =
            getSVGCoordinates(
                event
            );


        if (!point) {

            return;
        }


        if (
            point.x < 0
            ||
            point.x > 680
            ||
            point.y < 0
            ||
            point.y > 525
        ) {

            return;
        }


        const coordinates =
            svgToPitchCoordinates(
                point.x,
                point.y
            );


        updateShotMarker(
            point.x,
            point.y
        );


        updatePositionInfo(
            coordinates.x,
            coordinates.y
        );


        clearResult();
    }
);


// ==========================================================
// REQUEST BODY
//
// IMPORTANT:
// Field names must match backend/app.py.
// ==========================================================

function buildRequestBody() {

    const shotType =
        getRadioValue(
            "shot-type"
        );


    const requestBody = {

        x:
            selectedShot.x,

        y:
            selectedShot.y,

        body_part:
            getRadioValue(
                "body-part"
            ),

        shot_type:
            shotType,

        under_pressure:
            underPressure.checked,

        shot_first_time:
            firstTime.checked,

        shot_one_on_one:
            oneOnOne.checked,

        shot_open_goal:
            openGoal.checked
    };


    // Chance origin only applies
    // to Open Play.

    if (
        shotType
        ===
        "Open Play"
    ) {

        requestBody.chance_origin =
            getRadioValue(
                "chance-origin"
            );
    }


    return requestBody;
}


// ==========================================================
// API CALL
// ==========================================================

async function calculateXG() {

    if (!selectedShot) {

        showError(
            "Select a shot location first."
        );


        return;
    }


    const body =
        buildRequestBody();


    errorMessage.classList.add(
        "hidden"
    );


    predictButton.disabled =
        true;


    const originalButtonText =
        predictButton.textContent;


    predictButton.textContent =
        "Calculating...";


    try {

        const response =
            await fetch(
                API_URL,
                {

                    method:
                        "POST",

                    headers: {

                        "Content-Type":
                            "application/json"
                    },

                    body:
                        JSON.stringify(
                            body
                        )
                }
            );


        let result;


        try {

            result =
                await response.json();
        }


        catch (
            jsonError
        ) {

            throw new Error(
                "The backend returned an invalid response."
            );
        }


        if (
            !response.ok
        ) {

            throw new Error(
                result.error
                ||
                "Prediction failed."
            );
        }


        if (
            result.success
            ===
            false
        ) {

            throw new Error(
                result.error
                ||
                "Prediction failed."
            );
        }


        showResult(
            result
        );
    }


    catch (
        error
    ) {

        console.error(
            "xG prediction error:",
            error
        );


        showError(
            error.message
            ||
            "Could not connect to the xG API. Make sure the backend is running."
        );
    }


    finally {

        predictButton.disabled =
            false;


        predictButton.textContent =
            originalButtonText;
    }
}


// ==========================================================
// EVENTS
// ==========================================================

predictButton.addEventListener(
    "click",
    calculateXG
);


// ----------------------------------------------------------
// Body part
// ----------------------------------------------------------

bodyPartInputs.forEach(
    input => {

        input.addEventListener(
            "change",
            clearResult
        );
    }
);


// ----------------------------------------------------------
// Shot type
// ----------------------------------------------------------

shotTypeInputs.forEach(
    input => {

        input.addEventListener(
            "change",
            applyShotTypeRules
        );
    }
);


// ----------------------------------------------------------
// Chance origin
// ----------------------------------------------------------

chanceOriginInputs.forEach(
    input => {

        input.addEventListener(
            "change",
            clearResult
        );
    }
);


// ----------------------------------------------------------
// Under Pressure
// ----------------------------------------------------------

underPressure.addEventListener(
    "change",
    function () {

        applyContextCompatibility();

        clearResult();
    }
);


// ----------------------------------------------------------
// First Time
// ----------------------------------------------------------

firstTime.addEventListener(
    "change",
    clearResult
);


// ----------------------------------------------------------
// One-on-One
// ----------------------------------------------------------

oneOnOne.addEventListener(
    "change",
    function () {

        applyContextCompatibility();

        clearResult();
    }
);


// ----------------------------------------------------------
// Open Goal
// ----------------------------------------------------------

openGoal.addEventListener(
    "change",
    function () {

        applyContextCompatibility();

        clearResult();
    }
);


// ==========================================================
// INITIAL STATE
// ==========================================================

predictButton.disabled =
    true;


applyShotTypeRules();