from pathlib import Path

import pandas as pd


# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "multicompetition_training_shots.csv"
)


# ==========================================================
# LOAD DATA
# ==========================================================

print("=" * 70)
print("SHOT TECHNIQUE INSPECTION")
print("=" * 70)

print(f"\nLoading:\n{INPUT_FILE}\n")

df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded: {len(df):,}")
print(f"Columns: {len(df.columns):,}")


# ==========================================================
# FIND RELEVANT COLUMNS
# ==========================================================

print("\n" + "=" * 70)
print("AVAILABLE SHOT-RELATED COLUMNS")
print("=" * 70)

shot_columns = [
    col
    for col in df.columns
    if "shot" in col.lower()
    or "technique" in col.lower()
]

for col in shot_columns:
    print(col)


# ==========================================================
# DETECT TECHNIQUE COLUMN
# ==========================================================

possible_technique_columns = [
    "shot_technique",
    "technique",
    "shot.technique.name",
    "shot_technique_name",
]

technique_col = None

for col in possible_technique_columns:
    if col in df.columns:
        technique_col = col
        break


if technique_col is None:

    candidates = [
        col
        for col in df.columns
        if "technique" in col.lower()
    ]

    if len(candidates) == 1:
        technique_col = candidates[0]

    elif len(candidates) > 1:
        print("\nPossible technique columns found:")

        for col in candidates:
            print(f"  - {col}")

        raise SystemExit(
            "\nMore than one technique column was found. "
            "Check the column names above."
        )

    else:
        raise SystemExit(
            "\nNo shot technique column was found in the dataset."
        )


print("\n" + "=" * 70)
print("TECHNIQUE COLUMN")
print("=" * 70)

print(f"\nUsing column: {technique_col}")


# ==========================================================
# DETECT GOAL COLUMN
# ==========================================================

possible_goal_columns = [
    "is_goal",
    "goal",
    "shot_is_goal",
]

goal_col = None

for col in possible_goal_columns:
    if col in df.columns:
        goal_col = col
        break


# If there is no ready-made binary goal column,
# try to construct it from shot outcome.
if goal_col is None:

    possible_outcome_columns = [
        "shot_outcome",
        "shot_outcome_name",
        "outcome",
        "shot.outcome.name",
    ]

    outcome_col = None

    for col in possible_outcome_columns:
        if col in df.columns:
            outcome_col = col
            break

    if outcome_col is None:
        raise SystemExit(
            "\nCould not find either a goal column "
            "or a shot outcome column."
        )

    print(
        f"\nNo binary goal column found."
        f"\nCreating is_goal from: {outcome_col}"
    )

    df["is_goal_temp"] = (
        df[outcome_col]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("goal")
        .astype(int)
    )

    goal_col = "is_goal_temp"


print(f"Goal column: {goal_col}")


# ==========================================================
# CLEAN TECHNIQUE VALUES
# ==========================================================

df["technique_clean"] = (
    df[technique_col]
    .fillna("Missing")
    .astype(str)
    .str.strip()
)

df[goal_col] = pd.to_numeric(
    df[goal_col],
    errors="coerce"
).fillna(0)


# ==========================================================
# SUMMARY
# ==========================================================

summary = (
    df
    .groupby(
        "technique_clean",
        dropna=False
    )
    .agg(
        shots=(goal_col, "size"),
        goals=(goal_col, "sum"),
    )
    .reset_index()
)

summary["goals"] = (
    summary["goals"]
    .round()
    .astype(int)
)

summary["conversion_rate"] = (
    summary["goals"]
    / summary["shots"]
)

summary["conversion_percent"] = (
    summary["conversion_rate"]
    * 100
)

summary = (
    summary
    .sort_values(
        "shots",
        ascending=False
    )
    .reset_index(drop=True)
)


# ==========================================================
# PRINT RESULTS
# ==========================================================

print("\n" + "=" * 70)
print("SHOT TECHNIQUE SUMMARY")
print("=" * 70)

display_summary = summary.copy()

display_summary["conversion_rate"] = (
    display_summary["conversion_rate"]
    .map(lambda x: f"{x:.4f}")
)

display_summary["conversion_percent"] = (
    display_summary["conversion_percent"]
    .map(lambda x: f"{x:.2f}%")
)

print(
    display_summary.to_string(
        index=False
    )
)


# ==========================================================
# TOTALS
# ==========================================================

print("\n" + "=" * 70)
print("TOTALS")
print("=" * 70)

total_shots = len(df)

total_goals = int(
    df[goal_col].sum()
)

overall_conversion = (
    total_goals / total_shots
    if total_shots > 0
    else 0
)

print(
    f"Total shots: {total_shots:,}"
)

print(
    f"Total goals: {total_goals:,}"
)

print(
    f"Overall conversion: "
    f"{overall_conversion:.2%}"
)

print(
    f"Technique categories: "
    f"{summary['technique_clean'].nunique()}"
)


# ==========================================================
# SMALL SAMPLE WARNING
# ==========================================================

print("\n" + "=" * 70)
print("SMALL-SAMPLE TECHNIQUES")
print("=" * 70)

small_samples = summary[
    summary["shots"] < 100
]

if small_samples.empty:

    print(
        "\nNo technique has fewer than 100 shots."
    )

else:

    print(
        "\nThe following techniques have fewer than "
        "100 shots and should be treated cautiously:\n"
    )

    print(
        small_samples[
            [
                "technique_clean",
                "shots",
                "goals",
                "conversion_percent",
            ]
        ].to_string(
            index=False,
            formatters={
                "conversion_percent":
                    lambda x: f"{x:.2f}%"
            }
        )
    )


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)