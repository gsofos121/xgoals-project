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

print("=" * 80)
print("SHOT CONTEXT INSPECTION")
print("=" * 80)

print(f"\nLoading:\n{INPUT_FILE}\n")

df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded: {len(df):,}")
print(f"Columns: {len(df.columns):,}")


# ==========================================================
# FIND RELEVANT COLUMNS
# ==========================================================

keywords = [
    "play",
    "pattern",
    "counter",
    "cross",
    "corner",
    "pass",
    "shot",
    "aerial",
    "set",
]

relevant_columns = []

for col in df.columns:
    col_lower = col.lower()

    if any(keyword in col_lower for keyword in keywords):
        relevant_columns.append(col)


print("\n" + "=" * 80)
print("RELEVANT COLUMNS")
print("=" * 80)

for col in relevant_columns:
    print(col)


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
            "\nCould not find a goal or shot outcome column."
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


print(f"\nGoal column: {goal_col}")


# ==========================================================
# HELPER
# ==========================================================

def summarize_categorical(column_name):
    if column_name not in df.columns:
        return

    print("\n" + "=" * 80)
    print(column_name.upper())
    print("=" * 80)

    temp = df.copy()

    temp[column_name] = (
        temp[column_name]
        .fillna("Missing")
        .astype(str)
        .str.strip()
    )

    summary = (
        temp
        .groupby(column_name, dropna=False)
        .agg(
            shots=(goal_col, "size"),
            goals=(goal_col, "sum"),
        )
        .reset_index()
    )

    summary["goals"] = (
        pd.to_numeric(summary["goals"], errors="coerce")
        .fillna(0)
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

    print(
        summary.to_string(
            index=False,
            formatters={
                "conversion_rate":
                    lambda x: f"{x:.4f}",
                "conversion_percent":
                    lambda x: f"{x:.2f}%"
            }
        )
    )


# ==========================================================
# PLAY PATTERN
# ==========================================================

play_pattern_candidates = [
    "play_pattern",
    "play_pattern_name",
    "pattern",
]

for col in play_pattern_candidates:
    if col in df.columns:
        summarize_categorical(col)


# ==========================================================
# SHOT TYPE
# ==========================================================

shot_type_candidates = [
    "shot_type",
    "shot_type_name",
]

for col in shot_type_candidates:
    if col in df.columns:
        summarize_categorical(col)


# ==========================================================
# BODY PART
# ==========================================================

body_part_candidates = [
    "shot_body_part",
    "shot_body_part_name",
    "body_part",
]

for col in body_part_candidates:
    if col in df.columns:
        summarize_categorical(col)


# ==========================================================
# TECHNIQUE
# ==========================================================

technique_candidates = [
    "shot_technique",
    "shot_technique_name",
    "technique",
]

for col in technique_candidates:
    if col in df.columns:
        summarize_categorical(col)


# ==========================================================
# BOOLEAN / FLAG INSPECTION
# ==========================================================

flag_keywords = [
    "cross",
    "counter",
    "corner",
    "aerial",
    "first_time",
    "one_on_one",
    "open_goal",
    "under_pressure",
]

flag_columns = []

for col in df.columns:
    col_lower = col.lower()

    if any(keyword in col_lower for keyword in flag_keywords):
        flag_columns.append(col)


print("\n" + "=" * 80)
print("POSSIBLE FLAG COLUMNS")
print("=" * 80)

if not flag_columns:
    print("\nNo obvious flag columns found.")

else:
    for col in flag_columns:
        print(f"\n--- {col} ---")

        values = (
            df[col]
            .fillna("Missing")
            .astype(str)
            .value_counts(dropna=False)
        )

        print(values.to_string())


# ==========================================================
# MISSING VALUES
# ==========================================================

print("\n" + "=" * 80)
print("MISSING VALUES IN RELEVANT COLUMNS")
print("=" * 80)

for col in relevant_columns:

    missing_count = df[col].isna().sum()

    if missing_count > 0:
        missing_percent = (
            missing_count
            / len(df)
            * 100
        )

        print(
            f"{col}: "
            f"{missing_count:,} "
            f"({missing_percent:.2f}%)"
        )


# ==========================================================
# FINAL SEARCH FOR IMPORTANT WORDS IN VALUES
# ==========================================================

search_terms = [
    "corner",
    "counter",
    "cross",
    "throw",
    "free kick",
    "regular",
]

print("\n" + "=" * 80)
print("SEARCH FOR IMPORTANT TERMS INSIDE COLUMN VALUES")
print("=" * 80)

for col in df.columns:

    if df[col].dtype != "object":
        continue

    text_values = (
        df[col]
        .dropna()
        .astype(str)
        .str.lower()
    )

    matches_found = []

    for term in search_terms:

        count = (
            text_values
            .str.contains(
                term,
                regex=False
            )
            .sum()
        )

        if count > 0:
            matches_found.append(
                (term, count)
            )

    if matches_found:

        print(f"\nColumn: {col}")

        for term, count in matches_found:
            print(
                f"  {term}: {count:,}"
            )


# ==========================================================
# TOTALS
# ==========================================================

print("\n" + "=" * 80)
print("TOTALS")
print("=" * 80)

total_shots = len(df)
total_goals = int(
    pd.to_numeric(
        df[goal_col],
        errors="coerce"
    )
    .fillna(0)
    .sum()
)

print(f"Total shots: {total_shots:,}")
print(f"Total goals: {total_goals:,}")

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)