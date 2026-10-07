from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATA_PATH = Path(
    "data/processed/premier_league_2015_2016_features.csv"
)


def draw_pitch(ax):
    """
    Draw the attacking half of a 105 x 68 metre football pitch.
    """

    # Pitch limits
    ax.set_xlim(52.5, 105)
    ax.set_ylim(0, 68)

    # Halfway line
    ax.plot(
        [52.5, 52.5],
        [0, 68]
    )

    # Goal line
    ax.plot(
        [105, 105],
        [0, 68]
    )

    # Touchlines
    ax.plot(
        [52.5, 105],
        [0, 0]
    )

    ax.plot(
        [52.5, 105],
        [68, 68]
    )

    # Penalty area
    penalty_area_depth = 16.5
    penalty_area_width = 40.32

    penalty_y_min = (68 - penalty_area_width) / 2
    penalty_y_max = penalty_y_min + penalty_area_width

    ax.plot(
        [105 - penalty_area_depth, 105 - penalty_area_depth],
        [penalty_y_min, penalty_y_max]
    )

    ax.plot(
        [105 - penalty_area_depth, 105],
        [penalty_y_min, penalty_y_min]
    )

    ax.plot(
        [105 - penalty_area_depth, 105],
        [penalty_y_max, penalty_y_max]
    )

    # Six-yard box
    six_yard_depth = 5.5
    six_yard_width = 18.32

    six_y_min = (68 - six_yard_width) / 2
    six_y_max = six_y_min + six_yard_width

    ax.plot(
        [105 - six_yard_depth, 105 - six_yard_depth],
        [six_y_min, six_y_max]
    )

    ax.plot(
        [105 - six_yard_depth, 105],
        [six_y_min, six_y_min]
    )

    ax.plot(
        [105 - six_yard_depth, 105],
        [six_y_max, six_y_max]
    )

    # Goal
    goal_width = 7.32

    goal_y_min = (68 - goal_width) / 2
    goal_y_max = goal_y_min + goal_width

    ax.plot(
        [105, 106],
        [goal_y_min, goal_y_min]
    )

    ax.plot(
        [105, 106],
        [goal_y_max, goal_y_max]
    )

    ax.plot(
        [106, 106],
        [goal_y_min, goal_y_max]
    )

    # Penalty spot
    ax.scatter(
        [105 - 11],
        [34],
        s=20
    )

    ax.set_aspect("equal")

    ax.set_xlabel("Pitch X (metres)")
    ax.set_ylabel("Pitch Y (metres)")


def main():

    print("Loading processed shots...\n")

    shots = pd.read_csv(DATA_PATH)

    goals = shots[
        shots["goal"] == 1
    ]

    non_goals = shots[
        shots["goal"] == 0
    ]

    print(f"Total shots: {len(shots)}")
    print(f"Goals: {len(goals)}")
    print(f"Non-goals: {len(non_goals)}")

    fig, ax = plt.subplots(
        figsize=(12, 9)
    )

    draw_pitch(ax)

    # Non-goals
    ax.scatter(
        non_goals["x_m"],
        non_goals["y_m"],
        s=8,
        alpha=0.15,
        label="No Goal"
    )

    # Goals
    ax.scatter(
        goals["x_m"],
        goals["y_m"],
        s=20,
        alpha=0.7,
        label="Goal"
    )

    ax.set_title(
        "Premier League 2015/16 - Shot Locations"
    )

    ax.legend()

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()