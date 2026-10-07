from statsbombpy import sb


# ============================================================
# WORLD CUP SETTINGS
# ============================================================

COUNTRY = "International"
DIVISION = "FIFA World Cup"
SEASON = "2022"
GENDER = "male"


def main():

    print(
        "Loading World Cup 2022 events...\n"
    )

    events = sb.competition_events(
        country=COUNTRY,
        division=DIVISION,
        season=SEASON,
        gender=GENDER,
    )

    # ========================================================
    # SHOTS ONLY
    # ========================================================

    shots = (
        events[
            events["type"]
            == "Shot"
        ]
        .copy()
    )

    print(
        f"Total shots: "
        f"{len(shots)}"
    )

    # ========================================================
    # PENALTIES
    # ========================================================

    penalties = (
        shots[
            shots["shot_type"]
            == "Penalty"
        ]
        .copy()
    )

    print(
        f"Total penalties: "
        f"{len(penalties)}"
    )

    # ========================================================
    # PERIOD DISTRIBUTION
    # ========================================================

    print("\n")
    print("=" * 60)
    print("PENALTIES BY PERIOD")
    print("=" * 60)

    if "period" in penalties.columns:

        print(
            penalties[
                "period"
            ]
            .value_counts()
            .sort_index()
            .to_string()
        )

    else:

        print(
            "No period column found."
        )

    # ========================================================
    # OUTCOMES BY PERIOD
    # ========================================================

    if "period" in penalties.columns:

        print("\n")
        print("=" * 60)
        print("PENALTY OUTCOMES BY PERIOD")
        print("=" * 60)

        summary = (
            penalties
            .groupby(
                [
                    "period",
                    "shot_outcome",
                ],
                dropna=False
            )
            .size()
            .reset_index(
                name="shots"
            )
        )

        print(
            summary.to_string(
                index=False
            )
        )

    # ========================================================
    # SHOOTOUT PENALTIES
    # ========================================================

    if "period" in penalties.columns:

        shootout_penalties = (
            penalties[
                penalties["period"]
                == 5
            ]
            .copy()
        )

        normal_penalties = (
            penalties[
                penalties["period"]
                != 5
            ]
            .copy()
        )

        print("\n")
        print("=" * 60)
        print("NORMAL VS SHOOTOUT")
        print("=" * 60)

        print(
            f"Normal match penalties: "
            f"{len(normal_penalties)}"
        )

        print(
            f"Shootout penalties: "
            f"{len(shootout_penalties)}"
        )

        if len(normal_penalties) > 0:

            normal_goals = (
                normal_penalties[
                    "shot_outcome"
                ]
                .eq("Goal")
                .sum()
            )

            print(
                f"\nNormal penalty goals: "
                f"{normal_goals}"
            )

            print(
                f"Normal penalty conversion: "
                f"{normal_goals / len(normal_penalties):.3f}"
            )

        if len(shootout_penalties) > 0:

            shootout_goals = (
                shootout_penalties[
                    "shot_outcome"
                ]
                .eq("Goal")
                .sum()
            )

            print(
                f"\nShootout goals: "
                f"{shootout_goals}"
            )

            print(
                f"Shootout conversion: "
                f"{shootout_goals / len(shootout_penalties):.3f}"
            )

    # ========================================================
    # EXAMPLE SHOOTOUT EVENTS
    # ========================================================

    if (
        "period" in penalties.columns
    ):

        shootouts = (
            penalties[
                penalties["period"]
                == 5
            ]
            .copy()
        )

        if len(shootouts) > 0:

            print("\n")
            print("=" * 60)
            print("EXAMPLE SHOOTOUT PENALTIES")
            print("=" * 60)

            columns = [
                "match_id",
                "team",
                "player",
                "minute",
                "period",
                "shot_outcome",
                "shot_statsbomb_xg",
            ]

            columns = [
                column
                for column in columns
                if column in shootouts.columns
            ]

            print(
                shootouts[
                    columns
                ]
                .head(20)
                .to_string(
                    index=False
                )
            )


if __name__ == "__main__":
    main()