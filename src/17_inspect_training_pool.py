from statsbombpy import sb


# ============================================================
# CANDIDATE TRAINING DATASETS
# ============================================================

CANDIDATES = [
    {
        "name": "Premier League 2015/16",
        "competition_id": 2,
        "season_id": 27,
    },
    {
        "name": "La Liga 2015/16",
        "competition_id": 11,
        "season_id": 27,
    },
    {
        "name": "Ligue 1 2015/16",
        "competition_id": 7,
        "season_id": 27,
    },
    {
        "name": "Bundesliga 2015/16",
        "competition_id": 9,
        "season_id": 27,
    },
    {
        "name": "Serie A 2015/16",
        "competition_id": 12,
        "season_id": 27,
    },
    {
        "name": "Champions League 2015/16",
        "competition_id": 16,
        "season_id": 27,
    },
    {
        "name": "World Cup 2018",
        "competition_id": 43,
        "season_id": 3,
    },
    {
        "name": "Euro 2020",
        "competition_id": 55,
        "season_id": 43,
    },
    {
        "name": "World Cup 2022",
        "competition_id": 43,
        "season_id": 106,
    },
    {
        "name": "Copa America 2024",
        "competition_id": 223,
        "season_id": 282,
    },
    {
        "name": "Euro 2024",
        "competition_id": 55,
        "season_id": 282,
    },
]


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "INSPECTING POSSIBLE TRAINING DATASETS\n"
    )

    print(
        "=" * 100
    )

    print(
        f"{'Competition':<30}"
        f"{'Matches':>10}"
        f"{'Teams':>10}"
        f"{'First Date':>15}"
        f"{'Last Date':>15}"
    )

    print(
        "=" * 100
    )

    total_matches = 0

    for candidate in CANDIDATES:

        try:

            matches = sb.matches(
                competition_id=candidate[
                    "competition_id"
                ],
                season_id=candidate[
                    "season_id"
                ],
            )

            match_count = len(
                matches
            )

            teams = set(
                matches[
                    "home_team"
                ]
            ).union(
                set(
                    matches[
                        "away_team"
                    ]
                )
            )

            team_count = len(
                teams
            )

            if match_count > 0:

                first_date = (
                    matches[
                        "match_date"
                    ]
                    .min()
                )

                last_date = (
                    matches[
                        "match_date"
                    ]
                    .max()
                )

            else:

                first_date = "-"
                last_date = "-"

            print(
                f"{candidate['name']:<30}"
                f"{match_count:>10}"
                f"{team_count:>10}"
                f"{str(first_date):>15}"
                f"{str(last_date):>15}"
            )

            total_matches += (
                match_count
            )

        except Exception as error:

            print(
                f"{candidate['name']:<30}"
                f"{'ERROR':>10}"
            )

            print(
                f"    {error}"
            )

    print(
        "=" * 100
    )

    print(
        f"\nTotal matches across candidates: "
        f"{total_matches}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "A league season with only a few dozen "
        "matches is probably a partial open dataset."
    )


if __name__ == "__main__":
    main()