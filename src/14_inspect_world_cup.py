from statsbombpy import sb


COMPETITION_ID = 43
SEASON_ID = 106


def main():

    print("Loading FIFA World Cup 2022 matches...\n")

    matches = sb.matches(
        competition_id=COMPETITION_ID,
        season_id=SEASON_ID
    )

    print("=" * 60)
    print("WORLD CUP 2022 DATASET")
    print("=" * 60)

    print(
        f"Total matches available: "
        f"{len(matches)}"
    )

    teams = set(
        matches["home_team"]
    ).union(
        set(
            matches["away_team"]
        )
    )

    print(
        f"Unique teams: "
        f"{len(teams)}"
    )

    print("\nTeams:")
    print("-" * 60)

    for team in sorted(teams):
        print(team)

    print("\n")
    print("=" * 60)
    print("FIRST 10 MATCHES")
    print("=" * 60)

    columns = [
        "match_id",
        "match_date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
    ]

    print(
        matches[columns]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()