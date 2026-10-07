from statsbombpy import sb


def main():
    print("Loading Premier League 2015/2016 matches...\n")

    matches = sb.matches(
        competition_id=2,
        season_id=27
    )

    columns_to_show = [
        "match_id",
        "match_date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
    ]

    matches_clean = matches[columns_to_show].copy()

    print(matches_clean.to_string(index=False))

    print("\n-----------------------------")
    print(f"Total matches: {len(matches_clean)}")
    print("-----------------------------")


if __name__ == "__main__":
    main()