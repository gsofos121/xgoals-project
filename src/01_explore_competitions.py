from statsbombpy import sb


def main():
    print("Loading StatsBomb competitions...\n")

    competitions = sb.competitions()

    columns_to_show = [
        "competition_id",
        "season_id",
        "country_name",
        "competition_name",
        "season_name",
    ]

    competitions_clean = competitions[columns_to_show].copy()

    competitions_clean = competitions_clean.sort_values(
        by=["country_name", "competition_name", "season_name"]
    )

    print(competitions_clean.to_string(index=False))

    print("\n--------------------------------")
    print(f"Total competition/seasons: {len(competitions_clean)}")
    print("--------------------------------")


if __name__ == "__main__":
    main()