from statsbombpy import sb


MATCH_ID = 3754079


def main():
    print("Loading match events...\n")

    events = sb.events(match_id=MATCH_ID)

    print(f"Total events in match: {len(events)}")

    # Κρατάμε μόνο τα σουτ
    shots = events[events["type"] == "Shot"].copy()

    print(f"Total shots in match: {len(shots)}\n")

    columns_to_show = [
        "minute",
        "team",
        "player",
        "location",
        "shot_body_part",
        "shot_technique",
        "shot_type",
        "shot_outcome",
        "shot_statsbomb_xg",
    ]

    shots_clean = shots[columns_to_show]

    print(shots_clean.to_string(index=False))


if __name__ == "__main__":
    main()