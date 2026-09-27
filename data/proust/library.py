"""Programmatic index of the cycle: volumes, themes, involuntary-memory triggers."""

VOLUME_COUNT = 7
MADELEINE_VOLUME = 1          # the madeleine scene is in the first volume, "Combray"
FINAL_BALL_VOLUME = VOLUME_COUNT


def volume_titles() -> dict[int, str]:
    """English titles (C. K. Scott Moncrieff) of all seven volumes in order."""
    return {
        1: "Swann's Way",
        2: "Within a Budding Grove",
        3: "The Guermantes Way",
        4: "Sodom and Gomorrah",
        5: "The Captive",
        6: "The Fugitive",
        7: "Time Regained",
    }


def themes() -> list[str]:
    """The five major themes of the cycle used in the training course."""
    return ["memory", "time", "jealousy", "snobism", "art"]


def memory_triggers() -> dict[str, str]:
    """Sensation -> the hour of life it brings back."""
    return {
        "madeleine": "Combray, Sundays at Aunt Leonie's",
        "spoon": "the sound on the plate, the railway hammering",
        "paving-stones": "the baptistery of St Mark's, Venice",
        "steeple": "the two steeples of Martinville seen from the carriage",
    }
