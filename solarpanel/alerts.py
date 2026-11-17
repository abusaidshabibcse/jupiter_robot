"""Spoken alerts for objects that matter on a solar panel.

The camera model still knows the full COCO list plus leaf. This module keeps
only the classes that shade the sheet, soil it, or mean a person is there.
"""

from __future__ import annotations

# Higher entries are spoken first when several show up in the same frame.
ALERTS: dict[str, dict[str, str]] = {
    "person": {
        "one": "There is a person at the solar panel. Please stay clear of the array.",
        "many": "There are people at the solar panel. Please stay clear of the array.",
    },
    "dog": {
        "one": "There is a dog on the solar panel. Please move it away so it does not crack the glass.",
        "many": "There are dogs on the solar panel. Please move them away so they do not crack the glass.",
    },
    "cat": {
        "one": "There is a cat on the solar panel. Please remove it before it scratches the glass.",
        "many": "There are cats on the solar panel. Please remove them before they scratch the glass.",
    },
    "bird": {
        "one": "There is a bird on the solar panel. Please clear it so it does not leave droppings.",
        "many": "There are birds on the solar panel. Please clear them so they do not leave droppings.",
    },
    "leaf": {
        "one": "There is a leaf on the solar panel. Please clean it.",
        "many": "There are leaves on the solar panel. Please clean them.",
    },
}

ORDER = tuple(ALERTS)


def class_ids(names: dict[int, str]) -> list[int]:
    wanted = set(ALERTS)
    return [index for index, name in names.items() if name in wanted]


def message_for(counts: dict[str, int]) -> str:
    parts = []
    for name in ORDER:
        count = counts.get(name, 0)
        if count <= 0:
            continue
        form = "one" if count == 1 else "many"
        parts.append(ALERTS[name][form])
    return " ".join(parts)


def counts_from_boxes(class_ids_found: list[int], names: dict[int, str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for class_id in class_ids_found:
        name = names.get(int(class_id), "")
        if name not in ALERTS:
            continue
        counts[name] = counts.get(name, 0) + 1
    return counts
