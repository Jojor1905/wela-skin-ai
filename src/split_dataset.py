"""Deterministic, duplicate-aware split assignment for detection records.

Subject identifiers are not available for ACNE04, so this module prevents only
known image-duplicate leakage; it cannot guarantee person-level separation.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass


SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class SplitItem:
    """The split-relevant, non-sensitive metadata for one converted image."""

    image_id: str
    object_count: int
    small_box_count: int
    duplicate_group_id: str = ""


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> dict[str, float]:
    """Validate and return ratios with a precise actionable error."""
    ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    if any(ratio <= 0 for ratio in ratios.values()):
        raise ValueError("All split ratios must be greater than zero.")
    if abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError("Train, validation, and test ratios must sum to 1.0.")
    return ratios


def assign_splits(
    items: list[SplitItem],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, str]:
    """Assign all items while keeping every known duplicate group together.

    A deterministic greedy assignment balances images, object totals, and
    small-box totals. It uses no unverified source split-list semantics.
    """
    if not items:
        raise ValueError("Cannot split an empty item list.")
    if len({item.image_id for item in items}) != len(items):
        raise ValueError("Split items must have unique image IDs.")
    ratios = validate_ratios(train_ratio, val_ratio, test_ratio)
    groups: dict[str, list[SplitItem]] = defaultdict(list)
    for item in items:
        group_key = item.duplicate_group_id or f"single:{item.image_id}"
        groups[group_key].append(item)

    random_generator = random.Random(seed)
    ordered_groups = list(groups.items())
    random_generator.shuffle(ordered_groups)
    ordered_groups.sort(
        key=lambda entry: (
            -sum(item.object_count for item in entry[1]),
            -len(entry[1]),
            -sum(item.small_box_count for item in entry[1]),
            entry[0],
        )
    )
    totals = {
        "images": len(items),
        "objects": sum(item.object_count for item in items),
        "small_boxes": sum(item.small_box_count for item in items),
    }
    targets = {
        split: {metric: totals[metric] * ratio for metric in totals}
        for split, ratio in ratios.items()
    }
    current = {split: {metric: 0 for metric in totals} for split in SPLIT_NAMES}
    assignments: dict[str, str] = {}
    for _, group_items in ordered_groups:
        group_values = {
            "images": len(group_items),
            "objects": sum(item.object_count for item in group_items),
            "small_boxes": sum(item.small_box_count for item in group_items),
        }

        def score(split: str) -> tuple[float, str]:
            """Choose the least-filled target before placing the next whole group.

            Comparing current target utilisation (rather than the first group's
            post-placement error) prevents early high-object groups from being
            funnelled into the smaller validation or test targets.
            """
            utilisation = 0.0
            for metric, weight in (("images", 4.0), ("objects", 1.0), ("small_boxes", 1.0)):
                utilisation += weight * current[split][metric] / max(targets[split][metric], 1.0)
            return utilisation, split

        chosen_split = min(SPLIT_NAMES, key=score)
        for metric, value in group_values.items():
            current[chosen_split][metric] += value
        for item in group_items:
            assignments[item.image_id] = chosen_split

    if len(assignments) != len(items):
        raise RuntimeError("Split assignment did not assign every image.")
    for group_items in groups.values():
        if len({assignments[item.image_id] for item in group_items}) != 1:
            raise RuntimeError("A duplicate group crosses splits after assignment.")
    return assignments
