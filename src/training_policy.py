"""Central policy helpers to enforce official training temporal pair."""

from __future__ import annotations

OFFICIAL_TRAIN_PAIR: tuple[int, int] = (2022, 2023)
OFFICIAL_HOLDOUT_PAIR: tuple[int, int] = (2023, 2024)


def is_holdout_pair(year_t: int, year_t1: int) -> bool:
    """Return True when pair matches the reserved holdout period."""
    return (int(year_t), int(year_t1)) == OFFICIAL_HOLDOUT_PAIR


def enforce_official_train_pair(
    year_t: int,
    year_t1: int,
    *,
    allow_nontrain_pair: bool = False,
    allow_holdout_training: bool = False,
) -> None:
    """Enforce official training pair policy with explicit override flags."""
    requested = (int(year_t), int(year_t1))
    if requested == OFFICIAL_TRAIN_PAIR:
        return

    if not allow_nontrain_pair:
        raise ValueError(
            "Training pair policy violation: requested "
            f"{requested[0]}->{requested[1]}. "
            f"Official training pair is {OFFICIAL_TRAIN_PAIR[0]}->{OFFICIAL_TRAIN_PAIR[1]}. "
            "Use --allow-nontrain-pair to override (not recommended)."
        )

    if requested == OFFICIAL_HOLDOUT_PAIR and not allow_holdout_training:
        raise ValueError(
            "Training pair policy violation: requested holdout pair "
            f"{requested[0]}->{requested[1]}. "
            "This pair is reserved for evaluation only. "
            "Use --allow-holdout-training together with --allow-nontrain-pair "
            "to override (not recommended)."
        )

