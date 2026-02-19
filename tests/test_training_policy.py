import pytest

from src.training_policy import (
    OFFICIAL_HOLDOUT_PAIR,
    OFFICIAL_TRAIN_PAIR,
    enforce_official_train_pair,
    is_holdout_pair,
)


def test_default_pair_passes() -> None:
    enforce_official_train_pair(
        OFFICIAL_TRAIN_PAIR[0],
        OFFICIAL_TRAIN_PAIR[1],
        allow_nontrain_pair=False,
    )


def test_nontrain_pair_fails_without_override() -> None:
    with pytest.raises(ValueError, match="2022->2023"):
        enforce_official_train_pair(2022, 2024, allow_nontrain_pair=False)


def test_override_allows_nontrain_pair() -> None:
    enforce_official_train_pair(2022, 2024, allow_nontrain_pair=True)


def test_holdout_requires_extra_flag() -> None:
    with pytest.raises(ValueError, match="holdout pair"):
        enforce_official_train_pair(
            OFFICIAL_HOLDOUT_PAIR[0],
            OFFICIAL_HOLDOUT_PAIR[1],
            allow_nontrain_pair=True,
            allow_holdout_training=False,
        )

    enforce_official_train_pair(
        OFFICIAL_HOLDOUT_PAIR[0],
        OFFICIAL_HOLDOUT_PAIR[1],
        allow_nontrain_pair=True,
        allow_holdout_training=True,
    )


def test_is_holdout_pair() -> None:
    assert is_holdout_pair(2023, 2024) is True
    assert is_holdout_pair(2022, 2023) is False

