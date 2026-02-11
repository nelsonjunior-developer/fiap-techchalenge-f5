import src.config as config


def test_random_state_constant_is_defined_and_is_int() -> None:
    assert hasattr(config, "RANDOM_STATE")
    assert isinstance(config.RANDOM_STATE, int)
    assert config.RANDOM_STATE == 42


def test_seed_alias_matches_random_state() -> None:
    assert config.SEED == config.RANDOM_STATE


def test_set_global_seed_runs_without_error() -> None:
    config.set_global_seed()

