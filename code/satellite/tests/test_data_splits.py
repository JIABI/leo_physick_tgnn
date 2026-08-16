import pytest

from leo_pg.data.splits import split_episodes


def test_split_episodes_is_deterministic_and_disjoint():
    episodes = [{"id": index} for index in range(30)]
    first = split_episodes(episodes, seed=11)
    second = split_episodes(episodes, seed=11)
    assert first == second
    assert {name: len(values) for name, values in first.items()} == {
        "train": 24,
        "val": 3,
        "test": 3,
    }
    ids = [{episode["id"] for episode in first[name]} for name in ("train", "val", "test")]
    assert ids[0].isdisjoint(ids[1])
    assert ids[0].isdisjoint(ids[2])
    assert ids[1].isdisjoint(ids[2])


def test_split_ratios_must_be_valid():
    with pytest.raises(ValueError):
        split_episodes([1, 2, 3], ratios=(0.8, 0.2))
    with pytest.raises(ValueError):
        split_episodes([1, 2, 3], ratios=(0.8, 0.3, -0.1))
