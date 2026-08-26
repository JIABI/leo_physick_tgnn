import torch

from leo_pg.sim.visibility import (
    FrozenVisibilityConfig,
    sample_frozen_visibility_count,
    select_frozen_regional_satellites,
)


def test_frozen_visibility_selects_one_shared_unique_regional_subset():
    config = FrozenVisibilityConfig()
    count = sample_frozen_visibility_count(
        run_seed=17,
        episode_index=3,
        config=config,
    )
    first = select_frozen_regional_satellites(
        satellite_count=160,
        visible_count=count,
        run_seed=17,
        episode_index=3,
    )
    repeated = select_frozen_regional_satellites(
        satellite_count=160,
        visible_count=count,
        run_seed=17,
        episode_index=3,
    )
    other_episode = select_frozen_regional_satellites(
        satellite_count=160,
        visible_count=count,
        run_seed=17,
        episode_index=4,
    )

    assert config.minimum <= count <= config.maximum
    assert first.shape == (count,)
    assert torch.unique(first).numel() == count
    assert torch.equal(first, repeated)
    assert not torch.equal(first, other_episode)


def test_regional_subset_rejects_a_count_larger_than_the_population():
    try:
        select_frozen_regional_satellites(
            satellite_count=40,
            visible_count=41,
            run_seed=1,
            episode_index=0,
        )
    except ValueError as error:
        assert "visible_count" in str(error)
    else:  # pragma: no cover
        raise AssertionError("an impossible regional subset was accepted")
