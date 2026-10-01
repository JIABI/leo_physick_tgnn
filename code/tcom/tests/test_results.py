import numpy as np
import pytest
from cox_physick.metrics import episode_metrics, batch_prefix_metrics, summarize
from cox_physick.calibration import void_probability, estimate_kappa, bootstrap_kappa


def test_event_partition_and_distinct_hof_denominator():
    # Initial access is not an HO attempt; row1/user0 fails an actual attempt.
    a = np.array([[0, 7], [-1, 7], [0, -1], [7, -1]])
    q = np.array([[0, 7], [7, 7], [0, 7], [7, -1]])
    rates = np.where(a >= 0, 100., 0.)
    m = episode_metrics(a, q, rates, n_sat=2)
    assert m['outage_count'] == 3
    assert m['ho_attempt_count'] == 2
    assert m['ho_failure_count'] == 1
    assert m['hof_percent'] == 50
    assert m['outage_percent'] == 37.5
    assert m['source_retention_outage_count'] == 1
    assert m['empty_source_empty_request_outage_count'] == 1
    assert m['empty_source_access_outage_count'] == 0
    assert m['nonempty_source_empty_request_outage_count'] == 0
    assert m['nonattempt_outage_count'] == 2
    assert m['Cbar'] == pytest.approx(.3/8)
    assert m['Lbar'] == pytest.approx(5/80)
    assert m['J_user'] == pytest.approx(5*.375+.5*.3/8+.5*5/80-62.5/240)
    assert m['J_network'] == 2*m['J_user']


def test_empty_request_is_explicit_partition_not_ho_attempt():
    a = np.array([[0], [-1], [-1]])
    q = np.array([[0], [-1], [0]])
    m = episode_metrics(a, q, np.where(a >= 0, 40., 0.), n_sat=1)
    assert m['nonempty_source_empty_request_outage_count'] == 1
    assert m['empty_source_access_outage_count'] == 1
    assert m['ho_attempt_count'] == 0
    assert np.isnan(m['hof_percent'])


def test_load_squared_includes_outage_denominator_and_pair_stride():
    a = np.array([[0, 1, -1], [0, 7, -1], [0, 7, -1]])
    m = episode_metrics(a, a, np.where(a >= 0, 20., 0.), n_sat=2)
    assert m['load_square_sum'] == 8  # two beams on satellite0 in first row
    assert m['Lbar'] == pytest.approx(8/90)
    assert m['actual_nonempty_satellites_mean'] == pytest.approx(5/3)
    with pytest.raises(ValueError, match='capacity'):
        episode_metrics(a, a, np.where(a >= 0, 20., 0.), capacity=1)


def test_pp_excludes_empty_endpoints_and_uses_h_minus_two():
    a = np.array([[0], [7], [0], [-1], [0]])
    m = episode_metrics(a, a, np.where(a >= 0, 20., 0.))
    assert m['pp_event_count'] == 1
    assert m['pp_percent'] == pytest.approx(100/3)
    r = batch_prefix_metrics(a[:,None,:], a[:,None,:], np.where(a[:,None,:] >= 0, 20., 0.), [3,5])
    assert r[5]['peak_normalized'][0] >= r[3]['peak_normalized'][0]


def test_outage_nonzero_rate_is_rejected():
    with pytest.raises(ValueError, match='zero rate'):
        episode_metrics(np.array([[-1]]), np.array([[-1]]), np.array([[1.]]))


def test_seed_first_statistics_not_pooled_episodes():
    s = summarize([0, 2, 10, np.nan], [1, 1, 2, 2])
    assert s['mean'] == 5.5
    assert s['sample_sd'] == pytest.approx(9/np.sqrt(2))
    assert s['n_valid_by_unit'] == [2, 1]
    assert summarize([np.nan])['mean'] is None
    assert summarize([1])['sample_sd'] is None


def test_cox_full_void_and_limits():
    assert void_probability(3, .1, 0, 40) == 1
    assert void_probability(3, .1, 1, 1e8) == pytest.approx(np.exp(-3))
    assert void_probability(3, .1, .5, 20) < void_probability(3, .1, .5, 10)
    assert estimate_kappa([2,4],[4,6]) == .6
    boot = bootstrap_kappa([2,4],[4,6],[1,2], n_bootstrap=5)
    assert np.all(boot == .6)
    with pytest.raises(ValueError):
        void_probability(3,.1,1.1,10)
