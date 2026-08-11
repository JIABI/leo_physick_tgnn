from scripts.rollout import _summarize_results


def test_rollout_summary_keeps_requested_and_effective_horizons_distinct():
    results = [
        {"H": 10, "effective_H": 8, "diagnostic_steps": 7, "episode": 0, "mse_mean": 0.2},
        {"H": 10, "effective_H": 6, "diagnostic_steps": 5, "episode": 1, "mse_mean": 0.4},
    ]
    summary = _summarize_results(results)
    assert summary == [
        {
            "H": 10,
            "episodes": 2,
            "effective_H_min": 6,
            "effective_H_max": 8,
            "diagnostic_steps_min": 5,
            "diagnostic_steps_max": 7,
            "mse_mean_across_episodes": 0.30000000000000004,
        }
    ]
