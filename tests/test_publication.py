"""Checks that publication exports select the intended final experiment cells."""
from pathlib import Path

import pytest

from cox_physick.publication import export_publication_results, parameter_table
from cox_physick.results import read_csv, write_csv


DATA = Path(__file__).resolve().parents[1] / 'data'


def test_parameter_counts_match_the_reported_module_boundaries():
    assert [row['parameters'] for row in parameter_table()] == [
        83411, 112480, 78378, 83498, 78361,
    ]


def test_manuscript_rows_are_selected_once_with_full_precision(tmp_path):
    report = export_publication_results(DATA, tmp_path)
    assert report['manuscript_result_rows'] == 49
    nominal = read_csv(tmp_path / 'table_2_nominal.csv')
    horizon = read_csv(tmp_path / 'table_3_horizon.csv')
    ablations = read_csv(tmp_path / 'table_4_ablations.csv')
    hysteresis = read_csv(tmp_path / 'table_5_hysteresis.csv')
    assert [len(x) for x in (nominal, horizon, ablations, hysteresis)] == [8, 14, 18, 9]
    cases = [r['case_id'] for table in (nominal, horizon, ablations, hysteresis) for r in table]
    assert len(cases) == len(set(cases))
    original = {r['case_id']: r for r in read_csv(DATA / 'results_summary.csv')}
    for table in (nominal, horizon, ablations, hysteresis):
        for row in table:
            assert int(row['H']) in (200, 800)
            for key, value in row.items():
                if key.endswith(('_mean', '_sample_sd', '_n_units', '_n_valid_by_unit')):
                    assert value == original[row['case_id']][key]
    assert all(float(row['tau_hyst']) > .05 for row in hysteresis)
    assert {r['method_id'] for r in ablations if r['N_sat'] == '2000'} == {
        'no_triplet', 'no_cox_rst', 'eph_physick', 'mlp_coeff',
    }


def test_ambiguous_result_condition_is_rejected(tmp_path):
    rows = read_csv(DATA / 'results_summary.csv')
    write_csv(tmp_path / 'results_summary.csv', rows + [rows[0]])
    with pytest.raises(ValueError, match='Duplicate final result condition'):
        export_publication_results(tmp_path, tmp_path / 'tables')


def test_missing_manuscript_condition_is_rejected(tmp_path):
    rows = read_csv(DATA / 'results_summary.csv')
    rows = [r for r in rows if not (r['method_id'] == 'full' and r['N_sat'] == '2000'
                                   and r['K'] == '100' and r['H'] == '200'
                                   and r['weight_id'] == 'nominal' and r['tau_hyst'] == '0.05')]
    write_csv(tmp_path / 'results_summary.csv', rows)
    with pytest.raises(ValueError, match='Missing manuscript result condition'):
        export_publication_results(tmp_path, tmp_path / 'tables')
