"""Export the manuscript's final tables from the canonical result collection.

The CSVs retain the full supplied precision. Table-specific labels and ordering
are presentation metadata; every empirical value comes from results_summary.csv.
"""
from __future__ import annotations

import json
from pathlib import Path

from .results import read_csv, write_csv


BASELINES = (
    'maxsinr_ttt', 'maxrst', 'greedy_analytic', 'cox_only',
    'mlp213', 'kan_generic', 'leo_madrl', 'full',
)
ABLATIONS = (
    'cox_only', 'no_triplet', 'no_cox_rst', 'eph_physick',
    'mlp213', 'mlp_coeff', 'full',
)
LABELS = {
    'maxsinr_ttt': 'Max-SINR+TTT', 'maxrst': 'Max-RST',
    'greedy_analytic': 'Load-aware greedy', 'cox_only': 'Cox-only',
    'mlp213': 'TGN-MLP', 'kan_generic': 'TGN-KAN',
    'leo_madrl': 'LEO-MADRL', 'full': 'Cox-PhysiCK',
    'no_triplet': 'No arrival/RST descriptors',
    'no_cox_rst': 'No Cox, RST retained',
    'eph_physick': 'Ephemeris-PhysiCK',
    'mlp_coeff': 'MLP coefficient head',
}
TABLE_METRICS = (
    'outage_percent', 'hof_percent', 'pp_percent',
    'peak_normalized', 'tp_Mbps', 'J_user',
)
IDENTIFIERS = (
    'case_id', 'method_id', 'N_sat', 'K', 'H', 'weight_id', 'tau_hyst',
    'aggregation_unit', 'n_training_seeds', 'episodes_per_seed_or_rule',
)


def parameter_table():
    """Count the stated modules, excluding the shared GRUs and readouts."""
    def mlp(a, b, c):
        return (a + 1) * b + (b + 1) * c

    def kan(a, b, c):
        # Eight spline coefficients plus one base coefficient per connection.
        return 9 * (a * b + b * c) + b + c

    head = kan(262, 32, 10)
    return [
        {'module': 'TGN-MLP / no-kernel', 'mapping': '262 -> 213 -> 128',
         'parameters': mlp(262, 213, 128)},
        {'module': 'TGN-KAN message', 'mapping': '262 -> 32 -> 128',
         'parameters': kan(262, 32, 128)},
        {'module': 'KAN coefficient head', 'mapping': '262 -> 32 -> 10',
         'parameters': head},
        {'module': 'Full message, including bank',
         'mapping': 'KAN coefficient head + 10 physical lifts (4 -> 128)',
         'parameters': head + 10 * 4 * 128},
        {'module': 'MLP coefficient head', 'mapping': '262 -> 287 -> 10',
         'parameters': mlp(262, 287, 10)},
    ]


def export_publication_results(data_root, output):
    """Write the five manuscript tables and the reported cost/event views.

    ``output`` is the table directory, normally ``results/tables``. Nominal
    conditions appear only in Table II; Tables IV and V omit the corresponding
    repeated rows. Supplementary views expose cost and event components already
    contained in the canonical summaries, without changing their aggregation.
    """
    root, out = Path(data_root), Path(output)
    rows = read_csv(root / 'results_summary.csv')
    index = {}
    for row in rows:
        key = (row['method_id'], int(row['N_sat']), int(row['K']), int(row['H']),
               row['weight_id'], float(row['tau_hyst']) if row['tau_hyst'] else None)
        if key in index:
            raise ValueError(f'Duplicate final result condition: {key}')
        index[key] = row

    def select(method, n=2000, k=100, h=200, tau=.05):
        if method in ('maxsinr_ttt', 'maxrst', 'leo_madrl'):
            tau = None
        key = (method, n, k, h, 'nominal', tau)
        if key not in index:
            raise ValueError(f'Missing manuscript result condition: {key}')
        return index[key]

    def project(source, metrics, ablation=False):
        projected = []
        for position, row in enumerate(source, 1):
            item = {'row_in_table': position}
            item.update({field: row[field] for field in IDENTIFIERS})
            item['method'] = ('No kernel bank (TGN-MLP)' if
                              ablation and row['method_id'] == 'mlp213'
                              else LABELS[row['method_id']])
            for metric in metrics:
                for suffix in ('mean', 'sample_sd', 'n_units', 'n_valid_by_unit'):
                    field = f'{metric}_{suffix}'
                    if field in row:
                        item[field] = row[field]
                    elif suffix in ('mean', 'sample_sd'):
                        raise ValueError(f'Missing {field} in {row["case_id"]}')
            projected.append(item)
        return projected

    nominal = [select(method) for method in BASELINES]
    horizon = [select(method, n=n, h=800)
               for n in (500, 1000) for method in ('mlp213', 'kan_generic', 'full')]
    horizon += [select(method, h=800) for method in BASELINES]
    ablations = [select(method, n=n)
                 for n in (500, 1000) for method in ABLATIONS]
    ablations += [select(method) for method in
                  ('no_triplet', 'no_cox_rst', 'eph_physick', 'mlp_coeff')]
    hysteresis = [select(method, tau=tau)
                  for method in ('greedy_analytic', 'cox_only', 'full')
                  for tau in (.10, .20, .30)]

    tables = [
        ('II', 'table_2_nominal.csv',
         'Nominal controllers; N_sat=2000, K=100, H=200',
         project(nominal, TABLE_METRICS)),
        ('III', 'table_3_horizon.csv',
         'Longer prefix; H=800, K=100',
         project(horizon, TABLE_METRICS[:-1] + ('served_p5tp_Mbps', 'J_user'))),
        ('IV', 'table_4_ablations.csv',
         'Component comparisons; K=100, H=200; nominal repeats omitted',
         project(ablations, ('pp_percent', 'hof_percent', 'peak_normalized',
                             'tp_Mbps', 'J_user'), ablation=True)),
        ('V', 'table_5_hysteresis.csv',
         'Test-time hysteresis; nominal threshold is in Table II',
         project(hysteresis, ('outage_percent', 'hof_percent', 'pp_percent',
                              'tp_Mbps', 'J_user'))),
        ('VI', 'table_6_parameters.csv',
         'Message-module parameters; shared memories and readouts excluded',
         parameter_table()),
    ]
    assert sum(len(table[3]) for table in tables[:4]) == 49

    cost_cases = nominal + [select(method, k=500) for method in BASELINES]
    cost = project(cost_cases, (
        'outage_percent', 'Cbar', 'Lbar', 'tp_Mbps', 'weighted_outage',
        'weighted_ho', 'weighted_load', 'weighted_negative_rate', 'J_user',
        'J_network',
    ))
    for source, target in zip(cost_cases, cost):
        target.update({field: source[field] for field in
                       ('weight_outage', 'weight_ho', 'weight_load', 'weight_rate')})
    tables.append(('', 'cost_components.csv',
                   'Nominal and K=500 per-user cost components', cost))
    tables.append(('', 'attempts_and_outages.csv',
                   'Nominal and hysteresis event metrics; original aggregation',
                   project(nominal + hysteresis, (
                       'attempt_percent', 'executed_ho_per_user_min',
                       'outage_percent', 'hof_percent',
                       'attempt_associated_outage_percent', 'nonattempt_outage_percent',
                   ))))

    inventory = []
    for table, filename, description, contents in tables:
        write_csv(out / filename, contents)
        inventory.append({'manuscript_table': table, 'file': filename,
                          'rows': len(contents), 'description': description,
                          'source': ('module architecture' if table == 'VI'
                                     else 'data/results_summary.csv')})

    # The validation costs are supplied separately from the test summaries.
    groups_path = root / 'validation' / 'selection_group_summary.json'
    if groups_path.exists():
        groups = json.loads(groups_path.read_text())
        if isinstance(groups, dict):
            groups = groups.get('groups', groups.get('group_summary', groups))
        if not isinstance(groups, list):
            raise ValueError('selection_group_summary.json must contain group rows')
        write_csv(out / 'validation_costs.csv', groups)
        inventory.append({'manuscript_table': '', 'file': 'validation_costs.csv',
                          'rows': len(groups),
                          'description': 'Selected-model validation and test cost differences',
                          'source': 'data/validation/selection_group_summary.json'})

    write_csv(out / 'index.csv', inventory)
    report = {'manuscript_result_rows': 49, 'parameter_rows': 5,
              'tables': inventory,
              'statistics': {
                  'training_seed_mean': 'Mean and sample SD of the five seed-level episode means.',
                  'episode': 'Mean and sample SD across the valid evaluation episodes.',
                  'precision': 'CSV numbers retain full input precision; manuscript display is rounded.',
              }}
    out.mkdir(parents=True, exist_ok=True)
    (out / 'table_manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    return report
