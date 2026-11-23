"""Recompute the 61-query diagnostic without models or unpublished source text."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/haiku-swap-20261123-v1'

def load(path):
    return json.loads(path.read_text())

summary = load(OUT / 'summary.json')
groups = {name: load(OUT / 'groups' / (name + '.json')) for name in summary}
reference = groups['QQ']['records']
checked = 0
for name, group in groups.items():
    rows = group['records']
    assert len(rows) == 61 and [r['i'] for r in rows] == list(range(61))
    assert [(r['task_id'], r['target'], r['query_sha256']) for r in rows] == [
        (r['task_id'], r['target'], r['query_sha256']) for r in reference]
    for row in rows:
        selected = row['selected_ids']
        assert len(selected) == len(set(selected)) == 6
        assert all(type(i) is int and 0 <= i < 365 for i in selected)
        assert row['hit6'] == (row['target'] in selected)
        candidate = row['candidate']
        if candidate is not None:
            ids = candidate['source_ids']
            assert candidate['index_target_present'] == (row['target'] in ids)
            picks = candidate['picker_local_indices']
            assert len(picks) == len(set(picks)) <= 6
            assert all(0 <= i < len(ids) for i in picks)
            assert {ids[i] for i in picks} <= set(selected)
        checked += 1
    computed = {
        'hit6': sum(r['hit6'] for r in rows),
        'text_hit6': sum(r['text_hit6'] for r in rows),
        'off_style_hit6': sum(r['off_style_hit6'] for r in rows),
        'candidate_covered': sum(bool(r['candidate'] and r['candidate']['index_target_present']) for r in rows),
        'n': len(rows),
    }
    for key, value in computed.items():
        assert value == group['summary'][key] == summary[name][key], (name, key)
    print(name, computed)

audit = load(OUT / 'protocol-picker-audit.json')
source = audit['friend_original_row']
assert source['design'] == 'hyde+prop' and source['mix'] == 'pick'
assert source['ctrl'] == 'wide' and source['pv'] == 2 and source['w'] == 8
assert source['scores']['inmind_off'] == [32, 61]
assert source['scores']['inmind_dense'] == [19, 61]
lost = [r for r in groups['HH']['records'] if r['candidate'] and r['candidate']['index_target_present'] and not r['hit6']]
assert [r['i'] for r in lost] == [r['i'] for r in audit['five_cases']]
assert len(lost) == 5 and sum(r['off_style_hit6'] for r in lost) == 4
for case, row in zip(audit['five_cases'], lost):
    pick = json.loads(case['raw'])['pick']
    assert pick == case['normalized']['pick']
    assert [n - 1 for n in pick] == row['candidate']['picker_local_indices'] == case['consumer_zero_based']
    assert row['candidate']['source_ids'].index(row['target']) + 1 == case['target_local_number']
    assert case['target_local_number'] not in pick and case['finish_reason'] == 'stop'
analysis = load(OUT / 'analysis.json')
for pair in analysis['paired']:
    a, b = groups[pair['a']]['records'], groups[pair['b']]['records']
    wins = sum(x['hit6'] and not y['hit6'] for x, y in zip(a, b))
    losses = sum(y['hit6'] and not x['hit6'] for x, y in zip(a, b))
    n = wins + losses
    p = min(1, 2 * sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / 2 ** n) if n else 1
    assert (wins, losses, wins - losses, p) == (pair['wins'], pair['losses'], pair['net'], pair['p_exact'])
print(f'PASS: {checked} rows; group totals, task alignment, picker indices, 5 raw outputs, source aggregate and paired comparisons.')
print('LIMIT: source texts/embeddings/raw scoring arrays are not published; text equality and off-style scores cannot be independently regenerated from this public export.')
print('LIMIT: original Haiku per-query picks are unavailable; this does not validate its 32/61 from per-query outputs or compare against RootMem.')
