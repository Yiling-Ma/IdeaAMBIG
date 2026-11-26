#!/usr/bin/env python3
"""Check released labels and frozen pipeline declarations without importing them."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check(root=ROOT):
    registry = json.loads((root / 'taxonomy.json').read_text())
    parents = {item['label']: item['level1'] for item in registry['level2']}
    level1 = set(registry['level1'])
    level2 = set(parents)
    assert len(level1) == 3 and len(level2) == 10
    assert set(parents.values()) == level1
    declarations = 0
    for path in sorted(root.rglob('*.py')):
        if '.git' in path.parts:
            continue
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if not isinstance(target, ast.Name):
                    continue
                name = target.id
                known = {'LEVEL2_TO_LEVEL1', 'VALID_LEVEL2_BY_LEVEL1',
                         'LEVEL2_DEFINITIONS', 'LEVEL2_DEFAULT_SPECIFICATION_SLOT',
                         'LEVEL2_NORMALIZED', 'LEVEL2_TO_CODIFICATION_SLOT',
                         'VALID_LEVEL1', 'LEVEL1_LABELS', 'VALID_LEVEL2',
                         'ALLOWED_LEVEL2', 'INCONSISTENCY_LEVEL2', 'DISALLOWED_LEVEL2'}
                if name not in known:
                    continue
                try:
                    value = ast.literal_eval(node.value)
                except ValueError:
                    # The GitHub route derives these sets/maps from the literal
                    # parent-to-children map checked independently above.
                    expression = ast.unparse(node.value)
                    derived = {
                        'LEVEL2_TO_LEVEL1': [
                            '{level2: level1 for level1, values in VALID_LEVEL2_BY_LEVEL1.items() for level2 in values}',
                            '{level2: level1 for level1, labels in VALID_LEVEL2_BY_LEVEL1.items() for level2 in labels}'],
                        'VALID_LEVEL1': ['set(VALID_LEVEL2_BY_LEVEL1.keys())'],
                        'VALID_LEVEL2': ['set(LEVEL2_TO_LEVEL1.keys())'],
                    }
                    assert expression in derived.get(name, []), f'{path}: unsupported derived declaration {name}'
                    declarations += 1
                    continue
                where = f'{path.relative_to(root)}:{node.lineno} {name}'
                if name == 'LEVEL2_TO_LEVEL1':
                    assert value == parents, where
                elif name == 'VALID_LEVEL2_BY_LEVEL1':
                    assert set(value) == level1, where
                    assert {label: parent for parent, labels in value.items()
                            for label in labels} == parents, where
                elif name in {'LEVEL2_DEFINITIONS', 'LEVEL2_DEFAULT_SPECIFICATION_SLOT',
                              'LEVEL2_NORMALIZED', 'LEVEL2_TO_CODIFICATION_SLOT'}:
                    assert set(value) == level2, where
                elif name in {'VALID_LEVEL1', 'LEVEL1_LABELS', 'VALID_LEVEL2', 'ALLOWED_LEVEL2'}:
                    expected = set(level1 if name in {'VALID_LEVEL1', 'LEVEL1_LABELS'} else level2)
                    if path.name == 'step_03_llm_route_classification.py':
                        expected.add('none')  # Routing sentinel, not a taxonomy category.
                    assert set(value) == expected, where
                elif name == 'INCONSISTENCY_LEVEL2':
                    assert set(value) == {k for k, v in parents.items() if v == 'Inconsistency'}, where
                else:
                    assert not (set(value) & level2), where
                declarations += 1

    label_pairs = 0
    def walk(value, where):
        nonlocal label_pairs
        if isinstance(value, dict):
            if 'level2' in value:
                assert value['level2'] in level2, f'{where}: invalid level2 {value["level2"]!r}'
                if 'level1' in value:
                    assert value['level1'] == parents[value['level2']], f'{where}: wrong parent'
                    label_pairs += 1
            elif 'level1' in value:
                assert value['level1'] in level1, f'{where}: invalid level1'
            for key, child in value.items():
                walk(child, f'{where}.{key}')
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f'{where}[{index}]')

    records = {}
    for path in sorted((root / 'Data').glob('*.json*')):
        if path.suffix == '.jsonl':
            values = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            records[path.name] = values
        else:
            values = json.loads(path.read_text())
        walk(values, path.name)
    all_ids = set()
    synthetic_ids = set()
    for filename, count in [('real_benchmark_instance.jsonl', 163),
                            ('syn_benchmark_instance.jsonl', 497),
                            ('syn_multi_defect_k2.jsonl', 50)]:
        rows = records[filename]
        assert len(rows) == count, filename
        expected_defects = 2 if filename == 'syn_multi_defect_k2.jsonl' else 1
        for row in rows:
            assert row['id'] not in all_ids, f'duplicate ID: {row["id"]}'
            all_ids.add(row['id'])
            assert len(row['defects']) == expected_defects, row['id']
            if filename == 'syn_benchmark_instance.jsonl':
                synthetic_ids.add(row['id'])
            if expected_defects == 1:
                targets = row['eval_targets']['underspecified']['expected_defects']
                assert len(targets) == 1, row['id']
                for key in ('level1', 'level2'):
                    assert targets[0][key] == row['defects'][0][key], row['id']
    for row in records['syn_multi_defect_k2.jsonl']:
        assert len(set(row['baseline_single_defect_ids'])) == 2, row['id']
        assert set(row['baseline_single_defect_ids']) <= synthetic_ids, row['id']
    docs = (root / 'TAXONOMY.md').read_text()
    assert all(label in docs for label in level1 | level2), 'TAXONOMY.md missing label'
    return declarations, label_pairs


if __name__ == '__main__':
    declarations, pairs = check()
    print(f'PASS: 3 Level-1 types, 10 Level-2 categories; {declarations} pipeline declarations; '
          f'{pairs} nested label pairs; 660 main records and 50 two-defect records.')
