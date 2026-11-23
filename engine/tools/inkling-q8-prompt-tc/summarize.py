"""Check complete measurements and geometry against the grouped pp512 trace."""
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
import sys

raw, geometry = (Path(p) for p in sys.argv[1:])
kernels = json.loads(geometry.read_text())['workspace_context_512']['kernels']
counts = defaultdict(lambda: defaultdict(int))
for item in kernels:
    pack = item['name'] == 'gguf_packed_gemv' and '<(int)8, (int)32, (int)34,' in item['signature']
    pair = item['name'] == 'inkling_q8_reuse_pair'
    if (pack or pair) and item['block_x'] == 256:
        key = (item['grid_y'] * (2 if pair else 1), item['grid_x'] * 8,
               item['dynamic_shared_bytes'] // (8 if pair else 4))
        counts[key]['pair' if pair else 'packed'] += item['count']

modes = ['packed', 'pair', 'fused_wmma', 'blas_cached_fp16', 'blas_unpack_each']
times, errors = defaultdict(dict), {}
with raw.open() as handle:
    device = next(handle).strip()
    assert 'RTX 2080 Ti sm=75' in device and 'synthetic_Q8_0' in device
    for kind, m, n, k, mode, *values in csv.reader(handle):
        key = (int(m), int(n), int(k), mode)
        if kind == 'time':
            trial, us = int(values[0]), float(values[1])
            assert trial not in times[key] and us > 0
            times[key][trial] = us
        else:
            assert kind == 'error' and key not in errors
            errors[key] = list(map(float, values))
assert set(times) == set(errors)
assert set(key[:3] for key in times) == set(counts)
assert len(counts) == 17
rows = []
for key in sorted(counts, key=lambda key: (key[2], key[1], key[0])):
    rec = dict(zip(('prompt_rows', 'out_features', 'in_features'), key))
    rec['trace_calls'] = counts[key]
    rec['median_us'] = med = {}
    rec['trial_range_us'] = ran = {}
    rec['error_max_abs_rel_l2_max_abs_over_ref_peak'] = err = {}
    for mode in modes:
        t = times[(*key, mode)]
        assert set(t) == set(range(10))
        med[mode] = median(t.values())
        ran[mode] = [min(t.values()), max(t.values())]
        err[mode] = errors[(*key, mode)]
    assert err['packed'] == err['pair'] == [0, 0, 0]
    rows.append(rec)
print(json.dumps({'device': device, 'rows': rows}, indent=2))
