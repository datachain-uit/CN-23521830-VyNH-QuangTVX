"""Shared deterministic CQ V4 data-building utilities; no model training."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
VERSION = 'CQ V4'
DATE = '2026-10-22'
TARGETS = ['q_alpha_0p1', 'y_alpha_0p1', 'q_alpha_1', 'y_alpha_1']
PHASES = ['P1', 'P2', 'P3', 'P4']
SPLITS = ['train', 'dev', 'test']


def qs(value):
    return "'" + str(value).replace("'", "''") + "'"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), default=str).encode()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + '.partial')
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + '\n', encoding='utf-8')
    partial.replace(path)


def records(con, sql):
    r = con.execute(sql)
    return [dict(zip([x[0] for x in r.description], row)) for row in r.fetchall()]


def view(con, name, path):
    con.execute(f'CREATE OR REPLACE VIEW "{name}" AS SELECT * FROM read_parquet({qs(path)})')


def write(con, path, sql):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix('.parquet.partial')
    con.execute(f'COPY ({sql}) TO {qs(partial)} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 50000)')
    partial.replace(path)


def connect(out, threads=2, memory='1200MB'):
    out = Path(out)
    (out / '_scratch').mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    for setting in [f'threads={threads}', f'memory_limit={qs(memory)}',
                    f'temp_directory={qs(out / "_scratch")}', "max_temp_directory_size='3GB'",
                    "TimeZone='UTC'", 'preserve_insertion_order=false']:
        con.execute('SET ' + setting)
    return con


def source_inventory(root):
    root = Path(root)
    return [[str(p.relative_to(root)), p.stat().st_size, p.stat().st_mtime_ns]
            for p in sorted(root.rglob('*')) if p.is_file() and '_scratch' not in p.parts]


def checksums(root):
    root = Path(root)
    paths = [p for p in sorted(root.rglob('*')) if p.is_file() and
             p.name != 'SHA256SUMS.txt' and '_scratch' not in p.parts and not p.name.endswith('.partial')]
    (root / 'SHA256SUMS.txt').write_text(''.join(f'{sha256(p)}  {p.relative_to(root).as_posix()}\n' for p in paths))


class Checks:
    def __init__(self):
        self.items = {}

    def add(self, name, passed, evidence=None):
        self.items[name] = {'status': 'PASS' if bool(passed) else 'FAIL', 'evidence': evidence}
        if not passed:
            raise AssertionError(f'{name}: {evidence}')

    def sql(self, con, name, sql):
        value = con.execute(sql).fetchone()[0]
        self.add(name, value, {'query': sql, 'result': value})

    def save(self, path):
        dump(path, {'dataset_version': VERSION, 'checks_passed': len(self.items),
                    'checks_total': len(self.items), 'checks': self.items})


def course_families(ccid_sets, threshold=0.9, minimum=5):
    """Global ccid graph, independent of titles; deterministic connected components."""
    keys = sorted(ccid_sets)
    parent = {key: key for key in keys}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    index = defaultdict(set)
    for key in keys:
        if len(ccid_sets[key]) >= minimum:
            for ccid in ccid_sets[key]:
                index[ccid].add(key)
    edges = []
    for key in keys:
        a = ccid_sets[key]
        if len(a) < minimum:
            continue
        candidates = set().union(*(index[v] for v in a))
        for other in sorted(x for x in candidates if x > key):
            b = ccid_sets[other]
            if min(len(a), len(b)) / max(len(a), len(b)) < threshold:
                continue
            shared, union = len(a & b), len(a | b)
            jaccard = shared / union
            if jaccard >= threshold:
                ra, rb = find(key), find(other)
                parent[max(ra, rb)] = min(ra, rb)
                edges.append({'course_id_a': key, 'course_id_b': other, 'intersection': shared,
                              'union_size': union, 'jaccard': jaccard})
    return {key: find(key) for key in keys}, edges


def family_split(frame, seed=42):
    """Review reference policy: chronological 20% test; rounded dev per stratum.

    Size rank ties use chronological order then family ID. Small strata can
    contribute zero dev families; there is no post-label manual reassignment.
    """
    frame = frame.sort_values(['family_start', 'course_family_id'], kind='stable').reset_index(drop=True)
    n_test = int(np.ceil(0.2 * len(frame)))
    frame['split'] = 'train'
    frame['size_quartile'] = pd.Series([pd.NA] * len(frame), dtype='Int64')
    frame['stratum'] = None
    frame.loc[len(frame) - n_test:, 'split'] = 'test'
    rest = frame.iloc[:len(frame) - n_test].copy()
    if len(rest) == 0:
        raise ValueError('No families left for train/dev')
    rest['size_quartile'] = pd.qcut(rest.n_enroll_labeled.rank(method='first'), min(4, len(rest)), labels=False)
    rest['stratum'] = rest.size_quartile.astype(str) + '_' + rest.has_E.astype(int).astype(str)
    frame.loc[rest.index, 'size_quartile'] = rest.size_quartile.astype('Int64')
    frame.loc[rest.index, 'stratum'] = rest.stratum
    rng = np.random.default_rng(seed)
    for _, group in rest.groupby('stratum', sort=True):
        positions = rng.permutation(group.index.to_numpy())
        k = int(round(0.125 * len(positions)))
        frame.loc[positions[:k], 'split'] = 'dev'
    if not {'train', 'dev', 'test'}.issubset(set(frame.split)):
        raise ValueError('Empty split under reference small-stratum policy; inspect before building')
    return frame.sort_values('course_family_id').reset_index(drop=True)
