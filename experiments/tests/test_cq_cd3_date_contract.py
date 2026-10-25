"""CD3 receipt regression: accept date32/timestamp identity, reject time changes.

Synthetic schemas only. No real CQ model fit or research test access.
"""
from datetime import date
import json

import pandas as pd
import pytest

from scripts.prepare_cq_cd3_dev import _immutable_equal, prepare
from test_cq_review2310_pipeline import _sources, _hashes


def test_date32_and_timestamp_us_represent_the_same_midnight():
    old = pd.Series([date(2020, 8, 29)], index=['row'])
    new = pd.Series(pd.to_datetime(['2020-08-29']).as_unit('us'), index=['row'])
    assert _immutable_equal(new, old, 'enroll_time')


@pytest.mark.parametrize('change', ['microsecond', 'day', 'timezone', 'row_key', 'numeric_epoch'])
def test_time_contract_does_not_accept_real_changes(change):
    old = pd.Series([date(2020, 8, 29)], index=['row'])
    new = pd.Series(pd.to_datetime(['2020-08-29']).as_unit('us'), index=['row'])
    if change == 'microsecond':
        new += pd.Timedelta(microseconds=1)
    elif change == 'day':
        new += pd.Timedelta(days=1)
    elif change == 'timezone':
        new = new.dt.tz_localize('UTC')
    elif change == 'row_key':
        new.index = ['other']
    else:
        new = pd.Series([1598659200], index=['row'])
    assert not _immutable_equal(new, old, 'enroll_time')


def test_non_time_spine_fields_still_use_strict_equality():
    assert not _immutable_equal(pd.Series([1]), pd.Series([1.]), 'q_alpha_1')


@pytest.mark.parametrize('shift', [False, True])
def test_prepare_uses_time_contract_before_fitting_any_transform(tmp_path, shift):
    raw, v4 = _sources(tmp_path)
    for path in v4.rglob('*.parquet'):
        frame = pd.read_parquet(path)
        if 'enroll_time' in frame:
            frame['enroll_time'] = pd.to_datetime(frame.enroll_time).dt.date
            frame.to_parquet(path, index=False)
    _hashes(v4)
    if shift:
        path = raw / 'raw_features/pool_b/train.parquet'
        frame = pd.read_parquet(path)
        # Fixture parquet can use timestamp[ms]; preserve the intentional delta.
        frame['enroll_time'] = frame['enroll_time'].dt.as_unit('ns')
        frame.loc[0, 'enroll_time'] += pd.Timedelta(microseconds=1)
        frame.to_parquet(path, index=False)
        _hashes(raw)
        with pytest.raises(ValueError, match='immutable V4 spine mismatch in enroll_time'):
            prepare(raw, v4, tmp_path/'out', tmp_path/'processed')
        assert not (tmp_path/'out').exists()
    else:
        assert prepare(raw, v4, tmp_path/'out', tmp_path/'processed').is_file()


def test_prepare_matches_canonical_fit_order_after_parquet_row_permutation(tmp_path):
    raw, v4 = _sources(tmp_path)
    prepare(raw, v4, tmp_path/'first', tmp_path/'processed_first')
    for path in (raw/'raw_features').rglob('*.parquet'):
        frame = pd.read_parquet(path).sample(frac=1, random_state=19)
        frame.to_parquet(path, index=False)
    _hashes(raw)
    prepare(raw, v4, tmp_path/'second', tmp_path/'processed_second')
    for name in ['preprocessing.json','qa23_by_phase.csv']:
        assert (tmp_path/'first'/name).read_bytes() == (tmp_path/'second'/name).read_bytes()
    for folder in ['first','second']:
        train = pd.read_parquet(tmp_path/folder/'pool_b/train.parquet')
        assert train.row_id.tolist() == sorted(train.row_id.tolist())
