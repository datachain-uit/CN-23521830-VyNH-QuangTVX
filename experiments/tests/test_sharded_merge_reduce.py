"""Kiểm tra P10/P11 dùng đủ train rows và không đổi contract budget."""

import numpy as np

from skq_exp.methods.proposed.sharded_merge_reduce import (
    ShardedMergeReduceSelector,
    _stratified_shards,
)
from skq_exp.methods.result import SelectionResult
from skq_exp.methods.proposed.structured_kquad import StructuredKQuadSelector


class _FakeParent:
    def __init__(self, method_id: str, candidate_stride: int):
        self.method_id = method_id
        self.candidate_stride = candidate_stride
        self.structure_parent_ids_ = None
        self.candidate_mask_ = None

    def select(self, X, y, budget_ratio):
        self.structure_parent_ids_ = np.arange(len(y), dtype=np.int64) // 8
        self.candidate_mask_ = np.arange(len(y)) % self.candidate_stride == 0
        return SelectionResult(
            indices=np.flatnonzero(self.candidate_mask_),
            weights=np.ones(int(self.candidate_mask_.sum())),
            requested_rows=max(1, round(len(y) * budget_ratio)),
            realized_rows=int(self.candidate_mask_.sum()),
            budget_mode="native_realized",
            method_id=self.method_id,
            diagnostics={"upstream_commit": "fake-locked-commit"},
            timings={"total": 0.0},
        )


def _factory(parent_method, seed, options):
    stride = 1 if parent_method == "n02_coretab_xgb_subset" else 3
    return _FakeParent(parent_method, stride)


def test_stratified_shards_are_deterministic_bounded_and_complete():
    y = np.repeat([0, 1, 2], [41, 37, 22])
    first = _stratified_shards(y, 17, 11)
    second = _stratified_shards(y, 17, 11)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    assert max(map(len, first)) <= 17
    assert np.array_equal(np.sort(np.concatenate(first)), np.arange(len(y)))


def test_p10_exact_budget_streams_all_rows():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(120, 6)).astype(np.float32)
    y = np.repeat([0, 1], 60)
    selector = ShardedMergeReduceSelector(
        "p10_skq_mr_coretab_xgb",
        ".",
        seed=11,
        n_components=16,
        options={
            "parent_method": "n02_coretab_xgb_subset",
            "shard_rows": 35,
            "budget_policy": "exact_total",
        },
        parent_factory=_factory,
    )
    result = selector.select(X, y, 0.1, row_ids=np.arange(len(y)))
    assert result.status == "success"
    assert result.realized_rows == 12
    assert result.diagnostics["all_train_rows_seen"] is True
    assert result.diagnostics["proxy_dataset"] is False
    assert result.diagnostics["rff_storage_mode"] == "group_streaming"


def test_p11_caps_at_bdis_candidate_pool_without_padding():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(120, 6)).astype(np.float32)
    y = np.repeat([0, 1], 60)
    selector = ShardedMergeReduceSelector(
        "p11_skq_mr_bdis",
        ".",
        seed=11,
        n_components=16,
        options={
            "parent_method": "n_bdis_native",
            "shard_rows": 35,
            "budget_policy": "cap_at_candidate_pool",
        },
        parent_factory=_factory,
    )
    result = selector.select(X, y, 0.5, row_ids=np.arange(len(y)))
    assert result.status == "success"
    assert result.realized_rows == result.diagnostics["candidate_rows_after_shards"]
    assert result.realized_rows < result.requested_rows
    assert len(np.unique(result.indices)) == result.realized_rows


def test_streaming_engine_matches_global_engine_on_same_groups():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(96, 5)).astype(np.float32)
    y = np.repeat([0, 1], 48)
    parents = np.arange(len(y), dtype=np.int64) // 16
    common = {
        "parent_ids": parents,
        "candidate_mask": np.ones(len(y), dtype=bool),
        "row_ids": np.arange(len(y), dtype=np.int64),
    }
    global_result = StructuredKQuadSelector(
        "p10_skq_mr_coretab_xgb", seed=11, n_components=24
    ).select(X, y, 0.25, **common)
    streamed_result = StructuredKQuadSelector(
        "p10_skq_mr_coretab_xgb", seed=11, n_components=24, stream_groups=True
    ).select(X, y, 0.25, **common)
    assert global_result.status == streamed_result.status == "success"
    assert np.array_equal(global_result.indices, streamed_result.indices)
    assert np.allclose(global_result.weights, streamed_result.weights, atol=1e-7)
