"""Train-pooled gender one-hot encoding for CQ CD3 LR (Review 2026-10-23).

Category frequencies are computed on ALL Pool B train, not on each coreset.
The numeric V1 block is unchanged. Unknown/rare categories share OTHER;
NULL has a separate column, independent of its frequency.
"""
from __future__ import annotations

import numpy as np

from ..artifacts import stable_hash


def lr_onehot_views(data: dict, min_frequency: int = 20) -> tuple[dict, dict]:
    contract = data.get("feature_contract") or {}
    if contract.get("feature_view") != "cd3":
        raise ValueError("Review 23/10 LR encoder requires prepared CD3 view")
    columns = list(contract["learner_columns"])
    gender_index = columns.index("gender")
    train_gender = np.asarray(data["X_train"])[:, gender_index]
    missing_code = contract.get("gender_missing_code")

    def missing(values):
        mask = np.isnan(values)
        if missing_code is not None:
            mask |= values == missing_code
        return mask

    values, counts = np.unique(train_gender[~missing(train_gender)], return_counts=True)
    kept = values[counts >= min_frequency]
    # Encoded unseen category 0 is never a train category of CQRawEarlyViews.
    kept = kept[kept != 0]
    output = dict(data)
    for key, raw in data.items():
        if not key.startswith("X_") or key.startswith("X_selector"):
            continue
        matrix = np.asarray(raw)
        gender = matrix[:, gender_index]
        null = missing(gender)
        onehot = np.zeros((len(matrix), len(kept) + 2), dtype=np.float32)
        for i, category in enumerate(kept):
            onehot[:, i] = (gender == category) & ~null
        onehot[:, -2] = ~null & ~np.isin(gender, kept)
        onehot[:, -1] = null
        output[key] = np.column_stack([np.delete(matrix, gender_index, axis=1), onehot])
    metadata = {
        "fit_scope": "all_pool_b_train", "min_frequency": min_frequency,
        "kept_gender_codes": kept.tolist(), "missing_code": missing_code,
        "output_columns": [c for c in columns if c != "gender"]
        + [f"gender_{v:g}" for v in kept] + ["gender_OTHER", "gender_NULL"],
    }
    metadata["encoding_hash"] = stable_hash(metadata)
    return output, metadata
