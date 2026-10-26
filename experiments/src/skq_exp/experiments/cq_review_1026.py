"""Review 26 endpoint integration and guarded access to selection payloads.

Uses the unchanged teacher appendix. Reads train/dev artifacts only; physical
budget and effective support are distinct. Missing payloads are never inferred.
"""
from __future__ import annotations

import hashlib
import io
import numpy as np

from ..data.cq_patch_1026 import (
    ul_a8_by_phase, support_report, e_reference_floor, e_run_flags,
    constant_marginal_log_loss,
)
from ..training.weights import balanced_class_weights, effective_sample_weights

PHASES = ("P1", "P2", "P3", "P4")
LABELS = ("E", "G", "A")


def selection_support(manifest, cost, source, train, full_a, read_payload):
    """Verify selected NPY checksums and align indices to the frozen train order."""
    is_a = manifest['method_id'] == 'c00_full_train_a'
    frame = full_a if is_a else train
    y_full = frame.y_alpha_1.map(dict(zip(LABELS, range(3)))).to_numpy(int)
    if manifest['method_id'] in {'c00_full_train', 'c00_full_train_a'}:
        indices = np.arange(len(frame))
        weights = np.ones(len(frame))
        basis = 'full_reference_uniform_selection'
    else:
        ref = manifest.get('method_artifact_ref')
        if not ref or ref.get('output_kind') != 'selection':
            raise ValueError('Review26 support needs the original selection payload')
        values = {}
        for name in ('selected_indices.npy', 'sample_weights.npy'):
            payload = read_payload(ref['relative_path'] + '/' + name)
            if hashlib.sha256(payload).hexdigest() != ref['payload_sha256'][name]:
                raise ValueError(f'Selection payload checksum mismatch: {name}')
            values[name] = np.load(io.BytesIO(payload), allow_pickle=False)
        indices, weights = values['selected_indices.npy'], values['sample_weights.npy']
        if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
                or len(np.unique(indices)) != len(indices)
                or np.any(indices < 0) or np.any(indices >= len(frame))):
            raise ValueError('Invalid physical selection indices')
        basis = 'checksum_verified_method_artifact'
    y = y_full[indices]
    cw = balanced_class_weights(y_full) if cost.get('class_weight_policy') == 'balanced' else None
    effective = effective_sample_weights(y, weights, cw)
    phase = frame.phase.to_numpy()[indices]
    report = support_report(effective, y, phase)
    report.update(payload_basis=basis, weights_basis='QD15_effective_learner_weights',
                  fulltrain_a_without_training_phase=is_a,
                  E_rows_selected=int((y == 0).sum()),
                  E_rows_positive_weight=int(((y == 0) & (effective > 0)).sum()))
    return report


def extend_endpoint(run, proba_by_phase, y_by_phase, source_prior, phase_priors,
                    target_priors, support, *, is_full_a=False):
    """Global A8 remains primary; conditional A8 is sensitivity only (O32)."""
    p = {ph: np.asarray(proba_by_phase[ph], float) for ph in PHASES}
    p = {ph: v / v.sum(1, keepdims=True) for ph, v in p.items()}
    conditional = {} if is_full_a else {
        ph: None if not phase_priors or phase_priors.get(ph) is None else
        np.array([phase_priors[ph][lab] for lab in LABELS]) for ph in PHASES
    }
    # Teacher helper accepts NaN for an absent phase, not a fabricated prior.
    conditional = {ph: np.full(3, np.nan) if v is None else v for ph, v in conditional.items()}
    result = ul_a8_by_phase(p, y_by_phase,
        {'global': np.array([source_prior[lab] for lab in LABELS]), 'by_phase': conditional},
        {ph: np.array([target_priors[ph][lab] for lab in LABELS]) for ph in PHASES})
    if is_full_a:
        result['a8_phase_basis'] = 'not_applicable_FullTrain_A_no_phase_in_train'
    if not np.isclose(result['ul_decision'], run['ul_decision'], atol=1e-10):
        raise ValueError('Review26 must not change the existing global decision endpoint')
    run.update(result, support_report=support,
        marginal_predictions={ph: constant_marginal_log_loss(p[ph], y_by_phase[ph]) for ph in PHASES},
        rule_revision='Review-2026-10-26_QD33_QD35')
    e = run['e_class_raw']['pooled_four_phases']
    run['e_run_flags'] = e_run_flags(e, support['E_rows_selected'], support['E_rows_positive_weight'])
    run['e_flags_are_descriptive_only'] = True
    if run['method_id'] == 'c00_full_train':
        run['e_reference_floor'] = e_reference_floor(e['ap_E'], e['observed_E'])
    return run
