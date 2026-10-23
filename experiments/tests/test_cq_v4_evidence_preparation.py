"""Guard dev prediction identity alignment and V4 config isolation from test."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.compute_cq_v4_dev_endpoints import align_predictions
from skq_exp.config import ExperimentConfig
from skq_exp.data.adapters.course_quality_v4 import _group_ids, _row_ids


def fixture_prediction():
    frame = pd.DataFrame({"row_id":["r0","r1","r2"], "course_id":["c0","c1","c2"], "course_family_id":["f0","f1","f2"], "y_alpha_1":["E","G","A"]})
    prediction = {"row_id":_row_ids(frame), "course_id":_group_ids(frame.course_id), "course_family_id":_group_ids(frame.course_family_id), "y_true":np.arange(3), "y_pred":np.arange(3), "y_prob":np.eye(3)}
    return frame, prediction


def test_v4_endpoint_reorders_by_identity_before_using_labels():
    frame, prediction = fixture_prediction()
    prediction = {key:value[[2,0,1]] for key,value in prediction.items()}
    assert np.array_equal(align_predictions(frame,prediction)["y_true"],np.arange(3))


def test_v4_endpoint_rejects_foreign_rows_and_family_mismatch():
    frame, prediction = fixture_prediction()
    prediction["row_id"][0] = 0
    with pytest.raises(ValueError, match="row IDs"):
        align_predictions(frame,prediction)
    frame, prediction = fixture_prediction()
    prediction["course_family_id"][0] = 0
    with pytest.raises(ValueError, match="course_family_id"):
        align_predictions(frame,prediction)


def test_prepared_v4_configs_keep_test_locked_and_natural_panel():
    root = Path(__file__).resolve().parents[1]
    paths = list((root/"configs").glob("cq_v4_m0_lr_tune_*.json"))
    assert len(paths) == 8
    for path in paths:
        config = ExperimentConfig.from_json(path)
        assert config.test_locked and config.dataset_ids == ("course_quality_v4",)
        assert config.method_ids == ("c00_full_train",)
    panel = ExperimentConfig.from_json(root/"configs/cq_v4_panel_dev_rho005_seeds11_29_47_v1.json")
    assert panel.test_locked and panel.class_weight_policy == "natural"
    assert len(panel.method_ids) == 13 and panel.selector_seeds == (11,29,47)
