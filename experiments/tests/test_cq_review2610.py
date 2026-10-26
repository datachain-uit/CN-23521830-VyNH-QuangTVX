"""Review26 integration guards: five seeds, isolated context ablation and payloads."""
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from skq_exp.config import ExperimentConfig
from skq_exp.data.cq_cd3_views import build_cd3_views
from skq_exp.data.cq_patch_1026 import PANEL_SEEDS
from skq_exp.experiments.cq_review_1025 import verdict_proposal, choose_fixed_budget_baseline
from skq_exp.experiments.cq_review_1026 import selection_support
from test_cq_review2310_runtime import _frames

ROOT=Path(__file__).parents[1]


def test_panel_wrapper_defaults_to_five_keeps_explicit_historical_mode():
    a={s:.49 for s in PANEL_SEEDS}; b={s:.5 for s in PANEL_SEEDS}
    result=verdict_proposal(a,b,.01,a,b,phase_a={s:.499 for s in PANEL_SEEDS},phase_b=b)
    assert result['verdict']=='vượt' and result['a8_dependent']
    assert result['n_seeds_better']==5 and result['rule_revision'].endswith('26')
    with pytest.raises(ValueError):verdict_proposal({11:.49,29:.49,47:.49},.5,.01,{11:.49,29:.49,47:.49},.5)
    assert verdict_proposal({11:.49,29:.49,47:.49},.5,.01,{11:.49,29:.49,47:.49},.5,historical=True)['verdict']=='vượt'


def test_baseline_selection_requires_five_seeds_and_excludes_C03_native():
    records=[{'method_id':m,'budget_mode':mode,'role':role,'ul_decision_by_seed':{s:v for s in PANEL_SEEDS}}
             for m,mode,role,v in [('c03','fixed','C03',.1),('native','native','baseline',.2),('c02','fixed','baseline',.5)]]
    assert choose_fixed_budget_baseline(records)['method_id']=='c02'
    records[-1]['ul_decision_by_seed'].pop(101)
    with pytest.raises(ValueError):choose_fixed_budget_baseline(records)


def test_minus_ctx_train_only_discovery_keeps_selector_and_main_unchanged():
    train,dev,full=_frames()
    for f in [train,full]:
        num=f.enrollment_id.str.extract(r'(\d+)$')[0].astype(int)
        f['course_id']=np.where(num%2,'train-course1','train-course2')
        f['resource_count']=np.where(num%2,10.,20.)
    before=train.copy(deep=True)
    main,_,mc,mp=build_cd3_views(train,dev,full)
    ready,_,c,p=build_cd3_views(train,dev,full,feature_variant='cd3_minus_ctx')
    assert 'resource_count' in c['removed_learner_context']['g_ctx']
    assert 'resource_count' not in c['learner_columns']
    assert c['selector_columns']==mc['selector_columns']
    pd.testing.assert_frame_equal(ready['train'][c['selector_columns']],main['train'][mc['selector_columns']])
    changed={ph:f.assign(resource_count=np.arange(len(f))*100) for ph,f in dev.items()}
    _,_,c2,p2=build_cd3_views(train,changed,full,feature_variant='cd3_minus_ctx')
    assert c==c2 and p==p2
    pd.testing.assert_frame_equal(train,before)
    assert mp['center']==p['center']


def test_minus_ctx_config_cannot_be_used_as_unregistered_candidate():
    base=ExperimentConfig.from_json(ROOT/'configs/cq_v4_review2310_l0_v1.json')
    replace(base,feature_variant='cd3_minus_ctx').validate()
    with pytest.raises(ValueError):replace(base,feature_variant='cd3_minus_ctx',method_ids=('skq_pc',),budget_ratio=.005).validate()


def test_support_missing_payload_fails_instead_of_fabricating_uniform_weights():
    f=pd.DataFrame({'y_alpha_1':['E','G','A'],'phase':['P1']*3})
    with pytest.raises(ValueError,match='original selection'):
        selection_support({'method_id':'c02_stratified_random'},{},None,f,None,lambda _:b'')


def test_minus_ctx_preparation_accepts_only_explicitly_registered_separate_views(tmp_path):
    from scripts.prepare_cq_cd3_dev import prepare
    from test_cq_review2310_pipeline import _sources
    import json
    raw,v4=_sources(tmp_path)
    # Two synthetic train courses, preserving all immutable fields in both inputs.
    for root,prefix in [(raw/'raw_features',''),(v4,'')]:
        for relative in ['pool_b/train.parquet','fulltrain_a/train.parquet']:
            p=root/relative;f=pd.read_parquet(p)
            num=f.enrollment_id.str.extract(r'(\d+)$')[0].astype(int)
            f['course_id']=np.where(num%2,'train1','train2')
            f['resource_count']=np.where(num%2,10.,20.)
            f.to_parquet(p,index=False)
        from skq_exp.artifacts import sha256_file
        files=sorted(p for p in root.rglob('*') if p.is_file() and p.name!='SHA256SUMS.txt')
        # For raw fixture hashes live one directory above raw_features.
        target=raw if root==raw/'raw_features' else v4
        allfiles=sorted(p for p in target.rglob('*') if p.is_file() and p.name!='SHA256SUMS.txt')
        (target/'SHA256SUMS.txt').write_text(''.join(f'{sha256_file(p)}  {p.relative_to(target).as_posix()}\n' for p in allfiles))
    # The fixture split assignment references its original course; reflect this in
    # the synthetic audit tables while preserving family-level split identity.
    prepare(raw,v4,tmp_path/'bundle',tmp_path/'processed',feature_variant='cd3_minus_ctx')
    contract=json.loads((tmp_path/'processed/course_quality_v4/feature_contract.json').read_text())
    assert 'resource_count' not in contract['learner_columns']
    assert 'resource_count' in contract['selector_columns']


def test_endpoint_review26_uses_payload_and_never_gives_full_A_phase_priors(tmp_path):
    from scripts.prepare_cq_cd3_dev import prepare
    from scripts.compute_cq_v4_dev_endpoints import compute
    from skq_exp.experiments.runner import run_config
    from test_cq_review2310_pipeline import _sources
    raw,v4=_sources(tmp_path)
    bundle,processed=tmp_path/'bundle',tmp_path/'processed'
    prepare(raw,v4,bundle,processed)
    base=ExperimentConfig.from_json(ROOT/'configs/cq_v4_review2310_lr_01_v1.json')
    cfg=replace(base,experiment_id='toy26',protocol_id='r26',method_ids=('c02_stratified_random',),budget_ratio=.5,
        method_options={'c02_stratified_random':{'allowed_learners':['lr']}},
        learner_params={'lr':{'standardize':True,'max_iter':200,'C':1.,'solver':'lbfgs'}},
        paths=replace(base.paths,processed_root=processed,artifact_root=tmp_path/'artifacts'))
    assert all(x['status']=='success' for x in run_config(cfg))
    ep=compute(bundle,tmp_path/'artifacts',review_1026=True,n_bootstrap=5)['runs'][0]
    assert ep['support_report']['payload_basis']=='checksum_verified_method_artifact'
    assert ep['support_report']['budget_rows_physical']>0
    assert ep['ul_a8_global']==pytest.approx(ep['ul_decision'])
    assert len(ep['marginal_predictions'])==4 and ep['e_flags_are_descriptive_only']
