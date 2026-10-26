"""Prepare five-seed dev configs from approved measured configs; never execute.

Review26 conditionally approved L0/XGB/LR, rho=.005 and delta=.010. Execution
still requires the merged code, locked environment and per-selector G10/QD16.
"""
import argparse
from dataclasses import replace
from pathlib import Path
from skq_exp.config import ExperimentConfig
from skq_exp.artifacts import atomic_json,sha256_file
from skq_exp.data.cq_patch_1026 import PANEL_SEEDS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('selected-xgb','selected-lr','output','processed-root','artifact-root'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Use fresh panel config output; preserve historical configs')
    project=Path(__file__).resolve().parents[1]
    xgb=ExperimentConfig.from_json(a.selected_xgb)
    lr=ExperimentConfig.from_json(a.selected_lr)
    values=xgb.learner_params['xgb']
    if values.get('max_depth')!=6 or values.get('learning_rate')!=.05 or values.get('min_child_weight',1)!=1:
        raise ValueError('Need Review26 approved measured XGB config')
    lrp=lr.learner_params['lr']
    if lrp.get('solver')!='newton-cholesky' or lrp.get('C')!=.1:
        raise ValueError('Need approved natural LR04')
    panel=ExperimentConfig.from_json(project/'configs/cq_v4_panel_dev_rho005_seeds11_29_47_v1.json')
    paths=replace(xgb.paths,processed_root=a.processed_root,artifact_root=a.artifact_root)
    policy=dict(xgb.artifact_policy or {});policy.pop('diagnostic_rounds',None)
    base=replace(xgb,evidence_role='dev_selection',learner_ids=('xgb','lr'),
        learner_params={'xgb':values,'lr':lrp},model_seed=42,budget_ratio=.005,
        non_inferiority_margin=.01,paths=paths,artifact_policy=policy)
    configs={}
    for seed in PANEL_SEEDS:
        options={m:{**(panel.method_options or {}).get(m,{}),'allowed_learners':['xgb','lr']} for m in panel.method_ids}
        cfg=replace(base,experiment_id=f'cq_cd3_r26_panel_ss{seed}',protocol_id='cq_review2610_panel_dev_v1',
            method_ids=panel.method_ids,method_options=options,selector_seeds=(seed,))
        cfg.validate();configs[f'panel_ss{seed}.json']=cfg
    for source in ('skq_pc','skq_pc_lrq_sq','skq_pc_sqrt'):
        old=ExperimentConfig.from_json(project/f'configs/cq_v4_dev_ablations_from_{source}_v1.json')
        # A2/A4 inherit the declared source. A3 remains the already registered
        # random-in-cell standard SKQ-PC; do not label it matched to LRQ.
        methods=tuple(m for m in old.method_ids if m!='c03_matched_skq_pc')
        opts=dict(old.method_options or {});opts.pop('c03_matched_skq_pc',None)
        if source=='skq_pc':
            methods=(*methods,'skq_pc_r');opts['skq_pc_r']=dict(opts['skq_pc'])
        for m in methods:
            if m not in {'a2_skq_pc_class_only','a4_skq_pc_uniform_weights'}:
                opts[m]={**opts.get(m,{}),'allowed_learners':['xgb','lr']}
        cfg=replace(base,experiment_id=f'cq_cd3_r26_ablation_{source}',protocol_id='cq_review2610_ablation_dev_v1',
            method_ids=methods,method_options=opts,selector_seeds=PANEL_SEEDS)
        cfg.validate();configs[f'ablation_source_{source}.json']=cfg
    a.output.mkdir(parents=True)
    for name,cfg in configs.items():atomic_json(a.output/name,cfg.to_dict())
    atomic_json(a.output/'PREPARATION_RECEIPT.json',{'review':'Review-2026-10-26_QD33',
        'source_xgb_sha256':sha256_file(a.selected_xgb),'source_lr_sha256':sha256_file(a.selected_lr),
        'panel_selector_seeds':list(PANEL_SEEDS),'model_seed':42,'rho_star':.005,'delta_ll':.01,
        'panel_methods':list(panel.method_ids),'main_panel_config_count':5,
        'conditional_ablation_templates':3,'ablation_source_not_selected_here':True,
        'execution_status':'not_run','test_locked':True,
        'execution_requirements':['code_merged','locked_environment','per_selector_G10_and_QD16'],
        'A3_basis':'registered_standard_SKQ_PC_random_in_cell; not LRQ-matched'})
    print('Prepared',len(configs),'dev configs; no fit/test executed')


if __name__=='__main__':main()
