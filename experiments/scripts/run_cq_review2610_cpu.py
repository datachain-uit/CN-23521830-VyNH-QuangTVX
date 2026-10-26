"""Execute registered Review26 dev-only CPU audits and ten supplementary fits.

Fresh results preserve Review25. Resuming verifies config/source fingerprints;
no test payload, new search, panel run or approval is introduced by this script.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
from pathlib import Path
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import subprocess
import sys
import time
import numpy as np
import pandas as pd
import psutil

from skq_exp.config import ExperimentConfig
from skq_exp.data.cq_patch_1026 import course_constant_columns, quasi_id_audit, PANEL_SEEDS
from skq_exp.data.cq_tempo_features import add_hour_cyclical
from skq_exp.data.adapters.course_quality_v4 import _verify_sha256s
from scripts.prepare_cq_cd3_dev import prepare


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,(np.integer,np.bool_)):return x.item()
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    if isinstance(x,Path):return str(x)
    return x


def dump(path,x):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    tmp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('raw-root','v4-bundle','main-bundle','main-processed','reference-run','output','artifact-root'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--audit-only',action='store_true')
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    repo=Path(__file__).resolve().parents[2]
    source_files=[Path(__file__),*(repo/'experiments/src/skq_exp').rglob('*.py')]
    source_hash={str(f.relative_to(repo)).replace('\\','/'):sha(f) for f in source_files}
    dump(out/'source_fingerprint.json',{'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        'files':source_hash,'test_data_read':False})
    # Version-only distributions do not serialize credential-bearing direct URLs.
    frozen=sorted({f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions() if d.metadata.get('Name') and d.metadata['Name'].lower()!='skq-exp'})
    (out/'environment-freeze-versions.txt').write_text('\n'.join(frozen)+'\n',encoding='utf-8')
    versions={k:metadata.version(k) for k in ['numpy','pandas','scikit-learn','xgboost','pytest','pyarrow','scipy','psutil']}
    expected={'numpy':'2.2.3','pandas':'2.2.3','scikit-learn':'1.6.1','xgboost':'3.0.0'}
    if platform.python_version()!='3.13.2' or any(versions[k]!=v for k,v in expected.items()):
        raise ValueError('Review26 CPU manifest must match D13 Python/core versions')
    dump(out/'environment.json',{'python':platform.python_version(),'executable':sys.executable,'versions':versions,'platform':platform.platform(),'threads':4,'device':'cpu'})
    for source in (a.raw_root,a.main_bundle):_verify_sha256s(source)
    contract=json.loads((a.main_bundle/'bundle_manifest.json').read_text(encoding='utf-8'))['contract']
    cols=list(dict.fromkeys(contract['learner_columns']+contract['selector_columns']))
    train=add_hour_cyclical(pd.read_parquet(a.raw_root/'raw_features/pool_b/train.parquet').sort_values('row_id',kind='stable').reset_index(drop=True))
    dev=add_hour_cyclical(pd.read_parquet(a.raw_root/'raw_features/pool_b/dev_phase4.parquet'))
    groups=course_constant_columns(train,[c for c in cols if c!='phase_fraction'])
    audit=quasi_id_audit(train,dev,groups['g_ctx'])
    dump(out/'Gctx_audit.json',{'rule':'QD35_train_only_raw_features','columns':groups,'audit':audit,
        'source_train_sha256':sha(a.raw_root/'raw_features/pool_b/train.parquet'),
        'source_devP4_sha256':sha(a.raw_root/'raw_features/pool_b/dev_phase4.parquet'),
        'requires_data_team_match_adjudication':bool(audit['dev_matches']),'test_data_read':False})
    pd.DataFrame(audit['dev_matches'],columns=['dev_course','dev_family','train_families']).to_csv(out/'Gctx_dev_matches_private.csv',index=False)
    print('AUDIT_DONE',json.dumps({k:v for k,v in audit.items() if k!='dev_matches'}),flush=True)
    del train,dev
    if a.audit_only:return 0
    bundle=out/'minus_ctx_bundle';processed=out/'minus_ctx_processed'
    if not bundle.exists():prepare(a.raw_root,a.v4_bundle,bundle,processed,feature_variant='cd3_minus_ctx')
    new=json.loads((bundle/'bundle_manifest.json').read_text())['contract']
    if new['feature_variant']!='cd3_minus_ctx' or new['selector_columns']!=contract['selector_columns']:
        raise ValueError('Ablation must preserve selector view and its isolated variant')
    # Confirm all selector matrices/row order are bitwise identical to main.
    for suffix in ['train']:
        name='X_selector_'+suffix+'.npy'
        old=np.load(a.main_processed/'course_quality_v4'/name,mmap_mode='r')
        current=np.load(processed/'course_quality_v4'/name,mmap_mode='r')
        if not np.array_equal(old,current):raise ValueError(f'Selector matrix changed: {suffix}')
    dump(out/'minus_ctx_contract_verification.json',{'main_learner_columns':len(contract['learner_columns']),
        'minus_ctx_learner_columns':len(new['learner_columns']),'selector_columns':len(new['selector_columns']),
        'selector_matrices_identical':True,'removed_learner_context':new['removed_learner_context'],
        'v1_refit_train_only':True,'main_contract_hash':contract['contract_hash'],'ablation_contract_hash':new['contract_hash']})
    base=ExperimentConfig.from_json(a.reference_run/'selected_xgb_config.json')
    plan=[]
    for seed in (42,43,44):
        plan.append((f'minus_ctx_fullB_ms{seed}',replace(base,feature_variant='cd3_minus_ctx',model_seed=seed,paths=replace(base.paths,processed_root=processed))))
    plan.append(('minus_ctx_c02_rho005',replace(base,feature_variant='cd3_minus_ctx',method_ids=('c02_stratified_random',),
        budget_ratio=.005,selector_seeds=PANEL_SEEDS,paths=replace(base.paths,processed_root=processed))))
    for seed in (43,44):
        plan.append((f'y36_depth4_ms{seed}',replace(base,model_seed=seed,
            learner_params={'xgb':{**base.learner_params['xgb'],'max_depth':4}})))
    fingerprint=hashlib.sha256(json.dumps(source_hash,sort_keys=True).encode()).hexdigest()
    statepath=out/'execution_manifest.json'
    state=json.loads(statepath.read_text()) if statepath.exists() else {'status':'running','runs':[],
        'source_fingerprint':fingerprint,'scope':'QD35 eight fits + Y36 two fits, descriptive only',
        'test_data_read':False,'config_selection_allowed':False}
    if state['source_fingerprint']!=fingerprint:raise ValueError('Code changed since this fit batch; use fresh output paths')
    env=dict(os.environ,PYTHONPATH=str(repo/'experiments/src'),PYTHONIOENCODING='utf-8',PYTHONUNBUFFERED='1',
        OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='')
    for name,original in plan:
        if any(x['name']==name and x['status']=='success' for x in state['runs']):continue
        cfg=replace(original,experiment_id='r26_'+name,protocol_id='cq_review2610_registered_cpu',
            paths=replace(original.paths,artifact_root=a.artifact_root.resolve()),
            artifact_policy={**(original.artifact_policy or {}),'save_models':True})
        cfg.validate();cp=out/'configs'/f'{name}.json';dump(cp,cfg.to_dict())
        log=out/'logs'/f'{name}.log';log.parent.mkdir(exist_ok=True)
        rec={'name':name,'config':str(cp),'config_sha256':sha(cp),'status':'running','experiment_id':cfg.experiment_id,
            'artifact_root':str(cfg.paths.artifact_root),'model_seed':cfg.model_seed,'selector_seeds':list(cfg.selector_seeds)}
        state['runs'].append(rec);dump(statepath,state)
        print('FIT_START',name,flush=True);start=time.perf_counter();last=start;peak=0
        cmd=[sys.executable,'-u','-m','skq_exp','run','--config',str(cp)]
        with log.open('wb') as f:
            child=subprocess.Popen(cmd,cwd=repo/'experiments',env=env,stdout=f,stderr=subprocess.STDOUT)
            while child.poll() is None:
                try:peak=max(peak,psutil.Process(child.pid).memory_info().rss)
                except psutil.Error:pass
                if time.perf_counter()-last>=30:
                    print('FIT_PROGRESS',name,round(time.perf_counter()-start,1),flush=True);last=time.perf_counter()
                time.sleep(.5)
        mfiles=list((cfg.paths.artifact_root/'experiments'/cfg.experiment_id).rglob('manifest.json'))
        manifests=[json.loads(x.read_text()) for x in mfiles]
        expected_evals=4*len(cfg.selector_seeds)
        if child.returncode!=0 or len(manifests)!=expected_evals or any(m['status']!='success' for m in manifests):
            rec.update(status='failed',returncode=child.returncode);dump(statepath,state)
            raise RuntimeError(f'Fit failed; inspect {log}')
        rec.update(status='success',fits=len(cfg.selector_seeds),evaluations=len(manifests),wall_seconds=time.perf_counter()-start,
            sampled_peak_rss_bytes=peak,command=cmd)
        dump(statepath,state);print('FIT_DONE',name,rec['evaluations'],flush=True)
    state.update(status='success',fits=sum(x['fits'] for x in state['runs']),evaluations=sum(x['evaluations'] for x in state['runs']))
    dump(statepath,state);print('DONE',state['fits'],state['evaluations'],flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
