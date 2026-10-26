"""Report registered context ablation/Y36 on dev with paired family intervals.

No training or reselection. Global A8 stays primary, phase A8 is sensitivity;
fixed model/selector seeds are averaged inside each whole-family resample.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from scripts.run_cq_review2610_cpu import dump,sha
from scripts.compute_cq_v4_dev_endpoints import compute,align_predictions
from skq_exp.data.cq_patch_1025 import a8_correct
from skq_exp.data.cq_patch_1023 import row_log_loss


def family_difference(delta,groups):
    phases=list(delta);keys=np.unique(groups[phases[0]].astype(str))
    sums=[];counts=[]
    for ph in phases:
        inverse=np.searchsorted(keys,groups[ph].astype(str))
        sums.append(np.bincount(inverse,weights=delta[ph],minlength=len(keys)))
        counts.append(np.bincount(inverse,minlength=len(keys)))
    sums=np.array(sums).T;counts=np.array(counts).T
    rng=np.random.default_rng(20261026);draws=[]
    for start in range(0,2000,100):
        w=rng.multinomial(len(keys),np.full(len(keys),1/len(keys)),size=100)
        draws.extend(((w@sums)/(w@counts)).mean(1))
    return {'mean':float(np.mean([x.mean() for x in delta.values()])),
        'ci95':np.quantile(draws,[.025,.975]),'n_bootstrap':2000,'seed':20261026,
        'bootstrap_unit':'course_family_id','phase_resampling':'joint',
        'seed_uncertainty':'conditional_on_fixed_seed_set; seed SD reported separately'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','main-bundle','old-reference-run'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();out=a.run/'analysis';out.mkdir(exist_ok=True)
    state=json.loads((a.run/'execution_manifest.json').read_text(encoding='utf-8'))
    assert state['status']=='success' and state['fits']==10 and state['evaluations']==40
    old=json.loads((a.old_reference_run/'execution_manifest.json').read_text(encoding='utf-8'))
    train=pd.read_parquet(a.main_bundle/'pool_b/train.parquet')
    dev={f'P{i}':pd.read_parquet(a.main_bundle/f'pool_b/dev_phase{i}.parquet') for i in range(1,5)}
    targets={ph:np.array([(train.loc[train.phase==ph,'y_alpha_1']==lab).mean() for lab in 'EGA']) for ph in dev}
    families={ph:f.course_family_id.to_numpy().astype(str) for ph,f in dev.items()}
    by_name={};rows=[];phase_rows=[];label_rows=[];replays=[];loss_cache={}

    def load_losses(rec,root):
        cache_key=(str(root),rec['experiment_id'])
        if cache_key in loss_cache:return loss_cache[cache_key]
        grouped={}
        for mp in (root/'experiments'/rec['experiment_id']).rglob('manifest.json'):
            m=json.loads(mp.read_text(encoding='utf-8'));ph='P'+m['evaluation_split'][-1]
            with np.load(mp.parent/'predictions.npz',allow_pickle=False) as z:pred=align_predictions(dev[ph],dict(z))
            cost=json.loads((mp.parent/'cost.json').read_text(encoding='utf-8'));source=np.zeros(3)
            for k,v in zip(cost['effective_class_prior']['labels'],cost['effective_class_prior']['probabilities']):source[int(k)]=v
            prob=pred['y_prob'].astype(float);prob/=prob.sum(1,keepdims=True)
            corr=a8_correct(prob,source,targets[ph]);assert corr is not None
            grouped.setdefault(m['selector_seed'],{})[ph]=row_log_loss(corr,pred['y_true'])
        loss_cache[cache_key]=grouped
        return grouped

    for rec in state['runs']:
        name=rec['name'];cfg=json.loads(Path(rec['config']).read_text(encoding='utf-8'))
        bundle=a.run/'minus_ctx_bundle' if cfg.get('feature_variant')=='cd3_minus_ctx' else a.main_bundle
        ep_path=out/f'endpoints_{name}.json'
        print('ENDPOINT_START',name,flush=True)
        if ep_path.exists():ep=json.loads(ep_path.read_text(encoding='utf-8'))
        else:
            ep=compute(bundle,Path(rec['artifact_root']),experiment_id=rec['experiment_id'],review_1026=True,n_bootstrap=2000)
            dump(ep_path,ep)
        by_name[name]=ep['runs']
        for run in ep['runs']:
            rows.append({'config':name,'selector_seed':run['selector_seed'],'model_seed':run['model_seed'],
                'UL_raw':run['ul_raw'],'UL_A8':run['ul_decision'],'UL_A8_phase':run['ul_a8_phase'],
                'macro_f1':run['mean_metrics']['f1_macro'],'MCC':run['mean_metrics']['mcc'],
                'EC':run['ec_decision']['ec_mean_p1_p4'],'E':run['e_class_raw']['pooled_four_phases']})
            for ph,values in run['overall_by_phase'].items():phase_rows.append({'config':name,'selector_seed':run['selector_seed'],'model_seed':run['model_seed'],'phase':ph,**values})
            for ph,labels in run['per_label_by_phase'].items():
                for lab in labels:label_rows.append({'config':name,'selector_seed':run['selector_seed'],'model_seed':run['model_seed'],'phase':ph,**lab})
        # Each model is physically saved once; replay a fixed 128-row P4 batch.
        runtime=Path(cfg['paths']['processed_root'])/'course_quality_v4'
        X=np.load(runtime/'X_dev_phase4.npy',mmap_mode='r')[:128]
        y=None
        for model_path in (Path(rec['artifact_root'])/'experiments'/rec['experiment_id']).rglob('model.joblib'):
            m=json.loads((model_path.parent/'manifest.json').read_text(encoding='utf-8'))
            assert m['evaluation_split']=='dev_phase1'
            candidates=[]
            for mp in (Path(rec['artifact_root'])/'experiments'/rec['experiment_id']).rglob('manifest.json'):
                mm=json.loads(mp.read_text(encoding='utf-8'))
                if mm['evaluation_split']=='dev_phase4' and mm['selector_seed']==m['selector_seed']:candidates.append(mp.parent)
            assert len(candidates)==1
            with np.load(candidates[0]/'predictions.npz',allow_pickle=False) as z:pred=align_predictions(dev['P4'],dict(z))
            actual=joblib.load(model_path).predict_proba(X)
            assert np.allclose(actual,pred['y_prob'][:128],atol=1e-6)
            replays.append({'config':name,'selector_seed':m['selector_seed'],'model_sha256':sha(model_path),'rows':128,'probabilities_match':True})
        print('ENDPOINT_DONE',name,flush=True)

    comparisons={};per_seed=[]
    for s in (42,43,44):
        name=f'minus_ctx_fullB_ms{s}';ref=next(x for x in old['runs'] if x['name']==f'fullB_ms{s}')
        newrec=next(x for x in state['runs'] if x['name']==name)
        new=next(iter(load_losses(newrec,Path(newrec['artifact_root'])).values()))
        main=next(iter(load_losses(ref,Path(old['artifact_root'])).values()))
        ci=family_difference({ph:new[ph]-main[ph] for ph in dev},families)
        comparisons[name]=ci;per_seed.append(ci['mean'])
    newrec=next(x for x in state['runs'] if x['name']=='minus_ctx_c02_rho005')
    ref=next(x for x in old['runs'] if x['name']=='c02_rho0.005')
    new=load_losses(newrec,Path(newrec['artifact_root']));main=load_losses(ref,Path(old['artifact_root']))
    comparisons['C02_mean5']=family_difference({ph:np.mean([new[s][ph]-main[s][ph] for s in new],axis=0) for ph in dev},families)
    comparisons['FullB_mean3']=family_difference({ph:np.mean([
        next(iter(load_losses(next(x for x in state['runs'] if x['name']==f'minus_ctx_fullB_ms{s}'),Path(newrec['artifact_root'])).values()))[ph]
        - next(iter(load_losses(next(x for x in old['runs'] if x['name']==f'fullB_ms{s}'),Path(old['artifact_root'])).values()))[ph]
        for s in (42,43,44)],axis=0) for ph in dev},families)
    interpretation=('Gctx_source_of_course_overfit' if all(v<=-.005 for v in per_seed) else
        'Gctx_transferable_information' if all(v>=.005 for v in per_seed) else 'not_distinguishable')
    dump(out/'QD35_COMPARISONS.json',{'differences_minus_ctx_minus_main':comparisons,'FullB_diff_by_model_seed':dict(zip((42,43,44),per_seed)),
        'registered_interpretation':interpretation,'main_configuration_changed':False,'test_data_read':False})
    dump(out/'REPLAY_VERIFICATION.json',{'models':replays,'no_refit':True})
    dump(out/'SUMMARY.json',{'rows':rows,'new_fits':10,'evaluations':40,'interpretation_QD35':interpretation,'test_data_read':False})
    pd.DataFrame(phase_rows).to_csv(out/'metrics_by_phase.csv',index=False)
    pd.DataFrame(label_rows).to_csv(out/'metrics_per_label_by_phase.csv',index=False)
    print('ANALYSIS_DONE',interpretation,flush=True)


if __name__=='__main__':main()
