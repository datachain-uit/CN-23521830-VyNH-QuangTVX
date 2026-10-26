"""Review26: split saved V50 true-positive E by seen/new learner, dev only."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scripts.analyze_cq_review2510_saved import read_runs
from scripts.run_cq_review2610_cpu import dump,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    train=pd.read_parquet(a.source/'pool_b/train.parquet',columns=['user_id'])
    dev={f'dev_phase{i}':pd.read_parquet(a.source/f'pool_b/dev_phase{i}.parquet') for i in range(1,5)}
    runs=read_runs(a.archive,dev)
    if len(runs)!=1:raise ValueError('Expected exactly one V50 fitted model')
    parts=next(iter(runs.values()));rows=[]
    for ph,values in parts.items():
        frame=dev['dev_phase'+ph[-1]];pred=values['pred']
        seen=frame.user_id.isin(train.user_id).to_numpy()
        y=pred['y_true'];estimated=pred['y_prob'].argmax(1)
        for label,mask in [('seen',seen),('new',~seen)]:
            e=mask & (y==0);tp=e & (estimated==0)
            predicted=mask & (estimated==0)
            rows.append({'phase':ph,'learner_group':label,'rows':int(mask.sum()),'true_E':int(e.sum()),
                'TP_E':int(tp.sum()),'FP_E':int((predicted & (y!=0)).sum()),
                'recall_E':float(tp.sum()/e.sum()) if e.any() else None,
                'precision_E':float(tp.sum()/predicted.sum()) if predicted.any() else None,
                'families_with_TP_E':len(np.unique(frame.course_family_id.to_numpy()[tp]))})
    a.output.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(a.output/'V50_TP_E_seen_new_by_phase.csv',index=False)
    dump(a.output/'receipt.json',{'archive_sha256':sha(a.archive),'source_sha256':sha(a.source/'SHA256SUMS.txt'),
        'rows':rows,'new_fits':0,'test_data_read':False,'group_unit':'learner user_id membership in Pool B train',
        'scope':'historical V4 view descriptive diagnosis; not a CD3 panel result'})
    print('V50_GROUP_REPORT_DONE',len(rows))


if __name__=='__main__':main()
