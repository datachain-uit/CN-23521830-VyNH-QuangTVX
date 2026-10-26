"""Y37 video-tail measurements and proposed Y31 teacher groups from CD3 stage.

Read-only train/dev audit; no features, labels, split or original data files are
changed. Teacher grouping is an explicit four-way proposal awaiting data-team
confirmation, not a fabricated official teacher_subgroup or D06 human signoff.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from skq_exp.data.cq_tempo_features import clean_video_events
from scripts.run_cq_review2610_cpu import dump,sha


def video_tail_table(video,spine):
    video=clean_video_events(video.copy())
    cols=['enrollment_id','phase','split','enroll_time','cutoff_time']
    x=video.merge(spine[cols],on='enrollment_id',validate='many_to_many')
    x=x[(x.event_time>=x.enroll_time)&(x.event_time<=x.cutoff_time)].copy()
    x['watch_wall_seconds']=(x.end_point-x.start_point)/x.speed
    remaining=(x.cutoff_time-x.event_time).dt.total_seconds()
    x['tail_wall_seconds']=(x.watch_wall_seconds-remaining).clip(lower=0)
    x['tail_content_seconds']=x.tail_wall_seconds*x.speed
    x['tail']=x.tail_wall_seconds>0
    x['strictly_crossing']=x['tail']&(x.event_time<x.cutoff_time)
    rows=[]
    for (split,phase),f in x.groupby(['split','phase'],sort=True):
        rows.append({'split':split,'phase':phase,'selected_segments':len(f),
            'segments_with_tail':int(f['tail'].sum()),'strictly_crossing_segments':int(f.strictly_crossing.sum()),
            'tail_at_cutoff_segments':int((f['tail']&(f.event_time==f.cutoff_time)).sum()),
            'tail_share':float(f['tail'].mean()),'total_watch_wall_seconds':float(f.watch_wall_seconds.sum()),
            'tail_wall_seconds':float(f.tail_wall_seconds.sum()),'tail_content_seconds':float(f.tail_content_seconds.sum()),
            'tail_wall_share_of_watch':float(f.tail_wall_seconds.sum()/f.watch_wall_seconds.sum()) if f.watch_wall_seconds.sum()>0 else None,
            'enrollments_with_tail':int(f.loc[f['tail'],'enrollment_id'].nunique())})
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    spine_path=a.stage/'spine.parquet'
    spine=pd.read_parquet(spine_path,columns=['enrollment_id','course_id','row_id','split','phase','enroll_time','cutoff_time'])
    if not set(spine['split'])<={'train','dev'}:raise ValueError('Stage contains non-dev scope; reject instead of processing test')
    early=spine[spine.phase.isin(['P1','P2','P3','P4'])].copy()
    for c in ['enroll_time','cutoff_time']:early[c]=pd.to_datetime(early[c],errors='raise')
    video=pd.read_parquet(a.stage/'video.parquet')
    video['event_time']=pd.to_datetime(video.event_time,errors='raise')
    rows=video_tail_table(video,early)
    pd.DataFrame(rows).to_csv(a.output/'Y37_video_tail_by_split_phase.csv',index=False)
    dump(a.output/'Y37_video_tail_receipt.json',{'rows':rows,'source_sha256':{n:sha(a.stage/n) for n in ['video.parquet','spine.parquet']},
        'formula_wall':'max(0, (end_point-start_point)/speed - (cutoff_time-event_time).seconds)',
        'formula_content':'tail_wall_seconds * speed','eligible_segments':'clean video; enroll_time <= event_time <= cutoff_time',
        'scope':'train/dev P1-P4 only; FULL_A excluded','same_event_can_appear_at_multiple_phases':True,
        'human_D06_signoff':False,'test_data_read':False,'original_features_changed':False})
    course_teacher=pd.read_parquet(a.stage/'course_teacher.parquet').dropna(subset=['course_id','teacher_id']).drop_duplicates()
    course_teacher[['course_id','teacher_id']]=course_teacher[['course_id','teacher_id']].astype(str)
    train_courses=set(early.loc[early.split=='train','course_id'].astype(str))
    seen_teachers=set(course_teacher.loc[course_teacher.course_id.isin(train_courses),'teacher_id'])
    by_course=course_teacher.groupby('course_id').teacher_id.agg(set).to_dict()
    mapping=[]
    for course in sorted(set(early.loc[early.split=='dev','course_id'].astype(str))):
        teachers=by_course.get(course,set());old=teachers&seen_teachers
        group='missing' if not teachers else ('seen' if len(old)==len(teachers) else ('new' if not old else 'mixed'))
        mapping.append({'course_id':course,'teacher_count':len(teachers),'seen_teacher_count':len(old),
            'new_teacher_count':len(teachers-old),'teacher_subgroup_proposed':group})
    mp=pd.DataFrame(mapping);mp.to_parquet(a.output/'Y31_teacher_group_proposal_by_course_private.parquet',index=False)
    dev=early[early.split=='dev'].copy();dev['course_id']=dev.course_id.astype(str)
    dev=dev[['row_id','course_id','phase']].merge(mp,on='course_id',validate='many_to_one')
    dev.to_parquet(a.output/'Y31_teacher_group_proposal_by_row_private.parquet',index=False)
    dump(a.output/'Y31_teacher_group_proposal_receipt.json',{'status':'proposal_pending_data_team_definition_and_source_confirmation',
        'definitions':{'seen':'all mapped teachers observed in PoolB train','new':'no mapped teacher observed in PoolB train',
                       'mixed':'some but not all mapped teachers observed in train','missing':'no usable teacher mapping'},
        'course_counts':mp.teacher_subgroup_proposed.value_counts().to_dict(),
        'row_counts_by_phase':dev.groupby(['phase','teacher_subgroup_proposed']).size().unstack(fill_value=0).to_dict(),
        'source_course_teacher_sha256':sha(a.stage/'course_teacher.parquet'),
        'seen_teachers_in_train':len(seen_teachers),'original_dataset_changed':False,
        'IDs_are_metadata_only':True,'test_data_read':False})
    print('STAGE_AUDIT_DONE',len(rows),'video groups;',len(mapping),'teacher course mappings')


if __name__=='__main__':main()
