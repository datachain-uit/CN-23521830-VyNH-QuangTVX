"""Check Y37 wall/content units and cutoff boundaries on synthetic segments."""
import pandas as pd
import pytest
from scripts.audit_cq_review2610_stage import video_tail_table


def test_video_tail_respects_speed_start_at_cutoff_and_excludes_future():
    t=pd.Timestamp('2020-01-01')
    video=pd.DataFrame({'enrollment_id':['e']*4,
        'event_time':[t+pd.Timedelta(seconds=s) for s in (9,10,11,5)],
        'start_point':[0.,0.,0.,4.],'end_point':[4.,2.,5.,1.],
        'speed':[2.,1.,1.,1.]})
    spine=pd.DataFrame({'enrollment_id':['e'],'phase':['P1'],'split':['dev'],
        'enroll_time':[t],'cutoff_time':[t+pd.Timedelta(seconds=10)]})
    row=video_tail_table(video,spine)[0]
    assert row['selected_segments']==2
    assert row['segments_with_tail']==2 and row['strictly_crossing_segments']==1
    assert row['tail_at_cutoff_segments']==1
    assert row['tail_wall_seconds']==pytest.approx(3.)
    assert row['tail_content_seconds']==pytest.approx(4.)
    assert row['tail_wall_share_of_watch']==pytest.approx(.75)


def test_video_tail_does_not_include_future_or_pre_enrollment_segments():
    t=pd.Timestamp('2020-01-01')
    video=pd.DataFrame({'enrollment_id':['e','e'],
        'event_time':[t-pd.Timedelta(seconds=1),t+pd.Timedelta(seconds=20)],
        'start_point':[0.,0.],'end_point':[2.,2.],'speed':[1.,1.]})
    spine=pd.DataFrame({'enrollment_id':['e'],'phase':['P1'],'split':['train'],
        'enroll_time':[t],'cutoff_time':[t+pd.Timedelta(seconds=10)]})
    assert video_tail_table(video,spine)==[]
