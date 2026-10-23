"""Protocol edge cases that can leak content or invalidate a V4 split."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cq_v4_common import course_families, family_split, canonical_hash


def test_family_graph_uses_ccid_and_preserves_small_singletons():
    a=set(range(100))
    mapping,edges=course_families({'A':a,'B':set(range(95))|{200},'C':{1,2,3},'D':set()})
    assert mapping=={'A':'A','B':'A','C':'C','D':'D'}
    assert len(edges)==1 and edges[0]['jaccard']==95/101


def test_transitive_family_without_direct_endpoint_edge():
    mapping,edges=course_families({'A':set(range(100)),'B':set(range(1,101)),'C':set(range(2,102))},0.97)
    assert len(set(mapping.values()))==1
    assert {(e['course_id_a'],e['course_id_b']) for e in edges}=={('A','B'),('B','C')}


def test_family_ids_independent_of_dictionary_order():
    sets={'Z':set(range(20)),'A':set(range(20)),'M':set(range(50,80))}
    assert course_families(sets)==course_families(dict(reversed(list(sets.items()))))
    assert course_families(sets)[0]['Z']=='A'


def fixture_families(n=201):
    rng=np.random.default_rng(12)
    return pd.DataFrame({'course_family_id':[f'F{i:04d}' for i in range(n)],
        'family_start':pd.Timestamp('2020-01-01')+pd.to_timedelta(rng.integers(0,400,n),unit='D'),
        'n_enroll_labeled':rng.integers(1,1000,n),'has_E':rng.random(n)<0.25})


def test_test_ceil_count_chronology_and_no_duplicate_assignment():
    result=family_split(fixture_families())
    assert result.course_family_id.is_unique
    assert (result.split=='test').sum()==41
    assert result.loc[result.split=='test','family_start'].min()>=result.loc[result.split!='test','family_start'].max()
    assert set(result.split)=={'train','dev','test'}


def test_family_split_stable_for_reordered_input():
    frame=fixture_families()
    pd.testing.assert_frame_equal(family_split(frame),family_split(frame.sample(frac=1,random_state=9)))


def test_empty_dev_under_reference_rounding_is_detected():
    frame=pd.DataFrame({'course_family_id':[f'F{i:03d}' for i in range(40)],
        'family_start':pd.date_range('2020-01-01',periods=40),
        'n_enroll_labeled':range(1,41),'has_E':[i%2==0 for i in range(40)]})
    with pytest.raises(ValueError,match='Empty split'):
        family_split(frame)


def test_assignment_digest_independent_of_json_key_order():
    assert canonical_hash({'a':'train','b':'test'})==canonical_hash({'b':'test','a':'train'})
