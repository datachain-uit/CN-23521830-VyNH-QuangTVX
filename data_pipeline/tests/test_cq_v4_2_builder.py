"""Integration checks of source mapping, temporal boundaries and bundle wiring."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import duckdb
import numpy as np
import pandas as pd
import pytest

PIPELINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE))
spec = importlib.util.spec_from_file_location("cd3_builder", PIPELINE/"25_build_phase_features_v4_2.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def sqlfile(root, rel, sql):
    p = root/rel
    p.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    try:
        con.execute(f"COPY ({sql}) TO '{p}' (FORMAT PARQUET)")
    finally:
        con.close()


def make_input(root, include_test=False):
    for phase, days in [("P1",10),("P2",20),("P3",30),("P4",36),("FULL_A",40)]:
        for split in (["train"] if phase=="FULL_A" else ["train","dev"]+(["test"] if include_test else [])):
            users = [("U_a","C_1","e1",.1),("U_b","C_1","e2",.6)] if split=="train" else (
                [("U_c","C_2","e3",.4)] if split=="dev" else [("U_4","C_3","e4",.9)])
            frame = pd.DataFrame([dict(user_id=u,course_id=c,course_family_id=c,enrollment_id=e,
                row_id=e+phase,phase=phase,split=split,enroll_time=pd.Timestamp("2020-01-01"),
                cutoff_time=pd.Timestamp("2020-01-01")+pd.Timedelta(days=days),course_end=pd.Timestamp("2020-02-10"),
                q_alpha_0p1=q,y_alpha_0p1="E" if q<.2 else "A",q_alpha_1=q,y_alpha_1="E" if q<.2 else "A") for u,c,e,q in users])
            for col in ["n_video_segments","video_media_seconds","video_mean_segment_seconds","video_mean_speed",
                        "n_videos_watched","video_active_days","n_problem_submissions","n_exercises_attempted",
                        "problem_attempts","n_correct_problems","n_comments","comment_mean_characters"]:
                frame[col]=0.0
            rel = "fulltrain_a/train_raw.parquet" if phase=="FULL_A" else f"supervised_raw/{phase}/{split}.parquet"
            p = root/"phase_features_v4"/rel
            p.parent.mkdir(parents=True,exist_ok=True)
            con = duckdb.connect()
            con.register("f",frame)
            con.execute(f"COPY f TO '{p}' (FORMAT PARQUET)")
            con.close()
    tables = {
        "entities/course": "SELECT 'C_1' id,[{'resource_id':'V_1'},{'resource_id':'Ex_1'}] resource UNION ALL SELECT 'C_2',[{'resource_id':'V_2'},{'resource_id':'Ex_2'}]",
        "entities/user": "SELECT * FROM (VALUES ('U_a',1,1990),('U_b',2,1991),('U_c',99,1900)) t(id,gender,year_of_birth)",
        "relations/user_video": """SELECT 'U_a' user_id,[{'video_id':'V_1','segment':[
            {'start_point':0.0,'end_point':100.0,'speed':1.0,'local_start_time':1577966400::BIGINT},
            {'start_point':0.0,'end_point':50.0,'speed':2.0,'local_start_time':1579521600::BIGINT}]}] seq
            UNION ALL SELECT 'U_c',[{'video_id':'V_2','segment':[
            {'start_point':0.0,'end_point':1e9,'speed':1.0,'local_start_time':1577959200::BIGINT}]}]""",
        "entities/problem": """SELECT * FROM (VALUES ('1','Ex_1',4.0,'[1,2]'::JSON),
            ('2','Ex_1',2.0,'[1,2]'::JSON),('3','Ex_1',1.0,'[2]'::JSON),
            ('4','Ex_2',2.0,'[99]'::JSON)) t(problem_id,exercise_id,score,context_id)""",
        "relations/user_problem": """SELECT * FROM (VALUES
            ('U_a','Pm_1',2.0,1,1,'2020-01-02 12:00:00'),
            ('U_a','Pm_2',NULL,0,2,'2020-01-03 12:00:00'),
            ('U_a','Pm_3',1.0,1,0,'2020-01-04 12:00:00')) t(user_id,problem_id,score,is_correct,attempts,submit_time)""",
        "entities/translated_comment": """SELECT * FROM (VALUES
            ('Cm_1',1::BIGINT,'hi','V_1',1577959200000::BIGINT),
            ('Cm_2',1::BIGINT,'hi','V_1',1577959200000::BIGINT),
            ('Cm_3',1::BIGINT,'',NULL,1577959200000::BIGINT)) t(id,user_id,text,resource_id,create_time)""",
        "relations/course_comment": "SELECT 'C_1' course_id,unnest(['Cm_1','Cm_2','Cm_3']) comment_id",
        "relations/course_teacher": "SELECT 'C_1' course_id,'T_1' teacher_id UNION ALL SELECT 'C_2','T_2'",
        "entities/teacher": "SELECT 'T_1' id,'org' org_name,'about' about UNION ALL SELECT 'T_2','new','long dev biography'",
        "relations/course_school": "SELECT 'C_1' course_id,'S_1' school_id UNION ALL SELECT 'C_2','S_1'",
        "entities/school": "SELECT 'S_1' id,'school' about,'learn' motto",
    }
    if include_test:
        extras = {
            "entities/course":"SELECT 'C_3',[{'resource_id':'V_3'},{'resource_id':'Ex_3'}]",
            "entities/user":"SELECT 'U_4',42,1950",
            "relations/user_video":"SELECT 'U_4',[{'video_id':'V_3','segment':[{'start_point':0.0,'end_point':1e6,'speed':2.0,'local_start_time':1577966400::BIGINT}]}]",
            "entities/problem":"SELECT '5','Ex_3',5.0,'[3000]'::JSON",
            "relations/user_problem":"SELECT 'U_4','Pm_5',5.0,1,100,'2020-01-02 12:00:00'",
            "entities/translated_comment":"SELECT 'Cm_4',4::BIGINT,'hi','V_3',1577959200000::BIGINT",
            "relations/course_comment":"SELECT 'C_3','Cm_4'",
            "relations/course_teacher":"SELECT 'C_3','T_3'",
            "entities/teacher":"SELECT 'T_3','test org','test biography'",
            "relations/course_school":"SELECT 'C_3','S_2'",
            "entities/school":"SELECT 'S_2','test school','test motto'",
        }
        for key,sql in extras.items():
            tables[key] += " UNION ALL "+sql
    # User IDs in raw comments are numeric and are normalized to U_<number>.
    # Use U_1 as the first enrollment and event identity throughout the fixture.
    for rel, sql in tables.items():
        sqlfile(root,"input/moocubex_parquet/tables/"+rel+"/part.parquet",sql.replace("U_a","U_1"))
    for p in (root/"phase_features_v4").rglob("*.parquet"):
        con=duckdb.connect()
        f=con.read_parquet(str(p)).df()
        f.user_id=f.user_id.replace({"U_a":"U_1"})
        con.register("f",f)
        con.execute(f"COPY f TO '{p}' (FORMAT PARQUET)")
        con.close()
    hash_hi=module.cardiff_hash("hi")
    for rel,sql in [
        ("afelo_sentiment_cardiff/text_predictions/part.parquet",f"SELECT '{hash_hi}' text_hash,.1 p_neg,.2 p_neu,.7 p_pos,'Positive' predicted_label"),
        ("final_quality_v2/sentiment/text_predictions/part.parquet","SELECT 'unused' text_hash,.2 p_neg,.6 p_neu,.2 p_pos,'Neutral' predicted_label")]:
        sqlfile(root,rel,sql)


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    root=tmp_path_factory.mktemp("cd3_source")
    make_input(root)
    out=root/"output"
    module.Builder(root,out,memory="512MB",batch_size=2).run()
    return root,out


def test_end_to_end_raw_mapping_and_frozen_spine(bundle):
    _,out=bundle
    con=duckdb.connect()
    raw=con.read_parquet(str(out/"raw_features/pool_b/train.parquet")).df()
    assert len(raw)==8 and raw.row_id.nunique()==8
    r=raw[(raw.enrollment_id=="e1") & (raw.phase=="P1")].iloc[0]
    assert r.problem_count==3 and r.problem_context_count==2
    assert r.score_sum==3 and r.course_problem_full_score_sum==7
    assert r.comment_count==3 and r.positive_comment_count==2
    assert r.avg_positive_score==pytest.approx(.7)
    assert r.segment_count==1  # the later event is available at P2, not P1
    assert raw[(raw.enrollment_id=="e1") & (raw.phase=="P2")].segment_count.item()==2
    absent=raw[(raw.enrollment_id=="e2") & (raw.phase=="P1")].iloc[0]
    assert pd.isna(absent.segment_count) and absent.video_observed_mask==0
    assert len(con.read_parquet(str(out/"raw_features/fulltrain_a/train.parquet")).df())==2
    con.close()


def test_train_fit_and_matrix_alignment(bundle):
    _,out=bundle
    prep=json.loads((out/"preprocessing.json").read_text())
    assert prep["fit_rows"]==8 and prep["static_fit_rows"]==2
    assert set(prep["category_maps"]["gender"])=={"1","2"}
    con=duckdb.connect()
    dev=con.read_parquet(str(out/"model_ready/pool_b/dev_phase1.parquet")).df()
    assert dev.gender.tolist()==[0] and dev.phase_fraction.tolist()==[.25]
    assert np.isfinite(dev[prep["columns"]["learner"]].to_numpy()).all()
    assert prep["columns"]["learner"]==prep["columns"]["selector"]+["gender","phase_fraction"]
    assert json.loads((out/"_SUCCESS.json").read_text())["checks_passed"]>30
    manifest=json.loads((out/"run_manifest.json").read_text())
    assert manifest["test_data_read"] is False
    d06=json.loads((out/"audit/d06_comparison.json").read_text())
    assert d06["status"]=="automatic_comparison" and d06["total_pairs"]==3
    # Delivery checksums cover every exported artifact and reproduce exactly.
    for line in (out/"SHA256SUMS.txt").read_text().splitlines():
        digest,path=line.split("  ",1)
        assert module.sha256(out/path)==digest
    con.close()


def test_loader_contract(bundle):
    _,out=bundle
    spec=importlib.util.spec_from_file_location("local_cq_loader",out/"load_cq_v4_2.py")
    loader=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    assert loader.load().count("*").fetchone()[0]==8
    assert loader.load(phase="FULL_A").count("*").fetchone()[0]==2
    with pytest.raises(ValueError,match="locked"):
        loader.load("test")
    with pytest.raises(ValueError,match="Specify"):
        loader.load("dev")
    with pytest.raises(ValueError,match="train only"):
        loader.load("dev",phase="FULL_A")


def test_independent_sql_bundle_verification(bundle):
    _,out=bundle
    spec=importlib.util.spec_from_file_location("independent_cq_verifier",PIPELINE/"26_verify_phase_features_v4_2.py")
    verifier=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    report=verifier.verify(out)
    assert report["status"]=="PASS" and report["checks_passed"]>70


def test_duplicate_context_fails_before_feature_build(tmp_path):
    make_input(tmp_path)
    p=tmp_path/"input/moocubex_parquet/tables/entities/user/duplicate.parquet"
    sqlfile(tmp_path,p.relative_to(tmp_path),"SELECT 'U_1' id,1 gender,1990 year_of_birth")
    builder=module.Builder(tmp_path,tmp_path/"out",memory="512MB")
    try:
        with pytest.raises(AssertionError,match="enrollment_context_one_per_cohort"):
            builder.prepare()
        assert not (tmp_path/"out/_SUCCESS.json").exists()
    finally:
        builder.con.close()


def test_future_events_cannot_change_earlier_features():
    from test_cq_tempo_features import _data
    from cq_tempo_features import build_cd3_phase_features
    spine,video,problem,comment,*context=_data()
    baseline=build_cd3_phase_features(spine,video,problem,comment,*context)
    late=pd.Timestamp("2050-01-01")
    video=pd.concat([video,video.iloc[:1].assign(event_time=late)],ignore_index=True)
    problem=pd.concat([problem,problem.iloc[:1].assign(submit_time=late)],ignore_index=True)
    comment=pd.concat([comment,comment.iloc[:1].assign(event_time=late,comment_id="Cm_future")],ignore_index=True)
    actual=build_cd3_phase_features(spine,video,problem,comment,*context)
    pd.testing.assert_frame_equal(baseline,actual)


@pytest.fixture(scope="module")
def complete_bundle(tmp_path_factory):
    root=tmp_path_factory.mktemp("cd3_test_extension")
    make_input(root,include_test=True)
    out=root/"output"
    module.Builder(root,out,memory="512MB",batch_size=2).run()
    spec=importlib.util.spec_from_file_location("cq_test_extension",PIPELINE/"27_build_test_features_v4_2.py")
    extension=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extension)
    before={p:module.sha256(out/p) for p in extension.FROZEN_POLICY}
    manifest=json.loads((out/"run_manifest.json").read_text())
    before.update({p:module.sha256(out/p) for p in manifest["raw_paths"]+manifest["ready_paths"]})

    def forbidden(*args,**kwargs):
        raise AssertionError("Test extension must never fit or run QA23")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(extension.CQRawEarlyViews,"fit",forbidden)
        patch.setattr(extension.TempoV1Preprocessor,"fit",forbidden)
        patch.setattr(extension.base,"qa23_audit",forbidden)
        report=extension.build_test(root,out,memory="512MB",batch_size=2)
    return root,out,before,report


def test_test_extension_preserves_train_artifacts_and_policy(complete_bundle):
    root,out,before,report=complete_bundle
    assert report["status"]=="PASS"
    assert report["test_used_for_fit"] is report["test_used_for_qa23"] is False
    assert before=={p:module.sha256(out/p) for p in before}
    manifest=json.loads((out/"run_manifest.json").read_text())
    assert manifest["test_data_read"] is manifest["test_in_bundle"] is True
    assert len(manifest["raw_paths"])==len(manifest["ready_paths"])==10
    assert json.loads((out/"provenance/pre_test_extension/run_manifest.json").read_text())["test_in_bundle"] is False
    con=duckdb.connect()
    for i in range(1,5):
        raw=con.read_parquet(str(out/f"raw_features/pool_b/test_phase{i}.parquet")).df()
        ready=con.read_parquet(str(out/f"model_ready/pool_b/test_phase{i}.parquet")).df()
        assert raw.row_id.tolist()==[f"e4P{i}"]
        assert raw.q_alpha_1.tolist()==[.9] and raw.problem_count.tolist()==[1]
        assert raw.comment_count.tolist()==[1]
        assert raw.segment_duration_seconds.tolist()==[1e6]
        assert ready.gender.tolist()==[0]  # test category 42 is absent from train
        assert ready.phase_fraction.tolist()==[[.25,.5,.75,.9][i-1]]
    con.close()


def test_complete_bundle_loader_and_independent_verifier(complete_bundle):
    _,out,_,_=complete_bundle
    spec=importlib.util.spec_from_file_location("complete_cq_loader",out/"load_cq_v4_2.py")
    loader=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    assert loader.load("test",phase="P3").count("*").fetchone()[0]==1
    assert "phase_fraction" not in loader.load("test",phase="P3",view="selector").columns
    assert "video_observed_mask" in loader.load("test",phase="P1",raw=True).columns
    with pytest.raises(ValueError,match="Specify test phase"):
        loader.load("test")
    with pytest.raises(ValueError,match="train only"):
        loader.load("test",phase="FULL_A")
    spec=importlib.util.spec_from_file_location("complete_cq_verifier",PIPELINE/"26_verify_phase_features_v4_2.py")
    verifier=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    report=verifier.verify(out)
    assert report["status"]=="PASS"
    assert sum(n for _,s,n in report["counts"] if s=="test")==4


def teacher_builder():
    spec=importlib.util.spec_from_file_location('teacher_subgroup_builder',PIPELINE/'29_build_teacher_subgroup_v4_2.py')
    builder=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    return builder


def test_teacher_subgroup_rules_use_train_relation_ids():
    builder=teacher_builder()
    con=duckdb.connect()
    try:
        con.execute("""CREATE VIEW courses AS SELECT course_id,course_id course_family_id,split FROM
            (VALUES ('train','train'),('train_missing','train'),('seen','dev'),('new','dev'),
                    ('mixed','test'),('invalid','test'),('shared','test')) c(course_id,split)""")
        con.execute("""CREATE VIEW source_course_teacher AS SELECT * FROM (VALUES
            ('train',' T_1 '),('train','T_1'),('train','T_2'),('train',NULL),
            ('seen','T_1'),('seen','T_2'),('new','T_new'),('outside_cohort','T_new'),
            ('mixed','T_2'),('mixed','T_unseen'),('invalid',NULL),('invalid','  '),
            ('shared','T_new')) t(course_id,teacher_id)""")
        # No teacher profile table exists: missing profiles must not discard IDs.
        builder.derive_teacher_subgroups(con)
        rows={r[0]:r[1:] for r in con.execute('SELECT course_id,teacher_subgroup,n_teachers,n_seen_teachers,n_new_teachers FROM teacher_groups').fetchall()}
        assert rows=={'train':('seen',2,2,0),'train_missing':('missing',0,0,0),
                      'seen':('seen',2,2,0),'new':('new',1,0,1),'mixed':('mixed',2,1,1),
                      'invalid':('missing',0,0,0),'shared':('new',1,0,1)}
        assert con.execute('SELECT teacher_id FROM train_teacher_ids ORDER BY 1').fetchall()==[('T_1',),('T_2',)]
    finally:
        con.close()


@pytest.mark.parametrize('teacher,expected',[(None,'missing'),('T_new','new')])
def test_teacher_subgroup_empty_train_reference(teacher,expected):
    con=duckdb.connect()
    try:
        con.execute("CREATE VIEW courses AS SELECT * FROM (VALUES ('train','f1','train'),('dev','f2','dev')) c(course_id,course_family_id,split)")
        con.execute('CREATE TABLE source_course_teacher(course_id VARCHAR,teacher_id VARCHAR)')
        con.execute('INSERT INTO source_course_teacher VALUES (?,?)',['dev',teacher])
        teacher_builder().derive_teacher_subgroups(con)
        assert con.execute("SELECT teacher_subgroup FROM teacher_groups WHERE course_id='dev'").fetchone()[0]==expected
        assert con.execute('SELECT count(*) FROM train_teacher_ids').fetchone()[0]==0
    finally:
        con.close()


def test_teacher_subgroup_delivery_preserves_matrices_and_loader(complete_bundle,tmp_path):
    source,out,_,_=complete_bundle
    clone=tmp_path/'output'
    shutil.copytree(out,clone)
    mapping=Path('input/moocubex_parquet/tables/relations/course_teacher')
    shutil.copytree(source/mapping,tmp_path/mapping)
    builder=teacher_builder()
    manifest=json.loads((clone/'run_manifest.json').read_text())
    protected=manifest['raw_paths']+manifest['ready_paths']+builder.POLICY_FILES
    before={p:module.sha256(clone/p) for p in protected}
    report=builder.build(tmp_path,clone)
    assert report['status']=='PASS' and report['model_input_sha256']==before
    assert before=={p:module.sha256(clone/p) for p in before}
    reference=json.loads((clone/builder.PATHS['reference']).read_text())
    assert reference['teacher_ids']==['T_1']
    spec=importlib.util.spec_from_file_location('teacher_metadata_loader',clone/'load_cq_v4_2.py')
    loader=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    assert 'teacher_subgroup' not in loader.load(root=clone).columns
    for split,phase in [('train','P1'),('dev','P2'),('test','P3'),('train','FULL_A')]:
        default=loader.load(split,phase,root=clone).order('row_id').df()
        joined=loader.load(split,phase,root=clone,include_teacher_subgroup=True).order('row_id').df()
        pd.testing.assert_frame_equal(default,joined[default.columns])
        assert joined.teacher_subgroup.tolist()==(['seen']*len(joined) if split=='train' else ['new'])
        assert joined.n_teachers.tolist()==[1]*len(joined)
    assert set(builder.META_COLUMNS)<=set(loader.load('dev','P1',raw=True,root=clone,include_teacher_subgroup=True).columns)
    assert not set(builder.META_COLUMNS)&set(loader.feature_columns(root=clone))
    spec=importlib.util.spec_from_file_location('teacher_metadata_verifier',PIPELINE/'26_verify_phase_features_v4_2.py')
    verifier=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    assert verifier.verify(clone)['status']=='PASS'
    first_hashes={p:module.sha256(clone/p) for p in builder.PATHS.values()}
    builder.build(tmp_path,clone)
    assert first_hashes=={p:module.sha256(clone/p) for p in first_hashes}
    # Changing the source cannot silently redefine the frozen train reference.
    completion=module.sha256(clone/'_SUCCESS.json')
    sqlfile(tmp_path,mapping/'extra.parquet',"SELECT 'C_1' course_id,'T_2' teacher_id")
    with pytest.raises(AssertionError,match='frozen_train_teacher_reference'):
        builder.build(tmp_path,clone)
    assert module.sha256(clone/'_SUCCESS.json')==completion
