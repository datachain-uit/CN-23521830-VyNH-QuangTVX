"""Generate the V4.2 data card and measured evidence, preserving all model inputs."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil

import duckdb
import numpy as np
import pandas as pd

from cq_tempo_features import VIDEO_FEATURES, PROBLEM_FEATURES, COMMENT_FEATURES, STATIC_FEATURES

ROOT = Path(__file__).resolve().parent
DATE = "2026-10-30"
CARD_NAME = "DATA_CARD_V4_2.md"
CARD_REVISION = "v2"
POLICY_FILES = ["preprocessing.json", "feature_columns.json", "learner_feature_columns.json",
                "selector_feature_columns.json", "audit/qa23_by_phase.json", "audit/held_out_columns.json"]
CTX_SOURCES = {
    "resource_count": ("entities/course.resource", "Số resource_id phân biệt của khóa"),
    "ex_counts": ("entities/course.resource", "Số resource_id bắt đầu Ex_"),
    "teacher_count": ("relations/course_teacher + entities/teacher", "Số teacher_id có profile ghép được"),
    "teacher_avg_bio_length_chars": ("relations/course_teacher + entities/teacher.about", "Mean độ dài about; hồ sơ trống nhận độ dài 0"),
    "teacher_org_count": ("relations/course_teacher + entities/teacher.org_name", "Số org_name phân biệt sau trim dấu cách"),
    "school_count": ("relations/course_school + entities/school", "Số school_id có profile ghép được"),
    "school_avg_bio_length_chars": ("relations/course_school + entities/school.about", "Mean độ dài about; hồ sơ trống nhận độ dài 0"),
    "school_avg_motto_length_chars": ("relations/course_school + entities/school.motto", "Mean độ dài motto; hồ sơ trống nhận độ dài 0"),
    "course_video_count": ("entities/course.resource", "Catalog video V_; NULL nếu chưa quan sát họ video"),
    "course_resource_count": ("entities/course.resource", "Catalog resource; NULL nếu chưa quan sát họ comment"),
    "course_problem_count": ("entities/problem + entities/course.resource", "Catalog problem theo exercise; NULL nếu chưa quan sát họ problem"),
    "course_problem_full_score_sum": ("entities/problem.score + entities/course.resource", "Tổng full_score catalog; NULL nếu chưa quan sát họ problem"),
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix(path.suffix+".partial")
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    tmp.replace(path)


def qs(value):
    return "'"+str(value).replace("'","''")+"'"


def records(con, sql, params=None):
    result = con.execute(sql,params or [])
    return [dict(zip([c[0] for c in result.description],r)) for r in result.fetchall()]


def num(value, decimals=4):
    if value is None:
        return "NULL"
    if isinstance(value,(int,np.integer)):
        return f"{int(value):,}".replace(",",".")
    return f"{float(value):.{decimals}f}".rstrip("0").rstrip(".").replace(".",",")


def table(headers, rows):
    def cell(v):
        return str(v).replace("|","\\|").replace("\n"," ")
    return "\n".join(["| "+" | ".join(headers)+" |", "|"+"---|"*len(headers)]+
                     ["| "+" | ".join(cell(v) for v in row)+" |" for row in rows])


def validate_index(bundle):
    marker = read_json(bundle/"_SUCCESS.json")
    if digest(bundle/"run_manifest.json") != marker["manifest_sha256"] or digest(bundle/"SHA256SUMS.txt") != marker["checksums_sha256"]:
        raise AssertionError("Bundle completion hashes do not match")
    for line in (bundle/"SHA256SUMS.txt").read_text().splitlines():
        expected,name = line.split("  ",1)
        if digest(bundle/name) != expected:
            raise AssertionError("Bundle checksum mismatch: "+name)


def collect(bundle, root, protected):
    manifest = read_json(bundle/"run_manifest.json")
    config = read_json(bundle/"feature_columns.json")
    prep = read_json(bundle/"preprocessing.json")
    con = duckdb.connect()
    con.execute("SET threads=2")
    con.execute("SET memory_limit='1000MB'")
    try:
        paths = "["+",".join(qs(bundle/p) for p in manifest["raw_paths"])+"]"
        con.execute("CREATE VIEW snapshots AS SELECT * FROM read_parquet("+paths+")")
        counts = records(con,"""SELECT phase,split,count(*) n_rows,count(DISTINCT enrollment_id) enrollments
            FROM snapshots GROUP BY 1,2 ORDER BY split,phase""")
        cohort = records(con,"""SELECT split,count(*) enrollments,count(DISTINCT user_id) users,
            count(DISTINCT course_id) courses,count(DISTINCT course_family_id) families,
            count(*) FILTER(WHERE y_alpha_1='E') E,count(*) FILTER(WHERE y_alpha_1='G') G,
            count(*) FILTER(WHERE y_alpha_1='A') A FROM snapshots WHERE phase='P1' GROUP BY 1 ORDER BY split""")
        null_cols = ",".join(f"count(*) FILTER(WHERE {c} IS NULL) AS null_{c}" for c in config["raw_feature_columns"])
        missing = records(con,"SELECT phase,split,count(*) n_rows,"+null_cols+","+
            ",".join(f"count(*) FILTER(WHERE {f}_observed_mask=0) absent_{f}" for f in ("video","problem","comment"))+
            ",count(*) FILTER(WHERE video_observed_mask+problem_observed_mask+comment_observed_mask=0) absent_all FROM snapshots GROUP BY 1,2 ORDER BY split,phase")
        schemas = {kind:records(con,"DESCRIBE SELECT * FROM read_parquet(?)",[str(bundle/manifest[key][0])])
                   for kind,key in [("raw","raw_paths"),("ready","ready_paths")]}
        # The denominator is selected segment x snapshot, and duration is real
        # watch seconds; all measurements use the frozen train/dev spine only.
        tails = records(con,"""WITH selected AS (
            SELECT s.phase,s.split,greatest(0.0,(v.end_point-v.start_point)/v.speed-
                epoch(s.cutoff_time-v.event_time)) tail_seconds
            FROM read_parquet(?) v JOIN read_parquet(?) s USING(enrollment_id)
            WHERE s.phase IN ('P1','P2','P3','P4') AND v.end_point>=v.start_point AND v.speed>0
                AND v.event_time BETWEEN s.enroll_time AND s.cutoff_time)
            SELECT phase,split,count(*) selected_segments,count(*) FILTER(WHERE tail_seconds>0) crossing_segments,
                sum(tail_seconds) total_tail_seconds,max(tail_seconds) max_tail_seconds,
                100.0*count(*) FILTER(WHERE tail_seconds>0)/count(*) crossing_percent
            FROM selected GROUP BY 1,2 ORDER BY split,phase""",
            [str(bundle/"_stage/video.parquet"),str(bundle/"_stage/spine.parquet")])
        candidates = [c for c in dict.fromkeys(config["learner"]+config["selector"])
                      if c in config["raw_feature_columns"] and c!="phase_fraction"]
        tr = con.execute("SELECT course_id,course_family_id,"+",".join(candidates)+
                         " FROM snapshots WHERE split='train' AND phase<>'FULL_A' ORDER BY row_id").df()
        grouped = tr.groupby("course_id",sort=False)[candidates]
        with_null,observed,overall = grouped.nunique(dropna=False).max(),grouped.nunique(dropna=True).max(),tr[candidates].nunique(dropna=True)
        constant = [c for c in candidates if with_null[c]==1 and overall[c]>1]
        conditional = [c for c in candidates if c not in constant and observed[c]<=1 and overall[c]>1]
        ctx = constant+conditional
        if set(ctx)!=set(CTX_SOURCES):
            raise AssertionError("Measured Gctx differs from the twelve documented source columns")
        dv = con.execute("SELECT course_id,course_family_id,"+",".join(ctx)+" FROM snapshots WHERE split='dev' AND phase='P4' ORDER BY row_id").df()
        train_courses = tr.groupby("course_id",sort=True)[ctx].first()
        dev_courses = dv.groupby("course_id",sort=True)[ctx].first()
        keys = lambda f:[tuple(r) for r in f.astype(float).round(9).fillna(-1e300).to_numpy()]
        tr_keys,dv_keys = keys(train_courses),keys(dev_courses)
        key_counts = Counter(tr_keys)
        quasi = {"basis":"raw train and raw dev P4; per-course first non-NULL; rounding 9 decimals",
                 "constant":constant,"constant_when_observed":conditional,"g_ctx":ctx,
                 "train_courses":len(train_courses),"train_distinct_vectors":len(key_counts),
                 "train_share_unique":float(np.mean([key_counts[k]==1 for k in tr_keys])),
                 "dev_courses":len(dev_courses),"dev_courses_matching_train":sum(k in key_counts for k in dv_keys)}
        lineage = []
        for c in ctx:
            source,definition = CTX_SOURCES[c]
            is_profile = "teacher" in source or "school" in source
            lineage.append({"column":c,"group":"constant" if c in constant else "constant_when_observed",
                            "source":"input/moocubex_parquet/tables/"+source,"definition":definition,
                            "source_time_basis":"profile snapshot at collection" if is_profile else "final course catalog snapshot",
                            "per_value_as_of_timestamp_available":False,
                            "known_available_before_cutoff":None})
        source_locks = [read_json(root/p) for p in ["afelo_sentiment_cardiff/model_lock.json","final_quality_v2/sentiment/model_lock.json"]]
        if source_locks[0]["revision"]!=source_locks[1]["revision"]:
            raise AssertionError("Sentiment cache revisions disagree")
        qa = read_json(bundle/"audit/qa23_by_phase.json")
        qa_summary = {"rows":len(qa),"columns":len({r['column'] for r in qa}),
                      "phases":sorted({r['phase'] for r in qa}),
                      "needs_review":sum(bool(r['needs_review']) for r in qa),
                      "hold_out_rows":sum(bool(r['hold_out']) for r in qa),
                      "approved_flagged_columns":sorted({r['column'] for r in qa if r['needs_review'] and not r['hold_out']}),
                      "held_out_columns":prep['held_out_columns']}
        evidence = {"dataset_version":manifest["dataset_version"],"model_input_sha256":protected,
                    "counts":counts,"cohort":cohort,"schemas":schemas,"null_and_absence":missing,
                    "video_tail_train_dev":tails,"g_ctx":quasi,"context_lineage":lineage,"qa23":qa_summary,
                    "sentiment":{"model_id":source_locks[0]['model_id'],"revision":source_locks[0]['revision'],
                        "weight_sha256":source_locks[0]['file_sha256']['pytorch_model.bin'],
                        "train_dev_full_a":read_json(bundle/"audit/sentiment_coverage.json"),
                        "test":read_json(bundle/"audit/test/sentiment_coverage.json") if manifest['test_in_bundle'] else []},
                    "temporal_overlap_frozen_v4":{k:v for k,v in read_json(root/"phase_features_v4/temporal_overlap.json").items() if k!='overlap_families'},
                    "frozen_v4_course_statistics":read_json(root/"phase_features_v4/course_statistics.json"),
                    "frozen_v4_protocol_constants":read_json(root/"phase_features_v4/protocol_constants.json"),
                    "measurement_scope":"Descriptive data audit; no fit, sentiment inference or model evaluation"}
        teacher = manifest.get('teacher_subgroup')
        if teacher:
            evidence['teacher_subgroup'] = {
                'contract':teacher,
                'reference':{k:v for k,v in read_json(bundle/teacher['reference_path']).items() if k!='teacher_ids'},
                'spec':read_json(bundle/teacher['spec_path']),
                'audit':read_json(bundle/teacher['audit_path'])}
        assert sum(r['n_rows'] for r in counts)==sum(r['enrollments']*4 for r in cohort)+sum(r['n_rows'] for r in counts if r['phase']=='FULL_A')
        assert len(config['raw_feature_columns'])==85 and len(config['learner'])==76 and len(config['selector'])==74
        return evidence
    finally:
        con.close()


def build_card_text(e, manifest, config, prep, date, confirmation, root):
    version = f"phase_features_v4_2-data-card-{date}-{CARD_REVISION}"
    by_split = {r['split']:r for r in e['cohort']}
    cohort_rows = [[s]+[num(by_split[s][k]) for k in ['families','courses','users','enrollments','E','G','A']]+[num(by_split[s]['enrollments']*4)] for s in ['train','dev','test'] if s in by_split]
    family_rows = []
    for title,features in [('Video',VIDEO_FEATURES),('Problem',PROBLEM_FEATURES),('Comment',COMMENT_FEATURES),('Static',STATIC_FEATURES)]:
        family_rows.append([title,len(features),", ".join('`'+c+'`' for c in features)])
    missing_rows = [[r['split'],r['phase'],num(r['n_rows'])]+[num(r['absent_'+f]) for f in ['video','problem','comment','all']]+[num(100*r['absent_all']/r['n_rows'])+'%'] for r in e['null_and_absence']]
    tail_rows = [[r['split'],r['phase'],num(r['selected_segments']),num(r['crossing_segments']),num(r['crossing_percent'],6)+'%',num(r['total_tail_seconds']),num(r['max_tail_seconds'])] for r in e['video_tail_train_dev']]
    lineage_rows = [["`"+r['column']+"`",r['group'],r['source'].removeprefix('input/moocubex_parquet/tables/'),r['definition'],
                     'Hồ sơ tại thời điểm thu thập' if r['source_time_basis'].startswith('profile') else 'Catalog cuối cùng'] for r in e['context_lineage']]
    files_rows = [["`"+raw+"`","`"+ready+"`"] for raw,ready in zip(manifest['raw_paths'],manifest['ready_paths'])]
    maps = prep['category_maps']['gender']
    source_md=(root/'BANG_GIAI_TRINH_CONG_THUC_FEATURE_VA_NHAN_CQ_V4_2026_10_23.md').read_text()
    part=source_md.split('## 10.')[1].split('## 11.')[0]
    adjustments=[]
    for line in part.splitlines():
        if not line.startswith('|'):
            continue
        cells=[c.strip() for c in line.strip('|').split('|')]
        if len(cells)==3 and cells[0] not in ['Nội dung','---']:
            adjustments.append(cells)
    overrides = {
        'Chuẩn hóa':['P95 dương và signed-P95 cho FA trong nhãn','Lựa chọn của phiên bản nhãn QĐ27; phạm vi reference là quần thể V2 đã đóng băng.'],
        'Video ở final so với phase':['Nhãn giữ quy tắc lịch sử; X V4.2 dùng full segment có thời điểm bắt đầu trong cửa sổ','X V4 trước đây clip segment; V4.2 theo helper CD3 và công bố số đo Y37.'],
        'Bộ X':['85 raw CD3; learner 76, selector 74','CD3 CQ_RAW_EARLY; bỏ ID, proxy theo allowlist và giữ coverage danh mục quan sát.'],
        'Scaler X':['V1-Median, log1p và robust median/IQR train P1–P4; cột tĩnh một dòng/enrollment','Thay median/mean/std của X V4; độc lập với P95 dùng định nghĩa nhãn.'],
    }
    for row in adjustments:
        if row[0] in overrides:
            row[1:]=overrides[row[0]]
    if len(adjustments)!=18:
        raise AssertionError('Expected the 18 disclosed adaptations')
    d06 = ('Đã được người dùng xác nhận hoàn tất; ngày ghi nhận xác nhận '+confirmation+'. '
           'Card ghi nhận xác nhận này và giữ nguyên hồ sơ kiểm thủ công gốc.'
           if confirmation else 'Trạng thái xác nhận lấy từ hồ sơ ký do nhóm dữ liệu quản lý; lần lập card không tự cấp xác nhận thủ công.')
    protected_rows = [["`"+p+"`","`"+e['model_input_sha256'][p]+"`"] for p in ['preprocessing.json','feature_columns.json','learner_feature_columns.json','selector_feature_columns.json']]
    sentiment_rows=[]
    for scope,key in [('train/dev/FULL_A','train_dev_full_a'),('test P1–P4','test')]:
        for r in e['sentiment'][key]:
            sentiment_rows.append([scope,num(r['n_comments']),num(r['with_sentiment']),num(r['empty_text']),num(r['nonempty_without_cached_sentiment'])])
    teacher_text = ('`teacher_subgroup` chưa được xuất thành metadata chính thức; '
                    'chạy bước 29 để bổ sung sidecar theo course_id, giữ ngoài X.')
    if e.get('teacher_subgroup'):
        teacher = e['teacher_subgroup']
        summary = {(r['split'],r['teacher_subgroup']):r['courses'] for r in teacher['audit']['summary']}
        rows = [[s]+[num(summary.get((s,g),0)) for g in ['seen','new','mixed','missing']] for s in ['train','dev','test'] if s in by_split]
        teacher_text = ('**teacher_subgroup chính thức:** phiên bản `'+teacher['contract']['version']+'`, '
            'bảng [metadata theo course_id](phase_features_v4_2/metadata/teacher_subgroup.parquet). '
            'Reference gồm '+num(teacher['reference']['n_teacher_ids'])+' teacher ID phân biệt của '+
            num(teacher['reference']['n_train_courses'])+' khóa train, đóng băng riêng trong '
            '[train reference](phase_features_v4_2/metadata/train_teacher_reference.json). '
            'Không dùng teacher ID từ dev/test để mở rộng reference.\n\n'
            '- `seen`: có giảng viên và toàn bộ ID nằm trong reference train.\n'
            '- `new`: có giảng viên và không ID nào nằm trong reference train.\n'
            '- `mixed`: có cả ID đã gặp và chưa gặp.\n'
            '- `missing`: không có teacher ID dùng được trong relation mapping; không kết luận khóa không có giảng viên.\n\n'+
            table(['Split','seen','new','mixed','missing'],rows)+'\n\n'
            'Dùng relation `course_teacher`, trim ID, loại NULL/chuỗi rỗng và khử trùng course–teacher. '
            'ID vẫn hợp lệ khi thiếu profile trong `entities/teacher`; không suy nhóm từ code danh mục 0. '
            'Sidecar có `course_id`, `course_family_id`, `split`, `teacher_subgroup`, `n_teachers`, '
            '`n_seen_teachers`, `n_new_teachers`; một dòng/course, giữ cùng nhóm ở mọi enrollment/phase. '
            'Parquet raw/ready và feature allowlist giữ nguyên. Loader chỉ ghép bốn cột metadata '
            'khi gọi `include_teacher_subgroup=True`; các cột này giữ ngoài X. '
            '[Quy tắc và nguồn](phase_features_v4_2/metadata/teacher_subgroup_spec.json), '
            '[mapping chuẩn hóa](phase_features_v4_2/metadata/course_teacher_mapping.parquet) và '
            '[audit](phase_features_v4_2/audit/teacher_subgroup_validation.json) có hash trong manifest. '
            'Mapping là snapshot cuối cùng, không có lịch sử as-of; nhóm này phục vụ phân tích hồi cứu.')
    items={
        'DATE':date,'VERSION':version,'DATASET':manifest['dataset_version'],'LABEL':manifest['label_version'],
        'CD3_COMMIT':manifest['cd3_commit'],'COHORT':table(['Split','Family','Course','User','Enrollment','E','G','A','Snapshot P1–P4'],cohort_rows),
        'FEATURES':table(['Họ','Số cột raw','Danh sách'],family_rows),
        'MISSING':table(['Split','Phase','Số dòng','Vắng video','Vắng problem','Vắng comment','Vắng cả ba','Tỷ lệ vắng cả ba'],missing_rows),
        'TAILS':table(['Split','Phase','Segment được chọn','Vượt cutoff','Tỷ lệ','Tổng giây thực sau cutoff','Tối đa giây'],tail_rows),
        'LINEAGE':table(['Cột','Nhóm','Nguồn','Phép tính','Nguồn thời gian'],lineage_rows),
        'FILES':table(['Raw','Ready'],files_rows),'ADAPTATIONS':table(['Nội dung','Cách tính hiện hành','Phạm vi và giới hạn'],adjustments),
        'HASHES':table(['Artifact bất biến','SHA256'],protected_rows),
        'SENTIMENT_TABLE':table(['Phạm vi staged event','Comment','Có cache sentiment','Text rỗng','Text không rỗng thiếu cache'],sentiment_rows),
        'MODEL':e['sentiment']['model_id'],'REVISION':e['sentiment']['revision'],'WEIGHT_SHA':e['sentiment']['weight_sha256'],
        'QA_ROWS':str(e['qa23']['rows']),'QA_FLAGS':str(e['qa23']['needs_review']),
        'QA_COLUMNS':', '.join('`'+c+'`' for c in e['qa23']['approved_flagged_columns']),
        'GCTX_TRAIN':num(e['g_ctx']['train_courses']),'GCTX_VECTORS':num(e['g_ctx']['train_distinct_vectors']),
        'GCTX_UNIQUE':num(100*e['g_ctx']['train_share_unique'])+'%','GCTX_DEV':num(e['g_ctx']['dev_courses']),
        'GCTX_MATCH':num(e['g_ctx']['dev_courses_matching_train']),'D06':d06,
        'FIT_ROWS':num(prep['fit_rows']),'STATIC_FIT':num(prep['static_fit_rows']),
        'GENDER_MAP':', '.join(f'`{k}`→`{v}`' for k,v in maps.items()),
        'ALL_MISSING':', '.join(prep['all_missing_numeric_train']) or 'không có',
        'LOG_COLUMNS':', '.join('`'+c+'`' for c in prep['log_columns']),
        'SOFTWARE':table(['Thành phần','Môi trường build ghi trong manifest'],[[k,v] for k,v in manifest['software'].items()]),
        'SNAPSHOTS':num(sum(r['n_rows'] for r in e['counts'])),
        'TEACHER_SUBGROUP':teacher_text,
    }
    text=DOCUMENT
    for key,value in items.items():
        text=text.replace('@@'+key+'@@',value)
    if '@@' in text:
        raise AssertionError('Unresolved data card placeholder')
    return text,version


DOCUMENT = r'''# Data card Phase Features V4.2 theo công thức CQ CD3

**Ngày cập nhật:** @@DATE@@. **Phiên bản tài liệu:** `@@VERSION@@`.

Bộ dữ liệu `@@DATASET@@` cung cấp snapshot hành vi tích lũy để nghiên cứu dự đoán sớm chất lượng vận hành của khóa học và đánh giá các phương pháp coreset trên khóa học mới. Tên gọi `phase_feature_v4_2` trong trao đổi chỉ cùng bộ dữ liệu có tên contract và thư mục là `phase_features_v4_2`. Công thức feature CQ lấy từ CD3; cohort, nhãn, family/split, khóa dòng và cutoff kế thừa V4. Bản hiện hành có train, dev, test P1–P4 và FULL_A train.

Card này ghi cách tính đang chạy và bằng chứng đo từ artifact. Ma trận feature và mọi tham số fit được giữ nguyên khi lập card; các phép đo mô tả không thay quyết định đánh giá mô hình. Trạng thái xác nhận D06 được ghi ở mục 11; phê duyệt M0/M1 và freeze thực nghiệm được quản lý theo hồ sơ giảng viên.

## 1 Định danh và nguồn gốc

| Thuộc tính | Giá trị |
|---|---|
| Dataset version | `@@DATASET@@` |
| Đơn vị quản lý | Nhóm CN-23521830-VyNH-QuangTVX |
| Dữ liệu gốc | Snapshot MOOCubeX đã chuyển thành Parquet trong kho dữ liệu riêng |
| Spine | V4 supervised có nhãn; giữ metadata và split theo family |
| Feature | CQ CD3, commit `@@CD3_COMMIT@@`; helper pandas giữ công thức đã giao trong QĐ28 |
| Phiên bản nhãn | `@@LABEL@@` theo QĐ27 |
| Target chính | `y_alpha_1`; `q_alpha_1` dùng audit/EC, nằm ngoài X |
| Target độ nhạy | `q_alpha_0p1`, `y_alpha_0p1`; không thay target chính theo kết quả dev |
| View chính | Learner 76 cột; selector 74 cột |
| Trạng thái dữ liệu | Đã build và kiểm tra kỹ thuật; có test theo yêu cầu bổ sung của người dùng |

[CD3 tại commit nguồn](https://github.com/datachain-uit/CD3/tree/@@CD3_COMMIT@@), [manifest](phase_features_v4_2/run_manifest.json), [feature contract](phase_features_v4_2/feature_columns.json) và [bằng chứng data card](phase_features_v4_2/audit/data_card_evidence.json) xác định phiên bản thực dùng. Đây là bản CQ thích nghi trên khung V4, không nhập nhãn, lịch mở khóa, tác vụ LO hoặc các biến thể imputation khác của CD3.

## 2 Quần thể và cách chia tập

V4.2 dùng toàn bộ 178.598 enrollment có nhãn đã giữ sau lọc V3/V4, thuộc 1.808 course và 1.740 family. Không tạo feature V4.2 cho toàn bộ 10.455.017 cặp đủ thời gian của kho final V4; 10.276.419 cặp chưa có nhãn vẫn ở kho nguồn và không được gán thành lớp A. Enrollment chỉ có một chapter quan sát và chapter đó là unknown `-1` đã bị loại ở V3; trường hợp có cả chapter đã biết và `-1` được giữ. Các bộ lọc này kế thừa, không chạy lại để chọn cohort thuận lợi hơn.

@@COHORT@@

Pool B train có 527.624 snapshot. Dev có 22.717 dòng/phase; test có 23.975 dòng/phase. FULL_A có 131.906 dòng, một dòng/enrollment train, không ghép vào Pool B. Tổng gói có @@SNAPSHOTS@@ dòng tính cả FULL_A. Bốn snapshot của một enrollment có cùng nhãn, không phải bốn người độc lập.

Family là connected component của graph tương đồng tập ccid video, Jaccard ≥0,9, tối thiểu 5 ccid để xét gộp. Test lấy ceil(20% family) mở muộn nhất theo `(family_start, family_id)`. Trong phần còn lại, dev lấy theo seed 42, tầng quartile kích thước family × có lớp E, với quy tắc làm tròn của V4; train là phần còn lại. Không chia lại theo selector seed. Family/course/enrollment không giao nhau giữa ba split; user có thể xuất hiện ở nhiều split.

Độ nhạy ngưỡng Jaccard 0,8 của V4 tạo 1.731 family, 18 course đổi nhóm; split chính giữ 0,9. Temporal overlap kế thừa V4 là 924/1.219 family train = 75,7998% còn hoạt động sau mốc mở family test đầu tiên. Cách chia này không bảo đảm mọi khóa train kết thúc trước test; đối chứng NOLFO thuộc phân tích sau của giao thức.

## 3 Grain khóa và schema

Khóa duy nhất cho snapshot là `row_id`, tương đương `(enrollment_id, phase)` trên cohort hiện hành. `enrollment_id` là hash SHA256 JSON của `(user_id, course_id)`; `row_id` là hash SHA256 JSON của enrollment và phase. Giá trị ID được sao chép từ V4, không sinh lại trong V4.2. Dùng `row_id` để ghép hai phiên bản; `enrollment_id` một mình có nhiều phase.

| Vai trò | Cột |
|---|---|
| Metadata, 10 cột | `user_id`, `course_id`, `course_family_id`, `enrollment_id`, `row_id`, `phase`, `split`, `enroll_time`, `cutoff_time`, `course_end` |
| Target, 4 cột | `q_alpha_0p1`, `y_alpha_0p1`, `q_alpha_1`, `y_alpha_1` |
| Feature raw | 85 cột trong `feature_columns.json.raw_feature_columns` |
| Mask audit | `video_observed_mask`, `problem_observed_mask`, `comment_observed_mask` |
| Kênh phase | `phase_fraction`: P1=0,25; P2=0,50; P3=0,75; P4=0,90; FULL_A=1 |

Mỗi raw table có 103 cột: 10 metadata + 4 target + 85 raw + 3 mask + phase_fraction. Ready có 90 cột: 10 metadata + 4 target + 76 learner. Selector là subset 74 cột của ready. Schema vật lý đầy đủ được lưu trong evidence JSON. `enroll_time` đổi kiểu DATE của V4 thành TIMESTAMP đầu ngày; giá trị ngày và cutoff giữ nguyên. ID/phase/split là chuỗi; thời gian là timestamp không timezone; `most_common_day` là ngày lịch dưới dạng timestamp trong raw.

@@TEACHER_SUBGROUP@@

## 4 Nhãn chất lượng vận hành và reference P95

Nhãn kế thừa nguyên giá trị V4. `q_alpha_1` là khoảng cách trung bình tới trạng thái chỉ số lý tưởng, không phải xác suất: q thấp hơn tương ứng mức chất lượng vận hành tốt hơn theo định nghĩa này. Tại chapter, `LOQV_alpha = sqrt(sum_available(w*(1-index)^2)/sum_available(w))`, với các index COELO/AFELO/ACELO và `w=(1,alpha,1)`. q của enrollment là mean LOQV trên các chapter có điểm. Thành phần NULL được bỏ khỏi tử/mẫu theo công thức; 0 hợp lệ được giữ.

| Nhãn | Khoảng q |
|---|---|
| E | `q < 0.2` |
| G | `0.2 <= q < 0.4` |
| A | `q >= 0.4` |

q thiếu thì y thiếu ở kho nguồn; cohort supervised V4.2 chỉ gồm enrollment có nhãn. COELO là mean TS/RV/AV/IF khả dụng, cần ít nhất 2/4; AFELO là mean FV/FA khả dụng; ACELO dùng tổng trọng số các problem đúng chia catalog problem, với trọng số attempt `max(0.1,1-0.1*(attempts-1))`.

**Reference nhãn:** P95 dương của TS/IF/FV và P95 trị tuyệt đối khác 0 của FA được đóng băng trên quần thể final V2, trước lọc V3 và trước family split V4. Reference có đóng góp của enrollment sau này thuộc dev/test V4. Đây là một phần định nghĩa nhãn cố định; không mô tả P95 này là fit chỉ trên train V4 và không đồng nhất nó với scaler X. V4.2 không refit reference, không suy luận lại sentiment và không tính lại nhãn.

Nhãn diễn tả chất lượng vận hành từ hành vi và các proxy đã công bố theo QĐ27; không phải nhãn chuyên gia hoặc kết quả đánh giá sư phạm độc lập. Các thích nghi từ công thức tham khảo được liệt kê ở mục 12 và [bảng giải trình nhãn](BANG_GIAI_TRINH_CONG_THUC_FEATURE_VA_NHAN_CQ_V4_2026_10_23.md).

## 5 Cửa sổ và làm sạch event

`cutoff_p = enroll_time + lambda_p * (course_end-enroll_time)` kế thừa chính xác độ phân giải microsecond của V4. Event được chọn theo điều kiện đóng hai đầu `enroll_time <= availability_time <= cutoff_time`. Video/comment dùng `event_time`; problem dùng `submit_time`. Enrollment là đầu ngày ghi danh. `course_end` là cuối ngày hoạt động cuối quan sát của khóa, không phải lịch đóng chính thức đã biết trước.

| Họ | Mapping và điều kiện thực tế |
|---|---|
| Video | `user_video.seq.segment` → tài nguyên V_ của course → enrollment theo user/course. Có start/end, end≥start, speed>0 và thời gian hợp lệ. Giữ zero-length và start âm nếu thỏa helper. Không tự đổi giờ Asia/Shanghai như CD3 upstream; theo cùng đồng hồ UTC-naive của V4. Các input hợp lệ có giá trị non-finite bị guard, không âm thầm sửa công thức |
| Problem | Raw user_problem → catalog entity problem theo exercise trong resource course → enrollment. `full_score` lấy score của entity problem, `context_id` là toàn bộ mảng JSON, không explode hoặc thay bằng chapter. Cần submit_time, full_score>0, attempts không thiếu và ≥0. Score thiếu/không parse được nhận 0; correctness đúng khi bằng 1, các giá trị khác kể cả NULL nhận 0 |
| Comment | Tập trước lọc text, mapping course-comment hợp với resource-course; khử trùng theo comment/course/enrollment. Giữ text rỗng và các ID khác nhau có cùng text. Không lấy reply làm comment. Xác suất/label sentiment ghép từ cache theo hash text sau preprocessing |
| Context | Catalog dedup theo course/resource; catalog problem theo course/problem. Hồ sơ teacher/school dùng mapping và inner join tới profile; đối tượng thiếu profile không được thay bằng categorical code |

Các nhóm feature tích lũy đến cutoff. Thời lượng segment là `end-start`; thời gian thực là `(end-start)/speed`; segment được lấy toàn bộ khi thời điểm bắt đầu hợp lệ, kể cả đuôi sau cutoff. Phiên mới khi gap giữa hai điểm bắt đầu liên tiếp >30 phút; duration phiên lấy từ đuôi đầy đủ. Day offset dùng hiệu ngày lịch; std/variance dùng sample statistics của helper, với các ngoại lệ singleton nêu trong code.

Không dùng aggregate V4 đổi tên để thay cho CD3: validity, NULL, denominator, comment dedup và clipping có khác biệt. Ví dụ problem_count là distinct problem; `score_fraction_mean` là mean của score/full_score; `problem_score_catalog_coverage` dùng tổng điểm catalog của cả course.

## 6 Inventory feature và hai view

@@FEATURES@@

Raw 85 cột phục vụ audit và tái tạo. Theo CQ_RAW_EARLY, X loại `score_fraction_mean/std`, `correct_ratio/std`, ba coverage hồ sơ tĩnh (`teacher_bio_coverage`, `school_bio_coverage`, `school_motto_coverage`), `video_counts`, `year_of_birth`, `age_at_enroll`, `most_common_day`, cùng mọi ID/target/mask/metadata. `most_common_hour` được thay bằng sin/cos chu kỳ 24 giờ. Coverage catalog quan sát đến cutoff được giữ theo đúng quy tắc CD3.

Selector có 74 cột số đã qua V1. Learner dùng cùng 74 giá trị đó, thêm gender dạng code train và phase_fraction giữ nguyên. Thứ tự cột chuẩn nằm ở [learner allowlist](phase_features_v4_2/learner_feature_columns.json) và [selector allowlist](phase_features_v4_2/selector_feature_columns.json); không lấy tất cả cột numeric của Parquet để làm X. Với LR, bước one-hot gender thuộc adapter thực nghiệm đã đăng ký, không được thay bằng coi code là thang số liên tục.

## 7 V1 Median và thiếu dữ liệu

Fit duy nhất trên @@FIT_ROWS@@ snapshot train P1–P4, thứ tự row_id tăng dần. Cột tĩnh của helper dùng một dòng cho mỗi @@STATIC_FIT@@ enrollment train; cột theo phase dùng toàn pool. Đây không phải fit một dòng mỗi course. `course_video_count`, `course_problem_count`, `course_problem_full_score_sum`, `course_resource_count` thuộc nhóm theo phase của helper vì có mask hoạt động.

Với cột numeric j: log1p chỉ áp cho giá trị không âm khi tên khớp regex `(count|counts|sum|seconds|duration|length|attempts)`; tiếp đó `z=(logged_x-median_train)/IQR_train`. IQR nhỏ hơn `10*epsilon_float64` được thay bằng 1; ô NULL nhận 0 sau chuẩn hóa, bằng median train. Cột không có quan sát train nhận 0; danh sách thực tế: @@ALL_MISSING@@. Không thêm missing indicator vào X. Giá trị âm hữu hạn không được tự clip/log nếu helper không yêu cầu.

Danh sách log thực tế có trong preprocessing và được chốt như sau: @@LOG_COLUMNS@@.

Gender maps hiện hành là @@GENDER_MAP@@. Code được gán theo thứ tự xuất hiện của token trong train; NULL là token riêng nếu xuất hiện trong train; token chưa gặp nhận code 0. Không diễn giải các số code này thành nghĩa giới tính chưa được schema nguồn xác nhận. Phase_fraction không chuẩn hóa trong V4.2. Dev, test và FULL_A chỉ transform bằng cùng artifact train.

Không có event hợp lệ của một họ trước cutoff thì toàn bộ feature của họ là NULL, kể cả các catalog count ghép bên trong họ; mask audit=0. Có event thì mask=1, nhưng std singleton hoặc text/sentiment thiếu vẫn có thể NULL. Mask không thuộc X. Bảng sau đếm vắng họ bằng mask; tỷ lệ NULL cho từng feature được lưu đầy đủ trong evidence JSON.

@@MISSING@@

## 8 QA23 và kiểm soát proxy

QA23 chỉ chạy trên raw Pool B train, riêng P1–P4, với q_alpha_1; hour sin/cos được tạo trước audit, chưa qua V1. Audit có @@QA_ROWS@@ cặp cột–phase. Correlation dùng rank(method=min) như helper và NMI dùng ntile; NMI có thể nhạy với tie và thứ tự dòng, không dùng làm ngưỡng holdout.

Có @@QA_FLAGS@@ cặp có |rho|≥0,5, đều thuộc tám cột hành vi đã được CD3 duyệt: @@QA_COLUMNS@@. Holdout hiện rỗng, nên main view vẫn 76/74 và qa23_all trùng main. Quy tắc chung là union cột chưa được duyệt bị gắn cờ ở bất kỳ phase train nào, bỏ khỏi cả hai view; raw luôn giữ đủ inventory. Không chạy lại QA23 trên test để chọn feature.

## 9 Y37 và nguồn thời gian Gctx

### Đuôi video sau cutoff

Mỗi dòng đo là một segment được chọn trong một snapshot. Tổng giây sau cutoff là `max(0,(end-start)/speed-(cutoff-event_time))`, đơn vị giây thực; tỷ lệ dùng mẫu số selected segment của đúng split/phase. Không cộng các phase rồi coi là các segment độc lập.

@@TAILS@@

Dev không có đuôi vượt cutoff. Train có tỷ lệ cao nhất 0,006966% tại P1; một đuôi ở P3 dài 810,55 giây. Tỷ lệ nhỏ vẫn là giới hạn thời gian được công bố. Số đo này ghi nhận cách helper hiện hành vận hành, không thay thế clipping hoặc tự sửa X sau khi đã có panel.

### Mười hai cột ngữ cảnh

@@LINEAGE@@

Các nguồn đều là snapshot catalog cuối cùng hoặc hồ sơ ở thời điểm thu thập. Schema được dùng không có lịch sử phiên bản/per-value as-of timestamp để chứng minh từng giá trị đã có trước cutoff. File mtime hoặc thời gian chạy builder không phải bằng chứng về thời điểm nội dung gốc có sẵn. Bốn cột constant_when_observed được che bằng NULL khi vắng họ event, nhưng giá trị catalog bên dưới vẫn lấy từ snapshot cuối cùng. Đây là giới hạn hồi cứu bổ sung vào giới hạn cutoff dựa last activity.

Audit lại theo QĐ35 trên raw train và dev P4, một dòng/course lấy giá trị không thiếu đầu tiên, round 9 chữ số để đối chiếu: @@GCTX_TRAIN@@ course train có @@GCTX_VECTORS@@ vector, @@GCTX_UNIQUE@@ course train có vector duy nhất; @@GCTX_MATCH@@/@@GCTX_DEV@@ course dev trùng vector train. Audit này mô tả quasi-identifier, không đổi family hoặc tự loại Gctx khỏi main view. Kết quả cd3_minus_ctx của nhóm thực nghiệm được review 27/10 ghi nhận: bỏ Gctx làm ULA8 FullTrain-B tăng ở cả ba seed; giữ kết luận theo QĐ35(e) và không nhập kết quả mô hình đó thành chứng nhận của lần build dữ liệu này.

## 10 Sentiment và provenance văn bản

Mô hình cache là `@@MODEL@@`, revision `@@REVISION@@`, hash trọng số được lock là `@@WEIGHT_SHA@@`. Hai cache gốc và bổ sung V2 có cùng revision. Lần build V4.2 chỉ đọc cache; không tải hoặc suy luận lại trọng số.

Khóa cache là SHA256 của text sau preprocessing mention/URL: token @ được chuẩn thành @user, token bắt đầu http thành http, tách/nối theo dấu cách như code. Text input lấy từ entity translated_comment đã có trong snapshot; builder không thực hiện dịch mới. Phương pháp và thời điểm của bước dịch trước đó chưa được metadata nguồn này xác lập. Cache label được đưa về lower-case; positive/neutral/negative là nhãn sentiment, không phải E/G/A của target chất lượng. Mean/std xác suất và count/ratio label được dùng theo từng công thức feature CD3.

@@SENTIMENT_TABLE@@

Các số trên đếm event trong cửa sổ staged rộng nhất của từng enrollment, không phải cộng lại bốn phase. Comment không rỗng trong hai phạm vi đều ghép được cache; một comment text rỗng ở train/dev/FULL_A vẫn được giữ và không có sentiment. Model English/Twitter và text giáo dục đã dịch có khác biệt miền; sentiment là proxy trong định nghĩa vận hành. Rating và sentiment reply không được dựng thêm.

## 11 Kiểm tra trạng thái sử dụng và giới hạn đánh giá

Kiểm tra build ban đầu và bổ sung test có 1.163 check PASS. Kiểm tra độc lập bao gồm schema/khóa, metadata-target alignment, raw→V1 bằng SQL, tính lại median/IQR train, hash đầu vào, loader, và teacher_subgroup khi có metadata. Báo cáo mới nhất nằm trong [validation report](outputs/phase_features_v4_2_validation.json); số check bao gồm tài liệu và metadata có thể tăng, không phải số lượt chạy mô hình. Bộ test source kiểm tra thêm các trường hợp seen/new/mixed/missing, reference train và tính toàn vẹn của join.

**D06:** @@D06@@ Audit CD3 [20 cặp × 4 phase](phase_features_v4_2/audit/d06_comparison.json) là kết quả tự động lưu từ lúc build, giữ cờ lịch sử human_signoff=false; trạng thái xác nhận được ghi riêng trong evidence của card theo phát biểu người dùng. Không dùng kiểm tra máy để tự tạo chữ ký hoặc sửa phiếu gốc.

Các hạn chế cần giữ khi báo cáo gồm: cohort có nhãn chỉ chiếm khoảng 1,7083% kho cặp đủ thời gian; mất cân bằng lớp E; quan sát lặp theo enrollment và cụm family; temporal overlap; cutoff hồi cứu; đuôi video; catalog/profile không có lịch sử as-of; trạng thái problem tổng hợp; source text/cache sentiment; reference P95 V2 có trước split V4; chỉ một dataset và một split chính. Dev có 263 enrollment E trong 20 family, không phải 1.052 người khi gộp bốn phase. Bộ chỉ số E ở ngân sách 0,5% được báo mô tả theo QĐ34, không đặt thêm gate coreset sau khi thấy kết quả.

Nhãn và X được quản lý riêng. Việc fit X chỉ trên train không làm cho reference nhãn trở thành train-only. Các thống kê mô tả test ở card không được dùng để đổi split, feature, ngưỡng, seed hoặc siêu tham số.

Gói hiện có test vì người dùng đã yêu cầu bổ sung feature test: manifest ghi test_data_read=true và test_in_bundle=true; test_used_for_fit=false, test_used_for_qa23=false, model_training_run=false. Việc xây feature khác với đánh giá mô hình/test endpoint. Test model theo review vẫn thực hiện sau freeze được duyệt; trước freeze, nhóm thực nghiệm dùng gói chỉ train/dev/FULL_A. Card không tự cấp phép chạy test, không tự chốt M1 và không báo chưa từng đọc test cho toàn bộ quá trình xây dữ liệu.

Môi trường build trong manifest:

@@SOFTWARE@@

Môi trường panel được review 26–27/10 yêu cầu khóa riêng: Python 3.13.2, NumPy/pandas 2.2.3, scikit-learn 1.6.1, XGBoost 3.0.0; đây là cấu hình giao thức thực nghiệm, không phải môi trường build ở bảng trên. Rank/NMI khi có tie có thể lệch nhẹ giữa thư viện; giữ environment_id, feature order và các tham số đã xuất để tái lập. Source/version được đóng băng bằng hash; không lấy commit local hiện tại để thay hash script lịch sử đã chạy.

## 12 Những thích nghi công thức được công bố

Bảng kế thừa 18 mục giải trình O24 và cập nhật ba mục X/video/scaler theo V4.2. Các hàng TS/IF/FV/FA/COELO/AFELO/ACELO mô tả nhãn cố định; không tự áp dụng preprocessing feature X vào công thức nhãn.

@@ADAPTATIONS@@

Provenance nhãn còn giữ phát hiện lịch sử của script 15: hash ghi trong manifest V2 là `5ab5a8f62bc94070cf34949617b28a04e6e3a61158c1b518e3bb2be590fbdd48`, còn hash trong bảng giải trình là `0eaa181c674ec3c8214c6ad27c6df6aa1fe71d21fe0dd9d0772f7b636f8044c2`. Đối chiếu năm SQL được báo là khớp không chứng minh toàn script lịch sử giống file mới. V4.2 giữ nguyên nhãn đã đóng băng, lưu giới hạn provenance này theo QĐ27, không thay nhãn để khớp script mới.

## 13 File bàn giao cách đọc và tái tạo

@@FILES@@

Các file contract và audit đi cùng gồm preprocessing.json, feature_columns.json, learner/selector_feature_columns.json, audit/qa23_by_phase.json, audit/held_out_columns.json, audit/null_and_absence.json, audit/test/, audit/data_card_evidence.json, context_temporal_lineage.json, run_manifest.json, SHA256SUMS.txt và _SUCCESS.json. `_stage/` và `_scratch/` phục vụ tái tạo local, nằm ngoài bộ checksum bàn giao; không dùng toàn bộ cây staging như gói runtime thực nghiệm.

Ví dụ từ thư mục data_pipeline:

```python
from pathlib import Path
import importlib.util

root = Path("phase_features_v4_2").resolve()
spec = importlib.util.spec_from_file_location("cq_v4_2", root / "load_cq_v4_2.py")
loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loader)
train = loader.load(split="train", view="learner", root=root)
dev_p1 = loader.load(split="dev", phase="P1", view="learner", root=root)
full_a = loader.load(split="train", phase="FULL_A", root=root)
X_columns = loader.feature_columns("learner", root=root)
teacher_metadata = loader.teacher_subgroups(root=root)
dev_with_teacher = loader.load(split="dev", phase="P1", root=root,
                               include_teacher_subgroup=True)
# Sau freeze được duyệt:
# test_p1 = loader.load(split="test", phase="P1", view="learner", root=root)
```

Loader trả metadata, feature theo view và alias q/y; gọi raw=True để lấy raw đầy đủ. Chỉ X_columns là X, metadata/q/y đọc riêng. Loader hỗ trợ test khi manifest báo đã build và phải truyền phase; không có FULL_A dev/test.

```bash
python -m pip install -r requirements-v4_2.txt
python -m pytest -q -p no:cacheprovider test_cq_tempo_features.py tests/test_cq_v4_2_builder.py
python 25_build_phase_features_v4_2.py
python 27_build_test_features_v4_2.py
python 29_build_teacher_subgroup_v4_2.py
python 28_build_data_card_v4_2.py --d06-confirmed-on 2026-10-30
python 26_verify_phase_features_v4_2.py
```

Bước 28 cần artifact hiện hành và cache stage của bước 25 để đo Y37; tham số d06-confirmed-on ghi ngày nhận xác nhận thực tế, không tự thay kiểm thủ công. Cần dữ liệu nguồn private tương ứng manifest; không có dữ liệu giả thay thế. Các kiểm tra và data card không thay các lượt thực nghiệm đã đăng ký.

## 14 Hash bảo toàn và hồ sơ nguồn

@@HASHES@@

SHA256 của toàn bộ 20 raw/ready Parquet cùng sáu artifact policy được giữ trong evidence, đối chiếu trước/sau khi lập card. Card, evidence và lineage có version/hash riêng trong manifest; README nằm trong checksum gói. Thay đổi tài liệu làm đổi checksum gói nhưng không đổi hash X/y/cutoff/split/preprocessing. Khi đối chiếu panel, dùng hash đầu vào đúng phiên bản đã chạy; không suy ra hai archive bằng nhau chỉ vì cùng 76/74 cột hoặc cùng số dòng.

Nguồn: [helper CD3](cq_tempo_features.py), [builder](25_build_phase_features_v4_2.py), [test extension](27_build_test_features_v4_2.py), [builder data card](28_build_data_card_v4_2.py), [builder teacher_subgroup](29_build_teacher_subgroup_v4_2.py), [verifier](26_verify_phase_features_v4_2.py), [review 26/10](Review-2026-10-26.docx), [review 27/10](Review-2026-10-27.docx), [giải trình nhãn](BANG_GIAI_TRINH_CONG_THUC_FEATURE_VA_NHAN_CQ_V4_2026_10_23.md). Dữ liệu dòng, ID giả danh và nguồn audit private chỉ trao đổi trong nhóm có quyền truy cập. Gói này không tự cấp giấy phép công bố lại raw data hoặc đánh giá cá nhân người học/giảng viên.
'''


def build(root=ROOT, bundle=None, date=DATE, d06_confirmed_on=None):
    root=Path(root).resolve()
    bundle=Path(bundle or root/'phase_features_v4_2').resolve()
    validate_index(bundle)
    manifest,config,prep = [read_json(bundle/p) for p in ['run_manifest.json','feature_columns.json','preprocessing.json']]
    if not manifest.get('test_in_bundle', False):
        raise ValueError('This data card describes the complete bundle; run 27_build_test_features_v4_2.py first')
    for name in ['video.parquet', 'spine.parquet']:
        if not (bundle/'_stage'/name).is_file():
            raise FileNotFoundError(f'Y37 measurement requires the frozen build stage: {bundle/"_stage"/name}')
    protected={p:digest(bundle/p) for p in manifest['raw_paths']+manifest['ready_paths']+POLICY_FILES}
    print('Collecting data card evidence and temporal lineage',flush=True)
    evidence=collect(bundle,root,protected)
    prior=read_json(bundle/'audit/data_card_evidence.json') if (bundle/'audit/data_card_evidence.json').exists() else {}
    confirmation=d06_confirmed_on or prior.get('d06_confirmation',{}).get('recorded_on')
    evidence['d06_confirmation']={'status':'confirmed_by_user' if confirmation else 'not_asserted_by_card_builder',
                                  'recorded_on':confirmation,'basis':'User statement in task conversation' if confirmation else None,
                                  'manual_source_files_modified':False}
    evidence['documentation_version']=f'phase_features_v4_2-data-card-{date}-{CARD_REVISION}'
    evidence['source_hashes']={name:digest(root/name) for name in ['cq_tempo_features.py','25_build_phase_features_v4_2.py',
        '27_build_test_features_v4_2.py','28_build_data_card_v4_2.py','29_build_teacher_subgroup_v4_2.py','Review-2026-10-26.docx','Review-2026-10-27.docx',
        'BANG_GIAI_TRINH_CONG_THUC_FEATURE_VA_NHAN_CQ_V4_2026_10_23.md']}
    text,version=build_card_text(evidence,manifest,config,prep,date,confirmation,root)
    if any(digest(bundle/p)!=h for p,h in protected.items()):
        raise AssertionError('A model input changed during documentation collection')
    # Preserve prior completion metadata before the metadata-only update.
    history=bundle/'provenance/pre_data_card'
    history.mkdir(parents=True,exist_ok=True)
    for name in ['run_manifest.json','_SUCCESS.json','SHA256SUMS.txt','README.md']:
        if not (history/name).exists():
            shutil.copyfile(bundle/name,history/name)
    original=read_json(bundle/'_SUCCESS.json')
    (bundle/'_SUCCESS.json').unlink(missing_ok=True)
    (root/CARD_NAME).write_text(text,encoding='utf-8')
    # Adapt Markdown links for the self-contained copy inside the dataset.
    inside=text.replace('](phase_features_v4_2/','](')
    for name in ['cq_tempo_features.py','25_build_phase_features_v4_2.py','27_build_test_features_v4_2.py','28_build_data_card_v4_2.py','29_build_teacher_subgroup_v4_2.py',
                 '26_verify_phase_features_v4_2.py','Review-2026-10-26.docx','Review-2026-10-27.docx',
                 'BANG_GIAI_TRINH_CONG_THUC_FEATURE_VA_NHAN_CQ_V4_2026_10_23.md']:
        inside=inside.replace(']('+name+')','](../'+name+')')
    inside=inside.replace('](outputs/phase_features_v4_2_validation.json)','](../outputs/phase_features_v4_2_validation.json)')
    (bundle/CARD_NAME).write_text(inside,encoding='utf-8')
    write_json(bundle/'audit/data_card_evidence.json',evidence)
    write_json(bundle/'context_temporal_lineage.json',{'dataset_version':manifest['dataset_version'],
        'documentation_version':version,'columns':evidence['context_lineage']})
    readme=re.sub(r'\n## Data card\n.*?(?=\n## |\Z)', '', (bundle/'README.md').read_text(), flags=re.S)
    readme+='\n## Data card\n\n[Data card V4.2](DATA_CARD_V4_2.md) describes the current CD3 feature view, frozen labels, Y37, context temporal lineage, missingness and test scope.\n'
    (bundle/'README.md').write_text(readme,encoding='utf-8')
    provenance=bundle/'provenance/data_card'
    provenance.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(Path(__file__),provenance/Path(__file__).name)
    documentation={'version':version,'updated_on':date,'data_card_path':CARD_NAME,'data_card_sha256':digest(bundle/CARD_NAME),
        'evidence_path':'audit/data_card_evidence.json','evidence_sha256':digest(bundle/'audit/data_card_evidence.json'),
        'context_lineage_path':'context_temporal_lineage.json','context_lineage_sha256':digest(bundle/'context_temporal_lineage.json'),
        'model_inputs_changed':False,'protected_model_artifacts':len(protected),'d06_confirmation':evidence['d06_confirmation']}
    manifest['documentation']=documentation
    write_json(bundle/'run_manifest.json',manifest)
    if any(digest(bundle/p)!=h for p,h in protected.items()):
        raise AssertionError('A model input changed during documentation installation')
    paths=[p for p in sorted(bundle.rglob('*')) if p.is_file() and not any(
        s in p.relative_to(bundle).parts for s in ['_stage','_scratch','__pycache__']) and
        p not in [bundle/'SHA256SUMS.txt',bundle/'_SUCCESS.json'] and not p.name.endswith('.partial')]
    (bundle/'SHA256SUMS.txt').write_text(''.join(f'{digest(p)}  {p.relative_to(bundle).as_posix()}\n' for p in paths))
    original.update({'manifest_sha256':digest(bundle/'run_manifest.json'),'checksums_sha256':digest(bundle/'SHA256SUMS.txt'),
                     'documentation_version':version,'model_inputs_changed':False})
    if manifest.get('teacher_subgroup'):
        original['teacher_subgroup_version']=manifest['teacher_subgroup']['version']
    write_json(bundle/'_SUCCESS.json',original)
    print(f'SUCCESS: {root/CARD_NAME}; {len(protected)} model artifacts unchanged',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--bundle',type=Path)
    parser.add_argument('--date',default=DATE)
    parser.add_argument('--d06-confirmed-on',help='Date a real user confirmation was received; does not sign a manual review form')
    args=parser.parse_args()
    build(args.root,args.bundle,args.date,args.d06_confirmed_on)
