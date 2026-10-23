"""Independent source checks and Vietnamese CQ V4 data card/handoff status."""
from __future__ import annotations

import importlib.util
import json
import re
import shutil
import sys

from cq_v4_common import *


def table(headers, rows):
    return '| ' + ' | '.join(headers) + ' |\n|' + '|'.join('---' for _ in headers) + '|\n' + ''.join(
        '| ' + ' | '.join(str(v).replace('|',' / ').replace('\n',' ') for v in row) + ' |\n' for row in rows)


def independent_checks(con, out, quality):
    checks=Checks()
    view(con,'snapshot',out/'supervised_raw/*/*.parquet')
    view(con,'assignment',quality/'tables/user_course_splits.parquet')
    view(con,'sample',out/'audit/cutoff_sample_20_pairs.parquet')
    view(con,'video_source',ROOT/'final_quality_v2/_stage/video_window.parquet')
    view(con,'problem_source',ROOT/'eda_engagement_features_comparison/_stage/problem_rows/user_shard=*/*.parquet')
    view(con,'comment_source',ROOT/'final_quality_v2/_stage/comment_window.parquet')
    view(con,'reply_source',ROOT/'final_quality_v2/_stage/replies_prepared.parquet')
    con.execute('''CREATE TEMP TABLE source_sample_events AS
        SELECT s.row_id,'video' event_kind,v.event_time,
          CASE WHEN v.valid_measurement THEN v.event_time+
            (least(v.end_point-v.start_point,greatest(0.0,epoch(s.cutoff_time)-epoch(v.event_time))*v.speed)/v.speed)*INTERVAL '1 second'
            ELSE v.event_time END observed_end_time
        FROM sample s JOIN video_source v USING(user_id,course_id) WHERE v.event_time BETWEEN s.enroll_time AND s.cutoff_time
        UNION ALL SELECT s.row_id,'problem',p.event_time,p.event_time
        FROM sample s JOIN problem_source p USING(user_id,course_id) WHERE p.event_time BETWEEN s.enroll_time AND s.cutoff_time
        UNION ALL SELECT s.row_id,'comment',c.event_time,c.event_time
        FROM sample s JOIN comment_source c USING(user_id,course_id) WHERE c.event_time BETWEEN s.enroll_time AND s.cutoff_time
        UNION ALL SELECT s.row_id,'reply',r.event_time,r.event_time
        FROM sample s JOIN reply_source r USING(user_id,course_id) WHERE r.event_time BETWEEN s.enroll_time AND s.cutoff_time
        UNION ALL SELECT s.row_id,'received',r.event_time,r.event_time
        FROM sample s JOIN comment_source c USING(user_id,course_id) JOIN reply_source r ON r.parent_comment_id=c.comment_id
        WHERE c.normalized_text IS NOT NULL AND c.exact_duplicate_rank=1 AND c.event_time BETWEEN s.enroll_time AND s.cutoff_time
          AND r.event_time BETWEEN c.event_time AND s.cutoff_time''')
    write(con,out/'audit/cutoff_sample_20_source_comparison.parquet','''WITH expected AS (
        SELECT row_id,min(event_time) expected_min,max(observed_end_time) expected_max,
        count(*) FILTER(WHERE event_kind<>'received') expected_n_events FROM source_sample_events GROUP BY 1)
        SELECT s.*,e.expected_min,e.expected_max,coalesce(e.expected_n_events,0) expected_n_events,
        (s.n_events=coalesce(e.expected_n_events,0)) AND
        (s.min_feature_event_time IS NOT DISTINCT FROM e.expected_min) AND
        (s.max_feature_event_time IS NOT DISTINCT FROM e.expected_max) source_match
        FROM sample s LEFT JOIN expected e USING(row_id) ORDER BY enrollment_id,phase''')
    view(con,'comparison',out/'audit/cutoff_sample_20_source_comparison.parquet')
    checks.sql(con,'20_pairs_80_snapshots_match_raw_event_sources','SELECT count(*)=80 AND bool_and(source_match) FROM comparison')
    checks.sql(con,'sample_source_events_within_cutoff', '''SELECT count(*)=0 FROM source_sample_events e JOIN sample s USING(row_id)
        WHERE e.event_time<s.enroll_time OR e.observed_end_time>s.cutoff_time+INTERVAL '1 microsecond' ''')
    for phase in PHASES:
        view(con,'new_seq',out/'sequences'/phase/'*.parquet')
        view(con,'old_seq',ROOT/'phase_features_v3/sequences'/phase/'*.parquet')
        checks.sql(con,phase+'_weekly_sequence_values_exactly_preserved', '''SELECT count(*)=0 FROM new_seq n FULL JOIN old_seq o USING(user_id,course_id)
            WHERE n.user_id IS NULL OR o.user_id IS NULL OR n.behavior_sequence IS DISTINCT FROM o.behavior_sequence''')
    view(con,'old_fullA',ROOT/'fullA/supervised_raw/*.parquet')
    checks.sql(con,'fullA_source_targets_match_frozen_quality','''SELECT count(*)=0 FROM old_fullA f JOIN assignment a USING(user_id,course_id)
        WHERE f.q_alpha_1 IS DISTINCT FROM a.q_alpha_1 OR f.y_alpha_1 IS DISTINCT FROM a.y_alpha_1''')
    view(con,'raw_courses',ROOT/'input/moocubex_parquet/tables/entities/course/*.parquet')
    view(con,'dim_resources',ROOT/'input/moocubex_model/processed/dim_lo_resource/*.parquet')
    checks.sql(con,'family_resource_video_set_matches_raw_catalog', '''WITH ids AS(SELECT DISTINCT course_id FROM assignment),
        a AS(SELECT DISTINCT c.id course_id,x.resource_id video_id FROM raw_courses c CROSS JOIN unnest(resource) t(x)
            JOIN ids ON c.id=ids.course_id WHERE starts_with(x.resource_id,'V_')),
        b AS(SELECT DISTINCT course_id,resource_id video_id FROM dim_resources JOIN ids USING(course_id) WHERE starts_with(resource_id,'V_'))
        SELECT (SELECT count(*) FROM a ANTI JOIN b USING(course_id,video_id))=0
        AND (SELECT count(*) FROM b ANTI JOIN a USING(course_id,video_id))=0''')
    for name in ['selector','learner']:
        v=json.loads((out/f'{name}_feature_columns.json').read_text())
        checks.add(name+'_feature_order_hash',v['columns_sha256']==canonical_hash(v['columns']))
    # Check metadata and feature access through both private and runtime loaders.
    spec=importlib.util.spec_from_file_location('final_v4_loader',out/'load_cq_v4.py')
    loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
    for phase in PHASES:
        private=loader.load_phase(phase=phase,split='dev',root=out,connection=con)
        runtime=loader.load_phase(phase=phase,split='dev',root=out/'runtime_dev_only',connection=con)
        checks.add(phase+'_private_runtime_same_dev_count',private.count('*').fetchone()[0]==runtime.count('*').fetchone()[0])
    checks.save(out/'independent_validation.json')
    return len(checks.items)


def build(documents_only=False):
    quality,out=ROOT/'final_quality_v4',ROOT/'phase_features_v4'
    if not (out/'_SUCCESS.json').exists():raise RuntimeError('Phase build incomplete')
    con=connect(out)
    n_independent=(json.loads((out/'independent_validation.json').read_text())['checks_passed']
                   if documents_only else independent_checks(con,out,quality))
    q=json.loads((quality/'summary.json').read_text())
    p=json.loads((out/'summary.json').read_text())
    cs=json.loads((out/'course_statistics.json').read_text())
    constants=json.loads((out/'protocol_constants.json').read_text())
    old=json.loads((ROOT/'phase_features_v3/course_random_split/summary.json').read_text())
    per_split={x['split']:x for x in q['split_counts']}
    class_counts={s:{x['y']:x['n'] for x in q['class_counts'] if x['split']==s} for s in SPLITS}
    view(con,'family_splits',quality/'family_splits.parquet')
    view(con,'assignment',quality/'tables/user_course_splits.parquet')
    extra=records(con,'''SELECT split,count(*) n_families,count(*) FILTER(WHERE has_E) n_families_E
        FROM family_splits GROUP BY 1 ORDER BY 1''')
    test_e=class_counts['test']['E'];overlap=q['temporal_overlap']
    counts=(q['checks_passed'],p['checks_passed'],n_independent)
    pending=[
        {'item':'ID-only and missing-mask probes on V4','owner':'experiment','status':'NOT_RUN'},
        {'item':'Shuffled-label / majority model checks on V4','owner':'experiment','status':'NOT_RUN'},
        {'item':'Selector weight contract and baseline reproduction','owner':'experiment','status':'NOT_RUN'},
        {'item':'FullTrain-A/B training, headroom, QD16 and panel','owner':'experiment','status':'NOT_RUN'},
        {'item':'NOLFO comparison because temporal_overlap>5%','owner':'experiment, thesis','status':'REQUIRED_LATER' if overlap['requires_nolfo'] else 'NOT_REQUIRED'},
        {'item':'Human review of 20-pair source comparison and scientific label validation','owner':'research','status':'PENDING'},
    ]
    handoff={'dataset_version':VERSION,'data_build':'PASS','data_ready_for_experiment':True,
        'overall_handoff':'PARTIAL_EXPERIMENT_CHECKS_PENDING','ready_for_m0_v4':False,
        'data_checks_passed':sum(counts),'pending':pending,'tempo_dq_comparison':'NOT_REQUIRED_BY_USER',
        'no_failed_data_check':True,'test_excluded_from_runtime':True}
    dump(out/'handoff_status.json',handoff)
    dump(out/'runtime_dev_only/handoff_status.json',handoff)
    splitrows=[]
    for s in SPLITS:
        row=per_split[s];cl=class_counts[s]
        splitrows.append([s,row['n_families'],row['n_courses'],row['n_enrollments'],row['n_users'],cl['E'],cl['G'],cl['A'],row['n_enrollments']*4])
    old_counts={x['split']:x['n'] for x in old['labeled_splits']}
    changes=[
        ['Cohort và bộ lọc','Giữ nguyên','178.598 cặp có nhãn / 1.808 course; loại only-unknown chapter như V3. Không oversample hoặc xóa lớp E.','Không thiếu ở bước build'],
        ['q/y, thành phần và reference P95','Giữ nguyên','Mọi giá trị gốc được đối chiếu với V3; chỉ metadata split/family thay đổi. Không inference sentiment lại.','Độ giá trị giáo dục/sentiment chưa được thẩm định độc lập'],
        ['Target chính','Đã chốt','q_alpha_1 / y_alpha_1; alpha 0,1 chỉ sensitivity.','Không có lựa chọn alpha theo kết quả dev'],
        ['Family mapping','Mới','Toàn cohort, graph ccid Jaccard≥0,9, minimum 5; connected components; không lọc theo tên.','Family là proxy tương đồng nội dung, không phải định danh lần mở chính thức'],
        ['Độ nhạy family','Đã làm','Ngưỡng 0,8 lưu riêng; không thay assignment chính 0,9.','Không chọn ngưỡng theo điểm mô hình'],
        ['Split','Thay đổi','Test 20% họ mở muộn nhất; train/dev seed42 theo quartile kích thước × has_E; không giao family/course/cặp.','Temporal overlap còn cao; cần NOLFO sau'],
        ['Raw feature P1–P4','Giữ nguyên','57 numeric + 5 categorical; toàn bộ raw numeric/gender/cutoff khớp V3. Các code danh mục fit lại.','Không có yêu cầu đối chiếu TEMPO-DQ theo chỉ dẫn người dùng'],
        ['Hai feature view','Đã xuất','Selector: 56 numeric bỏ phase_fraction và mọi categorical ID. Learner: 57 numeric gồm phase_fraction + 5 categorical, tổng 62.','Categorical codes là category, không phải biến liên tục; learner cần xử lý phù hợp'],
        ['Preprocessing','Thay đổi','Một median/scaler chung trên '+str(p['fit_rows'])+' snapshot train P1–P4; encoder chỉ fit train V4. Không missing indicator.','Không dùng ready/scaler V3 hoặc fit scaler riêng phase'],
        ['Phase và row ID','Bổ sung','Metadata family/enrollment/row ID; phase_fraction vẫn có bốn giá trị khác nhau sau chuẩn hóa chung.','Các snapshot của cùng cặp có phụ thuộc'],
        ['Chuỗi tuần','Giữ giá trị','Giá trị behavior_sequence khớp V3; cập nhật split và thêm family/row ID.','Chuỗi tuần thưa, không tự chèn tuần trống'],
        ['FullTrain-A','Đã tạo theo contract V4','Tái sử dụng raw λ=1 từ fullA; lấy train V4, remap code và dùng preprocessor Pool B. Thêm phase_fraction=1.','Chưa huấn luyện FullTrain-A; A không thuộc Pool B'],
        ['Qc/nc, μy và oracle','Đã tính','Qc/nc trên labeled enrollment; μy từ train, một dòng/cặp; oracle floor dev và signed oracle errors được lưu.','Đây là oracle của phép tổng hợp, không phải kết quả dự đoán mô hình'],
        ['Thống kê card','Đã tính','ICC(1), nc, concentration, missingness, class/family E, user/teacher cũ mới, temporal overlap.','Số liệu không chứng minh khả năng tổng quát hóa hoặc công bằng'],
        ['Cutoff','Đã kiểm tra tự động','Công thức toàn supervised; 20 cặp/80 snapshot đối chiếu độc lập từ video/problem/comment/reply.','Chưa có người duyệt thủ công hồ sơ 20 cặp'],
        ['Runtime dev-only','Đã đóng gói','Chỉ train/dev và A train; loader từ chối test; preprocessing cố định. Test chỉ có trong root private.','Không mount toàn root phase_features_v4 vào notebook dev'],
        ['Manifest/hash/loader','Đã có','Mỗi gói có manifest, SHA256SUMS và loader; danh sách/feature order có hash.','Tái tạo cần giữ input V3/fullA và phiên bản môi trường'],
        ['Kiểm tra 1–12','Một phần theo phạm vi dữ liệu','Các check dữ liệu đã PASS; kiểm tra nguồn 20 cặp đã có hồ sơ.','Probe ID/mask, shuffled label, majority, trọng số và baseline cần pipeline thực nghiệm'],
        ['Đối chiếu TEMPO-DQ','Không yêu cầu','Dùng nguyên feature do người dùng cung cấp; không coi thiếu mapping TEMPO-DQ là blocker.','Không tuyên bố tái lập feature TEMPO-DQ'],
    ]
    missing=json.loads((out/'missingness.json').read_text())
    env=json.loads((out/'preprocessing.json').read_text())['software']
    lines=[
        '# Data card CQ V4', '',
        '**Ngày tạo:** '+DATE+'. **Trạng thái:** dữ liệu final/phase/FullTrain-A đã build và kiểm tra kỹ thuật; còn các kiểm tra mô hình trước khi nghiệm thu M0-V4.',
        '', '**Phạm vi người dùng chốt:** dùng bộ feature hiện có của V3; không cần đối chiếu TEMPO-DQ. Tài liệu này phân biệt dữ liệu đã xây với công việc thực nghiệm chưa chạy.',
        '', '## 1. Định danh và quần thể', '',
        'Dataset version: **CQ V4**. Final: [final_quality_v4](final_quality_v4/). Phase: [phase_features_v4](phase_features_v4/). Runtime an toàn cho dev: [runtime_dev_only](phase_features_v4/runtime_dev_only/).',
        '', 'Khóa gốc: `(user_id,course_id)`; snapshot `(user_id,course_id,phase)`. `enrollment_id` là SHA256 JSON của khóa cặp; `row_id` là SHA256 JSON enrollment_id + phase. Metadata ID không thuộc X.',
        '', 'Final giữ **10.455.017** cặp đủ thời gian sau lọc V3, gồm **178.598** cặp có nhãn và **10.276.419** cặp chưa có nhãn. Tỷ lệ có nhãn **1,7083%**. Cặp chưa có nhãn không thành lớp A, giữ split/enrollment_id/row_id NULL trong kho đầy đủ. Family chỉ được xây trên 1.808 course có nhãn; course ngoài cohort có family NULL.',
        '', 'Mỗi phase giữ 10.455.017 dòng raw đầy đủ và 178.598 dòng supervised. Tổng bốn phase **714.392 snapshot supervised**. Những dòng của cùng cặp không phải quan sát độc lập.',
        '', '## 2. Bảng những gì đã làm và thay đổi so với V3', '',
        table(['Hạng mục','Trạng thái/thay đổi','Đã làm','Giới hạn/phần còn thiếu'],changes),
        '', '## 3. Family và chia tập', '',
        f'Ngưỡng 0,9 tạo **{q["families"]} họ**, **{q["family_edges"]} cạnh**, **{q["sensitivity"]["0.9"]["multi_course_families"]} họ nhiều course**. Phân bố trạng thái: `'+json.dumps(q['mapping_status'],ensure_ascii=False)+'`.',
        '', 'Tập ccid lấy từ resource video của catalog, nối tất cả quan hệ ccid khác rỗng và khử trùng. Không tự chọn một ccid tối thiểu cho video đa ánh xạ. Video-set đã được đối chiếu với catalog raw độc lập.',
        '', 'Test: ceil(0,2 × số họ), sắp `(family_start, family_id)`; ngày mở họ là min(course_start). Phần còn lại: qcut của rank kích thước, 4 nhóm × has_E; dev lấy round(0,125 × số họ trong tầng), thứ tự ngẫu nhiên seed42. Python ties-to-even được giữ; không di chuyển họ theo kết quả mô hình. Tỷ lệ áp dụng cho số họ, không cho số enrollment.',
        '',table(['Split','Family','Course','Cặp có nhãn','User','E','G','A','Snapshot P1–P4'],splitrows),
        '', table(['Split','Family có E'],[[r['split'],r['n_families_E']] for r in extra]),
        '', 'So với bản course-random V3: '+ '; '.join(f'{s}: {old_counts[s]:,} → {per_split[s]["n_enrollments"]:,} cặp' for s in SPLITS)+'.',
        '', f'Ngưỡng 0,8 tạo {q["sensitivity"]["0.8"]["families"]} họ; {q["sensitivity"]["courses_changed_component_membership"]} course thay đổi tập thành viên họ. Báo cáo: [family sensitivity](final_quality_v4/COURSE_FAMILY_SENSITIVITY.md).',
        '', f'**Temporal overlap:** {overlap["numerator"]}/{overlap["denominator"]} họ train = **{overlap["fraction"]:.4%}**, so với family test mở sớm nhất tại {overlap["first_test_family_start"]} theo timestamp pipeline. `requires_nolfo={overlap["requires_nolfo"]}`. Đã lưu danh sách overlap; chưa dựng/chạy đối chứng NOLFO. Split chính được giữ.',
        '', '## 4. Target và thời gian', '',
        'Target chính q_alpha_1/y_alpha_1, giữ nguyên từ V3. q = mean LOQV các chương quan sát; LOQV_alpha = sqrt(sum_available(w × (1-index)^2)/sum_available(w)), w=(1,alpha,1). Mean bỏ NULL và giữ 0. E:q<0,2; G:0,2≤q<0,4; A:q≥0,4. Alpha 1 khớp khoảng cách Euclid chuẩn hóa ba chiều khi đủ ba chỉ số; khi thiếu chiều, chỉ dùng chiều khả dụng như V3.',
        '', 'COELO=mean khả dụng TS/RV/AV/IF, tối thiểu hai thành phần; AFELO=mean khả dụng FV/FA; ACELO dùng số problem đúng có điều chỉnh attempts trên mẫu số catalog. P95, signed FA, model sentiment và chapter -1 giữ phiên bản V3. Xem [formula policy](final_quality_v4/formula_policy.json) và [card V3](DATA_CARD_V3.md) cho công thức thành phần, revision sentiment, missing/0 và điều kiện tracking.',
        '', 'Enrollment về 00:00:00; course_end là cuối ngày hoạt động cuối quan sát, không phải lịch đóng chính thức. Cutoff P1–P4 = enrollment + λ × (course_end−enrollment), λ=0,25/0,50/0,75/0,90. FullTrain-A có λ=1. Timestamp không tự cộng offset; session SQL UTC, ngày tài liệu Asia/Ho_Chi_Minh.',
        '', '## 5. Feature và preprocessing', '',
        f'Learner **{p["learner_features"]} cột** = 57 numeric + 5 categorical; selector **{p["selector_features"]} numeric**. Giữ feature của người dùng; phase_fraction là feature giao thức của learner, được giữ trong metadata phase để phân tầng selector. Các mã learner/teacher_set/school_set/field_set fit từ train mới; gender giữ mã nguồn. Unknown/missing category=0, riêng gender không hợp lệ=-1.',
        '', f'Fit median/mean/std trên **{p["fit_rows"]:,} snapshot train** gộp bốn phase. Raw giữ NULL; ready=(coalesce(raw,median_train)−mean_train)/std_train. Cột hằng dùng scale=1; không thêm missing indicators. Dev, test lưu riêng và FullTrain-A chỉ transform, không fit. FullTrain-A raw giữ nguyên 56 numeric đã có, thêm window fraction=1 để đồng nhất schema learner.',
        '', 'Allowlist và định nghĩa: [selector](phase_features_v4/selector_feature_columns.json), [learner](phase_features_v4/learner_feature_columns.json), [dictionary](phase_features_v4/feature_dictionary.json), [preprocessing](phase_features_v4/preprocessing.json). Danh sách numeric/categorical không chứa q/y, chỉ số cuối cửa sổ, course_end, cutoff, split, family hay ID metadata.',
        '', 'Tỷ lệ thiếu X theo cột/phase/split có trong [missingness.json](phase_features_v4/missingness.json), trường `n` là mẫu số. Kho sequences giữ giá trị tuần V3, thêm metadata V4.',
        '', '## 6. Thống kê cấp khóa học', '',
        'Qc=mean q_alpha_1 trên cặp có nhãn; nc là số cặp đó. Đây là quần thể được QĐ21 chấp nhận, không đại diện toàn bộ người đăng ký. μy là hằng số train; nmin=30, delta_C=0,01, delta=0,005; class weights tự nhiên.',
        '', table(['Hằng số','Giá trị'],[[k, f'{constants["mu"][k]:.12f}'] for k in ['E','G','A']]),
        '', table(['Split','Course','nc min','Q1','Median','Q3','Max','Course nc≥30','Family có course nc≥30'],[[r['split'],r['n_courses'],r['min_n'],r['p25_n'],r['median_n'],r['p75_n'],r['max_n'],r['n_courses_ec'],r['n_families_with_ec_course']] for r in cs['course_sizes']]),
        '', table(['Split','ICC(1)'],[[r['split'],f'{r["icc1"]:.8f}'] for r in cs['icc1']]),
        '', table(['Split','Course dùng EC','Oracle floor','Signed oracle error'],[[r['split'],r['n_courses'],f'{r["floor"]:.8f}',f'{r["signed"]:.8f}'] for r in cs['oracle_floor']]),
        '', table(['Split','Course','Tỷ trọng lớn nhất','Số course hiệu dụng'],[[r['split'],r['n_courses'],f'{r["largest_share"]:.4%}',f'{r["effective_courses"]:.4f}'] for r in cs['concentration_course']]),
        '', 'Các chỉ số oracle dùng đúng prior lớp của course nhân μy train; chúng không phải kết quả FullTrain-B. SD của πc và concentration theo family cũng lưu trong [course_statistics.json](phase_features_v4/course_statistics.json).',
        '', 'User cũ/mới tính trên lượt ghi danh test, so với user đã có trong train: `'+json.dumps(cs['test_user_old_new'])+'`.',
        '', 'Giảng viên cũ/mới: phân loại mỗi lượt test thành tất cả giảng viên cũ, tất cả mới, mixed, hoặc thiếu metadata; không suy ra từ teacher_set_code=0. Số liệu: `'+json.dumps(cs['test_teacher_old_new'])+'`.',
        '', '## 7. Kiểm tra và việc còn thiếu', '',
        f'Quality build **{counts[0]}/{counts[0]}**, phase/FullTrain-A/loader **{counts[1]}/{counts[1]}**, kiểm tra độc lập bổ sung **{counts[2]}/{counts[2]}**; tổng **{sum(counts)}** check dữ liệu PASS. Có 7 unit test giao thức family/split đã chạy đạt.',
        '', 'Đối chiếu độc lập 20 cặp × 4 phase đã tái đọc video/problem/comment/reply và reply nhận, kiểm tra số event cùng min/max timestamp khớp raw snapshot/audit. Hồ sơ dòng lưu riêng trong `phase_features_v4/audit/`; card chỉ ghi số tổng hợp. Việc người đánh giá duyệt thủ công hồ sơ này còn chờ.',
        '',table(['Việc chưa thực hiện','Đơn vị thực hiện','Trạng thái'],[[r['item'],r['owner'],r['status']] for r in pending]),
        '', '**Không có phần đối chiếu TEMPO-DQ còn thiếu:** hạng mục này được bỏ theo yêu cầu người dùng. Không suy ra tính hợp lệ nhãn, fairness hoặc chất lượng mô hình từ các check kỹ thuật. Nhãn vẫn có thiên lệch chọn cohort, dữ liệu thiếu modality, P95 hiệu chuẩn trên quần thể V2 và mốc thời gian hồi cứu.',
        '', 'Trạng thái bàn giao: **DATA_BUILD_PASS / dữ liệu sẵn cho nhóm thực nghiệm**, nhưng **chưa READY_FOR_M0_V4** vì còn các probe/contract mô hình và review độc lập. [handoff_status.json](phase_features_v4/handoff_status.json) ghi từng việc chờ.',
        '', '## 8. Cách đọc và tái tạo', '',
        'Notebook dev chỉ dùng `phase_features_v4/runtime_dev_only/`. Root private còn test data và toàn kho raw nên không dùng nguyên root đó làm bundle M0.',
        '', '```python\nimport importlib.util\nfrom pathlib import Path\nroot = Path("phase_features_v4/runtime_dev_only").resolve()\nspec = importlib.util.spec_from_file_location("cq_v4_loader", root / "load_cq_v4.py")\nloader = importlib.util.module_from_spec(spec)\nspec.loader.exec_module(loader)\ntrain = loader.load_phase(phase=None, split="train", root=root)\ndev_p1 = loader.load_phase(phase="P1", split="dev", root=root)\nfull_a = loader.load_fullA(root=root)\nx_cols = loader.feature_columns(view="learner", root=root)\n# train.project(",".join(x_cols)) là X; q/y và metadata đọc riêng.\n```',
        '', 'Từ `working_13-10/`, dùng Python '+sys.version.split()[0]+' có DuckDB/NumPy/Pandas. Môi trường build: `'+json.dumps(env)+'`. Phiên bản được khóa trong `requirements-v4.txt`.',
        '', '```bash\npython 21_build_final_quality_v4.py --threads 2 --memory 1200MB\npython 22_build_phase_features_v4.py --threads 2 --memory 1200MB\npython 23_finalize_data_card_v4.py\n```',
        '', 'Builder ghi riêng V4, không sửa input V3/fullA. Bước 21 khóa assignment trước bước 22 fit X. Tên phiên bản/ngày tạo cố định trong splits; thời gian chạy nằm ở manifest riêng. Mỗi root và runtime có SHA256SUMS. Tái tạo cần dữ liệu private đã có; code/manifest không chứa dữ liệu thay thế.',
        '', 'Dữ liệu dòng và ID giả danh giữ ngoài Git; `.gitignore` chặn các output V4. Điều khoản phát hành/đồng thuận/kiểm tra công bằng độc lập chưa được xác nhận; gói này không tự cấp giấy phép công bố dữ liệu.',
    ]
    card='\n'.join(lines)+'\n'
    (ROOT/'DATA_CARD_V4.md').write_text(card,encoding='utf-8')
    runtime=out/'runtime_dev_only'
    shutil.copyfile(ROOT/'requirements-v4.txt',runtime/'requirements-v4.txt')
    for name in ['missingness.json','course_statistics.json','COURSE_FAMILY_SENSITIVITY.md']:
        shutil.copyfile(out/name,runtime/name)
    shutil.copyfile(quality/'formula_policy.json',runtime/'formula_policy.json')

    def runtime_link(match):
        label,target=match.groups()
        if target=='phase_features_v4/runtime_dev_only/':
            return f'[{label}](.)'
        if target.startswith('phase_features_v4/'):
            local=target[len('phase_features_v4/'):]
            return f'[{label}]({local})' if (runtime/local).exists() else f'`{label}` (dataset private)'
        if target.startswith('final_quality_v4/'):
            local=target[len('final_quality_v4/'):]
            return f'[{label}]({local})' if (runtime/local).exists() else f'`{label}` (dataset private)'
        if target=='DATA_CARD_V3.md':
            return '`DATA_CARD_V3.md` (tài liệu nguồn, giữ riêng)'
        return match.group(0)

    runtime_card=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',runtime_link,card)
    runtime_card=runtime_card.replace('root = Path("phase_features_v4/runtime_dev_only").resolve()', 'root = Path(".").resolve()')
    (runtime/'DATA_CARD_V4.md').write_text(runtime_card,encoding='utf-8')
    dump(out/'finalization_manifest.json',{'dataset_version':VERSION,'date':DATE,
        'builder_sha256':sha256(__file__),'unit_test_source_sha256':sha256(ROOT/'tests/test_cq_v4_protocol.py'),
        'independent_checks':n_independent,'python':sys.version.split()[0],
        'root_card_sha256':sha256(ROOT/'DATA_CARD_V4.md')})
    dump(runtime/'run_manifest.json',{'dataset_version':VERSION,'date':DATE,'scope':'dev-only runtime',
        'source_phase_manifest_sha256':sha256(out/'run_manifest.json'),
        'finalization_manifest_sha256':sha256(out/'finalization_manifest.json'),
        'bundle_manifest_sha256':sha256(runtime/'bundle_manifest.json')})
    README='''# CQ V4 bàn giao dữ liệu

Trạng thái: DATA_BUILD_PASS; còn probe và kiểm tra mô hình theo handoff_status.json.
Đọc DATA_CARD_V4.md ở thư mục working_13-10 để xem đầy đủ thống kê/thay đổi.
Card trong runtime đã đổi đường dẫn sang các schema/preprocessor/loader đi kèm;
các nguồn private không phân phối được ghi bằng tên thay vì link hỏng.

Runtime này chỉ chứa train/dev Pool B và FullTrain-A train. Test feature/label
không có trong bundle. splits.json chỉ mô tả assignment theo family/course.
Loader trả relation có q,y và metadata; chỉ lấy X theo feature_columns(view).
Không fit lại preprocessor hoặc tự lấy mọi cột số làm X.

FullTrain-A có phase=FULL_A, phase_fraction=1 trước shared scaling.
Nó dùng đúng cohort train và không được ghép vào Pool B P1-P4.
SHA256SUMS.txt kiểm tra mọi file của gói, trừ chính checksum list.
'''
    (runtime/'README_HANDOFF.md').write_text(README)
    (out/'README_HANDOFF.md').write_text(README+'\nRoot private này có cả test; chỉ gửi runtime_dev_only/ cho notebook dev.\n')
    (quality/'README.md').write_text('# Final quality CQ V4\n\nTất cả q/y/chỉ số/cutoff giữ nguyên V3. Family split được khóa trong splits.json; family/time metadata thêm vào bảng final và các support table.\n\nXem ../DATA_CARD_V4.md để biết thay đổi, giới hạn và thống kê. Đây là dữ liệu private, không commit vào Git.\n')
    con.close()
    checksums(runtime);checksums(quality);checksums(out)
    print(json.dumps({'dataset_version':VERSION,'quality_checks':counts[0],'phase_checks':counts[1],
        'independent_checks':counts[2],'total_data_checks':sum(counts),'card':'DATA_CARD_V4.md',
        'status':'DATA_BUILD_PASS; model checks pending'},ensure_ascii=False))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--documents-only',action='store_true',help='Refresh documentation using the already completed independent checks')
    build(parser.parse_args().documents_only)
