"""Generate a Vietnamese data card and machine-readable evidence, read-only on data.

Counts and schemas are read from the persisted Parquet outputs. The card
describes the current alpha=0.1 targets and all earlier retained variants.
No features, labels, data splits or model parameters are changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import duckdb

AS_OF = '2026-10-17'
DATASET_ID = 'mooccubex_cq_review_phases_alpha01_v1'


def load(path):
    return json.loads(Path(path).read_text())


def write_json(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def qs(value):
    return "'"+str(value).replace("'", "''")+"'"


def rows(con,sql):
    cursor=con.execute(sql);names=[r[0] for r in cursor.description]
    return [dict(zip(names,row)) for row in cursor.fetchall()]


def fmt(n):
    return f'{int(n):,}'.replace(',','.')


def percent(n,d,places=2):
    return (f'{100*n/d:.{places}f}'.replace('.',',')+'%') if d else 'không xác định'


def num(n,digits=6):
    return f'{n:.{digits}f}'.replace('.',',')


def md_table(headers,data):
    def cell(value):return str(value).replace('|','\\|').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(map(cell,headers))+' |',
        '| '+' | '.join('---' for _ in headers)+' |']+
        ['| '+' | '.join(map(cell,row))+' |' for row in data])


def collect(base):
    con=duckdb.connect();con.execute("SET memory_limit='2GB'");con.execute('SET threads=2')
    summaries={name:load(base/name/'summary.json') for name in [
        'review_time_phases','final_quality_tables','afelo_sentiment_cardiff',
        'weighted_loqv_045_010_045','weighted_loqv_alpha_0p1','phase_features',
        'eda_review_course_enrollment','eda_problem_types_and_scores']}
    source_docs=[base/'Review-2026-10-12.docx',base/'moocubex_schema_va_danh_gia_3_chi_so.md']
    evidence_paths=source_docs+[p for folder in summaries for p in [base/folder/'summary.json']]
    for folder in ['review_time_phases','final_quality_tables','afelo_sentiment_cardiff',
        'weighted_loqv_045_010_045','weighted_loqv_alpha_0p1','phase_features']:
        for name in ['run_manifest.json','validation.json','_SUCCESS.json']:
            p=base/folder/name
            if p.exists():evidence_paths.append(p)
    evidence_paths += [base/'phase_features'/name for name in ['feature_columns.json','feature_dictionary.json',
        'preprocessing.json','temporal_invariance_validation.json']]
    evidence_paths += [base/'afelo_sentiment_cardiff'/name for name in ['validation_postrun.json','label_diagnostics.json','model_lock.json']]
    evidence_paths += [base/'final_quality_tables'/name for name in ['protocol_constants.json','review_loqv_formula_audit.json']]
    evidence_paths += list(sorted(base.glob('*.py')))
    evidence_paths += [base/'input/moocubex_parquet/catalog.json',base/'input/moocubex_model/catalog.json']
    evidence_paths=list(dict.fromkeys(evidence_paths))
    source_hashes={str(p.relative_to(base)):sha256(p) for p in evidence_paths}
    inventory=[]

    def add(name,pattern,layer,grain,role):
        import glob
        files=[Path(x) for x in sorted(glob.glob(str(base/pattern),recursive=True))]
        if not files:return
        schema=rows(con,f'DESCRIBE SELECT * FROM read_parquet({qs(base/pattern)},union_by_name=true)')
        count=con.execute(f'SELECT count(*) FROM read_parquet({qs(base/pattern)})').fetchone()[0]
        inventory.append({'name':name,'layer':layer,'parquet_glob':pattern,'grain':grain,'role':role,
            'n_rows':count,'n_columns':len(schema),'n_files':len(files),
            'bytes':sum(p.stat().st_size for p in files),
            'columns':[{'name':r['column_name'],'dtype':r['column_type']} for r in schema]})

    raw_catalog=load(base/'input/moocubex_parquet/catalog.json')
    model_catalog=load(base/'input/moocubex_model/catalog.json')
    for key,value in raw_catalog['tables'].items():
        suffix=value['parquet_glob'].split('/tables/',1)[1]
        add('raw/'+key,'input/moocubex_parquet/tables/'+suffix,'raw','theo schema nguồn','Nguồn Parquet notebook 01; không phải toàn bộ bảng đều dùng làm X.')
    for key,value in model_catalog['tables'].items():
        suffix=value['parquet_glob'].split('/moocubex_model/',1)[1]
        add('model/'+key,'input/moocubex_model/'+suffix,'dimension_bridge','theo bảng dimension/bridge','Chuẩn hóa khóa, tài nguyên và các quan hệ; notebook 02.')
    for folder in ['eda_review_course_enrollment','eda_engagement_features_comparison','eda_problem_types_and_scores']:
        for p in sorted((base/folder/'tables').glob('*.parquet')):
            add(folder+'/'+p.stem,str(p.relative_to(base)),'eda','theo bảng EDA','Phân tích/kiểm toán; không dùng trực tiếp làm X.')
    for p in sorted((base/'review_time_phases/tables').glob('*.parquet')):
        add('time/'+p.stem,str(p.relative_to(base)),'time_protocol','theo bảng thời gian','Metadata giao thức; không phải feature matrix.')
    for phase in ['P1','P2','P3','P4']:
        add('cutoff/'+phase,f'review_time_phases/phase_cutoffs/{phase}.parquet','time_protocol','user–course–phase','Mốc cắt, chưa có X/y.')
    for p in sorted((base/'final_quality_tables/tables').glob('*.parquet')):
        add('quality/'+p.stem,str(p.relative_to(base)),'final_quality',
            'user–course–chapter' if p.stem.endswith('components') or p.stem=='chapter_quality' else 'theo schema',
            'Chỉ số/nguồn nhãn cuối kỳ; không dùng làm X phase.')
    add('quality/user_course_final','final_quality_tables/tables/user_course_final/*.parquet','final_quality','user–course','Master chuẩn LOQV Review; chapter_details giữ feature con từng chương.')
    for name,pattern,grain in [
        ('comment_sentiment','afelo_sentiment_cardiff/tables/comment_sentiment.parquet','comment_id'),
        ('manual_audit_sample','afelo_sentiment_cardiff/tables/manual_audit_sample.parquet','comment_id'),
        ('unique_text_predictions','afelo_sentiment_cardiff/text_predictions/*.parquet','text_hash')]:
        add('sentiment/'+name,pattern,'sentiment',grain,'Suy luận sentiment và mẫu để kiểm tra; chưa có nhãn người gán.')
    for folder in ['weighted_loqv_045_010_045','weighted_loqv_alpha_0p1']:
        for name,grain in [('chapter_loqv_weighted','user–course–chapter'),('user_course_loqv_weighted','user–course')]:
            add(folder+'/'+name,f'{folder}/tables/{name}.parquet','weighted_target',grain,'Biến thể trọng số; alpha=0,1 là nhãn hiện dùng.')
    for name,grain in [('user_course_targets','user–course'),('course_static_metadata','course')]:
        add('phase/'+name,f'phase_features/tables/{name}.parquet','phase_metadata',grain,'Metadata/target riêng, không tự động đưa vào X.')
    for phase in ['P1','P2','P3','P4']:
        for folder in ['features','audit','sequences']:
            add(f'phase/{folder}/{phase}',f'phase_features/{folder}/{phase}/*.parquet','phase_output','user–course–phase',
                {'features':'Feature thô, mọi cặp đủ điều kiện thời gian; không có y.',
                 'audit':'Cặp có sự kiện; timestamp/flag kiểm toán, không là X.',
                 'sequences':'Chuỗi tuần thưa cho cặp có sự kiện; không có y.'}[folder])
        for folder in ['supervised_raw','supervised_sequences','model_ready']:
            add(f'phase/{folder}/{phase}',f'phase_features/{folder}/{phase}.parquet','supervised','user–course–phase',
                {'supervised_raw':'Feature thô ghép q/y cho cặp có nhãn.',
                 'supervised_sequences':'Giữ đủ cohort có nhãn, kể cả chuỗi rỗng.',
                 'model_ready':'Số đã impute/scale train-only; mã danh mục vẫn cần xử lý như category.'}[folder])
    for p in sorted((base/'phase_features/encoders').glob('*.parquet')):
        add('encoder/'+p.stem,str(p.relative_to(base)),'encoder','token','Vocabulary fit trên train có nhãn; unknown=0.')
    print('Inventoried',len(inventory),'logical tables',flush=True)
    target=base/'weighted_loqv_alpha_0p1/tables/user_course_loqv_weighted.parquet'
    con.execute(f'CREATE VIEW target AS SELECT * FROM read_parquet({qs(target)})')
    con.execute(f'CREATE VIEW ready_p1 AS SELECT * FROM read_parquet({qs(base/"phase_features/model_ready/P1.parquet")})')
    con.execute(f'CREATE VIEW supervised AS SELECT * FROM read_parquet({qs(base/"phase_features/supervised_raw/*.parquet")})')
    con.execute(f'CREATE VIEW course_splits AS SELECT * FROM read_parquet({qs(base/"final_quality_tables/tables/course_splits.parquet")})')
    stats={'summaries':summaries,
        'full_targets':rows(con,'''SELECT count(*) n_pairs,count(DISTINCT user_id) n_users,count(DISTINCT course_id) n_courses,
            count(*) FILTER(WHERE y_weighted IS NOT NULL) n_labeled FROM target''')[0],
        'labeled_targets':rows(con,'''SELECT count(*) n_pairs,count(DISTINCT user_id) n_users,count(DISTINCT course_id) n_courses
            FROM target WHERE y_weighted IS NOT NULL''')[0],
        'split_population':rows(con,'''SELECT split,count(*) n_pairs,count(DISTINCT user_id) n_users,count(DISTINCT course_id) n_courses,
            count(*) FILTER(WHERE y_weighted IS NOT NULL) n_labeled_pairs,
            count(DISTINCT user_id) FILTER(WHERE y_weighted IS NOT NULL) n_labeled_users,
            count(DISTINCT course_id) FILTER(WHERE y_weighted IS NOT NULL) n_labeled_courses
            FROM target GROUP BY 1 ORDER BY CASE split WHEN 'train' THEN 0 WHEN 'dev' THEN 1 ELSE 2 END'''),
        'q_train_class_means':rows(con,"SELECT y_weighted y,avg(q_weighted) mean_q,count(*) n_pairs FROM target WHERE split='train' AND y_weighted IS NOT NULL GROUP BY 1 ORDER BY 1"),
        'labeled_per_course':rows(con,'''WITH a AS (SELECT course_id,count(*) n FROM target WHERE y_weighted IS NOT NULL GROUP BY 1)
            SELECT count(*) n_courses,min(n) min_n,quantile_cont(n,[0.25,0.5,0.75]) quartiles,max(n) max_n,
                count(*) FILTER(WHERE n<30) n_courses_under_30 FROM a''')[0],
        'within_course_label_variation':rows(con,'''WITH a AS (SELECT course_id,count(DISTINCT y_weighted) n_classes FROM target
            WHERE y_weighted IS NOT NULL GROUP BY 1) SELECT n_classes,count(*) n_courses FROM a GROUP BY 1 ORDER BY 1'''),
        'categorical_unknowns':rows(con,'''SELECT split,count(*) n_pairs,
            count(*) FILTER(WHERE learner_code=0) learner_unknown,
            count(*) FILTER(WHERE teacher_set_code=0) teacher_set_unknown,
            count(*) FILTER(WHERE school_set_code=0) school_set_unknown,
            count(*) FILTER(WHERE field_set_code=0) field_set_unknown,
            count(*) FILTER(WHERE gender_code=-1) gender_invalid_or_missing FROM ready_p1 GROUP BY 1
            ORDER BY CASE split WHEN 'train' THEN 0 WHEN 'dev' THEN 1 ELSE 2 END'''),
        'raw_missing_by_phase':rows(con,'''SELECT phase,count(*) n_rows,
            count(*) FILTER(WHERE learner_age_at_enrollment IS NULL) missing_age,
            count(*) FILTER(WHERE video_mean_speed IS NULL) missing_video_mean,
            count(*) FILTER(WHERE problem_correct_rate IS NULL) missing_problem_rate,
            count(*) FILTER(WHERE problem_attempted_score_ratio IS NULL) missing_attempted_score,
            count(*) FILTER(WHERE sentiment_p_positive_mean IS NULL) missing_sentiment,
            count(*) FILTER(WHERE n_events=0) no_observed_activity
            FROM supervised GROUP BY 1 ORDER BY 1'''),
        'source_time_ranges':summaries['eda_review_course_enrollment']['source_time_audit'],
        'source_mapping_coverage':summaries['eda_review_course_enrollment']['source_mapping_coverage'],
        'problem_type_profile':summaries['eda_problem_types_and_scores']['catalog_by_type'],
        'problem_quality':summaries['eda_problem_types_and_scores']['scope_totals'],
        'model':load(base/'afelo_sentiment_cardiff/run_manifest.json'),
        'afelo_diagnostics':load(base/'afelo_sentiment_cardiff/label_diagnostics.json'),
        'phase_feature_contract':load(base/'phase_features/feature_columns.json'),
        'preprocessing':load(base/'phase_features/preprocessing.json'),
        'feature_dictionary':load(base/'phase_features/feature_dictionary.json'),
        'quality_constants':load(base/'final_quality_tables/protocol_constants.json')}
    validations={}
    for path in sorted(base.glob('*/validation.json')):validations[str(path.relative_to(base))]=load(path)
    for rel in ['phase_features/temporal_invariance_validation.json','final_quality_tables/review_loqv_formula_audit.json',
                'afelo_sentiment_cardiff/validation_postrun.json']:
        validations[rel]=load(base/rel)
    con.close()
    return stats,inventory,source_hashes,validations


def render_card(s,inventory,as_of):
    S=s['summaries'];time=S['review_time_phases'];quality=S['final_quality_tables']
    alpha=S['weighted_loqv_alpha_0p1'];old=S['weighted_loqv_045_010_045'];phase=S['phase_features']
    sent=S['afelo_sentiment_cardiff'];enroll=S['eda_review_course_enrollment']
    ft=s['full_targets'];lt=s['labeled_targets'];cfg=s['phase_feature_contract'];pre=s['preprocessing']
    n_snap=sum(r['n_rows'] for r in phase['phase_feature_rows'])
    n_super=sum(r['n_rows'] for r in phase['supervised_phase_rows'])
    inv={r['name']:r for r in inventory}
    raw_schema={r['name']:r['dtype'] for r in inv['phase/features/P1']['columns']}
    ready_schema={r['name']:r['dtype'] for r in inv['phase/model_ready/P1']['columns']}
    checks=phase['validation'];overlap=quality['temporal_overlap'];afdiag=s['afelo_diagnostics']['chapter_afelo'][0]
    pieces=[f'''# Data card MOOCCubeX CQ Review Phases

Bộ dữ liệu dẫn xuất phục vụ dự đoán sớm chất lượng vận hành ở cấp người học–khóa học qua bốn phase. Bản đang dùng ghép feature từ log trước cutoff với nhãn cuối kỳ có hệ số AFELO **alpha = 0,1**. Card này mô tả cả nguồn, EDA, bảng chỉ số và các phiên bản nhãn đã tạo; không thay đổi dữ liệu nguồn.

**Trạng thái:** đã tạo và kiểm tra tính nhất quán kỹ thuật; chưa có đánh giá mô hình dự đoán, thẩm định nhãn bởi chuyên gia hoặc xác nhận tái lập đầy đủ TEMPO-DQ. Cohort có nhãn chiếm {percent(lt['n_pairs'],ft['n_pairs'])} cặp đủ điều kiện thời gian và hiện không có lớp E. Split hiện tại còn chồng thời gian.

## 1. Tổng quan về tập dữ liệu

''']
    pieces.append(md_table(['Thuộc tính','Giá trị'],[
        ['Tên','MOOCCubeX CQ Review Phases'],['Dataset ID',f'`{DATASET_ID}`'],
        ['Phiên bản data card','1.0'],['Ngày chốt mô tả',as_of+'; ngày trình bày theo Asia/Ho_Chi_Minh'],
        ['Lĩnh vực','Learning analytics, educational data mining, dự đoán sớm, coreset cho dữ liệu bảng/chuỗi'],
        ['Đơn vị nhãn','user_id × course_id'],['Đơn vị feature/snapshot','user_id × course_id × phase'],
        ['Đơn vị chỉ số trung gian','user_id × course_id × big chapter (lo_key)'],
        ['Phiên bản feature',f'`{cfg["version"]}`'],
        ['Phiên bản nhãn hiện tại',f'`{alpha["weighted_formula_version"]}`'],
        ['Số feature X',f'{len(cfg["numeric_features"])} số + {len(cfg["categorical_features"])} danh mục; chuỗi tuần riêng'],
        ['Dữ liệu gốc','MOOCCubeX; Knowledge Engineering Group, Tsinghua University, với hỗ trợ XuetangX'],
        ['Bộ dẫn xuất','Nhóm đề tài trong Review CN-23521830-VyNH-QuangTVX; người dùng đã chạy notebook 01–02, pipeline hiện tại gồm các bước 03–09'],
        ['Đầu mối duy trì/contact','Chưa được xác nhận trong metadata hiện có'],
        ['Ngôn ngữ','Metadata/nội dung nguồn chủ yếu tiếng Trung; sentiment dùng bảng translated-comment, chủ yếu tiếng Anh'],
        ['Định dạng','Parquet; cấu hình, manifest, validation và từ điển ở JSON/Markdown'],
        ['Phạm vi','Snapshot nghiên cứu đang có trong working_13-10; không đại diện toàn bộ lịch sử XuetangX']]))
    pieces.append('''

Nguồn tổ chức và quy mô công bố đối chiếu tại [repository MOOCCubeX](https://github.com/THU-KEG/MOOCCubeX). Nhóm dự án và định nghĩa bài toán dựa trên [Review ngày 12/10/2026](Review-2026-10-12.docx), mục 2, QĐ9–QĐ10 và Phụ lục C. Danh sách feature từ log hiện có được người dùng chọn làm baseline để đối chiếu; không gắn nhãn là tập feature gốc của TEMPO-DQ.

### 1.1. Các quy mô cần phân biệt

''')
    pieces.append(md_table(['Phạm vi','Quy mô','Diễn giải'],[
        ['Người học trong nguồn',fmt(enroll['raw_user_identity'][0]['raw_user_rows']),'Số user nguồn; không phải số mẫu có nhãn'],
        ['Ghi danh nguồn',fmt(time['statistics']['n_input_pairs'])+' cặp; 4.701 course_id','Một cặp user–course duy nhất trong snapshot'],
        ['Catalog course',fmt(time['course_bounds']['n_courses']),'Số course_id có metadata trong file đầu vào hiện tại'],
        ['Course có hoạt động đã nối được','2.730','Tính từ nguồn hoạt động hợp lệ, chưa lọc điều kiện ghi danh'],
        ['Cặp đủ điều kiện thời gian',fmt(ft['n_pairs'])+f'; {fmt(ft["n_users"])} user; {fmt(ft["n_courses"])} course','Master giữ cả cặp không có chapter quan sát'],
        ['Chapter quan sát',fmt(quality['chapter_rows'])+' dòng; 173.840 cặp user–course','Bảng thưa; không tạo mọi tổ hợp user × chapter'],
        ['Cặp có nhãn cuối',fmt(lt['n_pairs'])+f'; {fmt(lt["n_users"])} user; {fmt(lt["n_courses"])} course','Cohort supervised hiện tại'],
        ['Kho feature bốn phase',fmt(n_snap)+' dòng','4 × mọi cặp đủ điều kiện thời gian'],
        ['Snapshot có nhãn',fmt(n_super)+' dòng','4 × cặp có nhãn; không phải số quan sát độc lập'],
        ['Feature tabular X','55','Không đếm metadata, target hoặc phần tử của khối chuỗi vào 55 cột này']]))
    pieces.append('''

## 2. Nguồn gốc dữ liệu

Nguồn gốc là dữ liệu nền tảng MOOC XuetangX, được nhóm MOOCCubeX thu nhận và tổ chức thành catalog tài nguyên, hồ sơ người học, các quan hệ và log hành vi. Đây là dữ liệu hành chính/hành vi ghi nhận trên nền tảng, không phải khảo sát hay cảm biến do dự án hiện tại thu thập. Bài báo mô tả quyền tiếp cận dữ liệu từ XuetangX và log video HeartBeat gốc theo nhịp 5 giây; đầu vào dự án hiện tại là bản đã xuất/biểu diễn thành các dòng hoặc segment. [Bài báo MOOCCubeX, CIKM 2021](https://lfs.aminer.cn/misc/moocdata/publications/mooccubex.pdf).

Nguồn trực tiếp của pipeline là [catalog Parquet notebook 01](input/moocubex_parquet/catalog.json) và [catalog dimension/bridge notebook 02](input/moocubex_model/catalog.json). Không thực hiện web scraping hay thu thập mới từ người học trong các bước 03–10. Ngày tải dump gốc, phiên bản remote/Kaggle và checksum của các file JSON gốc trước notebook 01 chưa được xác nhận từ manifest hiện có.

### 2.1. Những nguồn được dùng chính

''')
    raw_used=[('entities/user.json','Hồ sơ, course_order/enroll_time','Ghi danh và thuộc tính tĩnh'),
        ('entities/course.json','Catalog course và tổ chức tài nguyên','Course/chapter/resource'),
        ('entities/video.json','Thông tin video/caption','Duration proxy qua bridge, không thay thế log xem'),
        ('entities/problem.json','Catalog câu hỏi, type và max score','Mẫu số điểm, phân tích loại problem'),
        ('relations/user-video.json','Dòng user có mảng seq/segment','Hành vi xem; một dòng file không bằng một segment'),
        ('relations/user-problem.json','Trạng thái user–problem','Submit time, attempts, is_correct, score'),
        ('entities/translated-comment.json','Comment dịch đã có trong input','Comment và sentiment; quy trình dịch gốc chưa có metadata'),
        ('entities/reply.json','Reply','Tương tác xã hội và timestamp'),
        ('relations/course-teacher.txt','Liên kết course–teacher','Thuộc tính tĩnh'),
        ('relations/course-school.txt','Liên kết course–school','Đơn vị cung cấp course')]
    pieces.append(md_table(['Nguồn logic','Dòng trong input','Nội dung','Vai trò'],[
        [name,fmt(inv['raw/'+name]['n_rows']),content,role] for name,content,role in raw_used]))
    pieces.append('''

Các bridge `exercise–problem`, `video–ccid`, `course–comment`, `comment–reply` và các dimension/bridge notebook 02 hỗ trợ nối dữ liệu. [Tài liệu user của MOOCCubeX](https://github.com/THU-KEG/MOOCCubeX/blob/main/docs/user-en.md) là tham chiếu schema; code vẫn dùng kiểu dữ liệu và khóa thực tế của snapshot đã kiểm tra. Các bảng nguồn và đầu ra hiện có, gồm bảng ít dùng như concept/paper/other, được liệt kê trong [schema inventory](data_card/schema_inventory.json).

Repository gốc công bố **4.216 course**, khác **3.781 course** trong catalog đầu vào hiện tại. Không coi hai số là cùng một snapshot/phạm vi. Nguyên nhân đầy đủ của chênh lệch chưa xác minh; không được suy ra toàn bộ phần chênh chỉ là course đăng ký mà không học. Trong chính catalog 3.781 course, 1.051 course không có hoạt động nối được; 2.730 có hoạt động, trong đó 2.728 có ít nhất một cặp đủ điều kiện thời gian. [Quy mô công bố MOOCCubeX](https://github.com/THU-KEG/MOOCCubeX).

### 2.2. Khoảng thời gian

''')
    pieces.append(md_table(['Nguồn/phạm vi','Đầu thời gian','Cuối thời gian','Ghi chú'],[
        ['Ghi danh nguồn',time['statistics']['first_enroll_time'],time['statistics']['last_enroll_time'],'Theo timestamp parse trong pipeline'],
        *[[r['source'],r['first_time'],r['last_time'],
            'Raw có timestamp ngoài cửa sổ nghiên cứu; không dùng trực tiếp làm course_end' if r['source']=='video' else 'Khoảng quan sát raw; không có nghĩa mọi course được phủ liên tục'] for r in s['source_time_ranges']],
        ['Video được phép dùng','2020-01-19 00:00:00','trước 2020-11-04 00:00:00','Bao gồm hết 03/11/2020'],
        ['course_end của cặp hợp lệ',time['statistics']['first_course_end'],time['statistics']['last_course_end'],'Hoạt động hợp lệ cuối, không phải ngày đóng khóa chính thức']]))
    pieces.append('''

Cửa sổ video 19/01–03/11/2020 theo [MOOCCubeX, mục 3.1.3 và 5.2.1](https://lfs.aminer.cn/misc/moocdata/publications/mooccubex.pdf). Nhãn/feature không áp dụng một cửa sổ lịch chung 01/05–14/12 cho mọi nguồn; khoảng đó chỉ là một đối chiếu EDA trước đây. Unix epoch được chuyển với session DuckDB UTC; các chuỗi thời gian không có timezone được parse thành TIMESTAMP không kèm offset. Chưa có xác nhận đầy đủ về timezone gốc giữa các nguồn; đây là giới hạn cần kiểm tra.

### 2.3. Điều khoản và dữ liệu nhận dạng

Repository nguồn có tệp [GPL-3.0](https://github.com/THU-KEG/MOOCCubeX/blob/main/LICENSE); chưa xác nhận riêng phạm vi điều khoản đối với mọi dump được tải và quyền cấp phép lại bộ dẫn xuất. Bản dẫn xuất chưa được gán giấy phép riêng trong metadata. Model sentiment công bố CC-BY-4.0 trên [model card Cardiff](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest).

ID dạng U_/C_ và encoder là khóa giả danh phục vụ nối dữ liệu, không tự chứng minh vô danh hoàn toàn. Bảng X không dùng tên người học hoặc nguyên văn comment, nhưng nguồn/auxiliary tables vẫn có free text và dictionary ánh xạ mã. Chưa có kiểm tra PII, hồ sơ đồng thuận hay đánh giá công bằng độc lập cho bộ dẫn xuất; data card chỉ trình bày thống kê, không trích lời người học cụ thể.

## 3. Quy trình thu thập và xử lý

### 3.1. Các bước đã thực hiện

''')
    pieces.append(md_table(['Bước','Sản phẩm chính','Quyết định xử lý'],[
        ['01','input/moocubex_parquet','Người dùng chuyển JSON/TXT thành Parquet, giữ nguồn quan hệ và entity'],
        ['02','input/moocubex_model','Chuẩn hóa khóa, big chapter, dimension/bridge; tách quarantine và báo cáo độ phủ'],
        ['03','eda_review_course_enrollment','Kiểm tra user/course, ghi danh, timestamp, mapping; không tự suy diễn lượt học lại'],
        ['04 / 04b','eda_engagement_features_comparison; eda_problem_types_and_scores','EDA thành phần chỉ số, thời gian, loại problem và điểm; giữ cache dòng problem đã mapping'],
        ['05','review_time_phases','Course bounds và cutoff cá nhân P1–P4'],
        ['06','final_quality_tables','Feature con/chỉ số cuối kỳ theo chương, master user–course; mặc định AFELO deferred trước bước 07'],
        ['07','afelo_sentiment_cardiff; cập nhật final_quality_tables','Inference sentiment thật, AFELO_proxy, LOQV Review và nhãn gốc'],
        ['08','weighted_loqv_alpha_0p1','Tạo nhãn alpha=0,1; giữ phiên bản cũ 0,45–0,10–0,45 để đối chiếu'],
        ['09','phase_features','Tính lại X từ log trước cutoff, tạo chuỗi tuần, ghép target và fit tiền xử lý train-only'],
        ['10','DATA_CARD.md; data_card/','Đọc metadata/thống kê/schema để lập card; không sửa feature hoặc nhãn']]))
    pieces.append('''

```mermaid
flowchart TD
  A[01 Parquet nguồn] --> B[02 Dimension và bridge]
  A --> C[03 và 04 EDA cùng cache dòng sự kiện]
  B --> C
  C --> D[05 Course bounds và cutoff]
  D --> E[06 Thành phần cuối kỳ theo chương]
  E --> F[07 Sentiment và nhãn LOQV gốc]
  F --> G[08 Nhãn alpha 0.1]
  C --> H[09 Feature tính lại trước cutoff]
  D --> H
  G --> I[Supervised P1 đến P4]
  H --> I
```

`kich_ban_monitor_5_phase.md` là phương án cũ theo cửa sổ lịch; không phải giao thức phase đang dùng ở bản này. Bản hiện tại theo bốn phase của Review.

### 3.2. Khóa, ghi danh và điều kiện đưa vào

Khóa đối tượng là `(user_id, course_id)`. Trong snapshot có 11.807.090 dòng ghi danh và cùng số cặp duy nhất; không quan sát thấy nhiều dòng của cùng user–course. Điều này chỉ mô tả dữ liệu được cung cấp, không chứng minh không tồn tại học lại trong lịch sử hay qua course_id khác. Theo lựa chọn người dùng, không tạo enrollment_id tổng hợp hoặc nhóm course_id bằng tên/course_root_id chưa được xác minh.

Big chapter lấy thành phần số đầu của chapter, khóa trong phạm vi course bằng `lo_key`/`(course_id, chapter_major)`. Chỉ dùng mapping tài nguyên/problem xác định duy nhất về chương; quan hệ không nối được hoặc mơ hồ không được phân bổ tùy ý.

`course_start` là hoạt động hợp lệ đầu tiên và `course_end` là hoạt động hợp lệ cuối cùng của toàn course, từ video/problem/comment/reply đã mapping và kiểm tra thời gian; không lấy mốc đăng ký làm hoạt động. Mỗi cặp có `e = enroll_time`, dùng cùng `T = course_end` của course. Điều kiện chính: có course trong catalog và `enroll_time < course_end`.

''')
    pieces.append(md_table(['Trạng thái thời gian','Số cặp','Xử lý'],[
        [r['time_status'],fmt(r['n_pairs']), 'Tạo cutoff/feature' if r['time_status']=='ELIGIBLE_TIME' else 'Giữ trong bảng kiểm toán thời gian, không tạo snapshot hợp lệ'] for r in time['eligibility']]))
    pieces.append('''

Người đăng ký trước hoạt động đầu tiên của course vẫn được giữ nếu đủ điều kiện; người đăng ký sau course_end không tự gán nhãn kém. Theo [summary notebook 05](review_time_phases/summary.json), 1.189.991 cặp hợp lệ đăng ký trước hoạt động đầu tiên đã quan sát của course.

### 3.3. Quy tắc phase và cắt sự kiện

```text
t_uc(p) = enroll_time_uc + lambda_p * (course_end_c - enroll_time_uc)
lambda = {P1: 0.25, P2: 0.50, P3: 0.75, P4: 0.90}
X_uc(p) = feature(log của đúng user-course trong [enroll_time, t_uc(p)], thuộc tính tĩnh)
```

Cutoff làm tròn xuống microsecond theo phép chia nguyên; bốn cửa sổ tích lũy, không phải bốn đoạn không giao nhau. Hai user cùng course có thể có ngày cutoff khác nhau. Timestamp cutoff/course_end là metadata giao thức, không tự đưa vào X. Cặp không có log ở phase sớm vẫn có dòng feature.

Video được clip theo phần nội dung có thể quan sát: `observed_media = min(end-start, max(0, cutoff-event_time)*speed)`. Thời gian xem duy nhất hợp các khoảng nội dung đã clip theo user–course–video. Không nhân tỷ lệ phase với tổng watch time cuối kỳ. Với problem, toàn dòng aggregate chỉ khả dụng khi `submit_time <= cutoff`; không thể tái dựng từng lần thử. Comment và reply nhận/viết đều lọc timestamp trước aggregate, gồm cả cutoff của người nhận reply.

### 3.4. Làm sạch điểm, comment và timestamp

Định dạng user/problem được chuẩn hóa về U_/Pm_. Timestamp parse được nhưng nằm ngoài cửa sổ nguồn vẫn bị loại khi cần; năm 2040/2080 không được coi là thời gian khóa học hợp lệ chỉ vì parse thành công. Trong 524.189 segment video đã mapping, 13 nằm ngoài cửa sổ video và không được dùng; toàn nguồn có 195 segment ngoài cửa sổ.

Score hợp lệ trong miền `[0,max_score]` được giữ, kể cả điểm từng phần khi is_correct=0. Chỉ fallback `is_correct*max_score` khi score không hợp lệ/thiếu và max score phù hợp. Không mặc định max score=1, không thay score thiếu bằng 0; dòng điểm chưa giải được được đánh dấu ở bảng kiểm toán/thành phần.

Comment hợp lệ loại text rỗng và bản trùng cùng user/resource/time/normalized-text. Không lọc toàn bộ nội dung ngoài học tập bằng ngữ nghĩa; comment điểm danh, phản ánh giao diện hoặc bài viết ngắn có thể còn lại. Không gán sentiment trung tính cho người không viết comment. Reply được nối về comment/tài nguyên và loại timestamp trước parent comment.

''')
    pq=next(x for x in s['problem_quality'] if x['scope']=='ALL_SOURCE')
    pieces.append(md_table(['Kiểm tra nguồn problem','Số lượng','Diễn giải'],[
        ['Tổng dòng user_problem',fmt(pq['n_rows']),'Trạng thái theo user–problem'],
        ['Score NULL',fmt(pq['score_null'])+'; '+percent(pq['score_null'],pq['n_rows']),'Không chỉ xảy ra ở câu tự luận'],
        ['Score âm',fmt(pq['score_negative']),'Sentinel/không hợp lệ, không dùng như điểm âm thật'],
        ['Dòng thiếu/không có max score phù hợp',fmt(pq['score_unresolved']),'Chưa thể giải score bằng quy tắc hiện tại'],
        ['Điểm từng phần với is_correct=0',fmt(pq['partial_credit_correctness_zero']),'Không được ghi đè mọi điểm hợp lệ bằng is_correct×max_score']]))
    type_names={1:'Một lựa chọn',2:'Nhiều lựa chọn',3:'Bình chọn',4:'Điền khuyết',5:'Chủ quan/tự luận',6:'Đúng–sai',9:'Lập trình'}
    pieces.append('\n\nTrong catalog thực tế có các type sau; exercise là nhóm problem, không phải một type câu hỏi:\n\n')
    pieces.append(md_table(['type','Nhãn phổ biến theo typetext','Số problem'],[
        [r['type'],type_names[r['type']],fmt(r['n_problems'])] for r in s['problem_type_profile']]))
    pieces.append('''

Có 1.057 problem lệch type–typetext so với nhãn phổ biến trong cùng type. Không giả định các mã 0/1/2 là ba loại duy nhất. Chi tiết ở [EDA problem](eda_problem_types_and_scores/summary.json).

### 3.5. Công thức chỉ số và nhãn cuối kỳ

Các thành phần dưới đây theo [schema mục 12 của dự án](moocubex_schema_va_danh_gia_3_chi_so.md), không tuyên bố mọi proxy tái lập nguyên bản QASOC/LoDiBi.

```text
COELO = mean_non_null(TS, RV, AV, IF), cần ít nhất 2 thành phần
TS = clip(unique_watch_seconds / caption_duration_proxy, 0, 1)
     chỉ có khi catalog duration của cả chương đầy đủ và mẫu số > 0
RV = n_videos_watched / n_videos_available
AV = n_exercises_attempted / n_exercises_available
IF_raw = total_problem_attempts / n_problems_available
IF = clip(log1p(IF_raw) / log1p(P95_train), 0, 1)
P95_train = 1.0, fit từ các dòng user-course-chapter cuối kỳ của train

sentiment_valence = ((sum(p_pos)-sum(p_neg))/n_sentiment_comments + 1)/2
AFELO_proxy = mean_non_null(comment_participation, reply_participation,
                           response_received_rate, sentiment_valence)
comment_participation = 1 nếu có comment hợp lệ, ngược lại 0
reply_participation = 1 nếu người học có reply, ngược lại 0
response_received_rate = số comment đã nhận reply / số comment hợp lệ

KL = earned_score_sum / available_max_score_sum
     chỉ có khi đủ catalog max score và không còn attempted score chưa giải được
CFA = sum(is_correct/attempts) / n_problems_available
ACELO = mean_non_null(KL, CFA), cần ít nhất 1 thành phần
```

`mean_non_null` bỏ thành phần null, không coi null là 0. Các tỷ lệ không xác định giữ null. ACELO có các mức KL_AND_CFA, CFA_ONLY hoặc UNAVAILABLE; CFA_ONLY không có cùng độ đầy đủ thông tin như đủ cả hai thành phần. Các counts/điểm/mẫu số được giữ để tái lập chỉ số.

Theo lựa chọn người dùng, chương hoàn toàn chưa quan sát không tạo dòng và không tự gán LOQV=1. Tuy nhiên, chương đã có hoạt động khác mà không có comment/reply vẫn nhận AFELO_proxy=0 do hai chỉ báo tham gia bằng 0, trong khi sentiment/response rate null. Đây là hệ quả công thức proxy, không phải model sentiment đánh giá tiêu cực.

Ba phiên bản LOQV đã lưu:

```text
Review:     LOQV = sqrt(((1-C)^2 + (1-F)^2 + (1-A)^2)/3)
Bản cũ:    LOQV = sqrt(0.45*(1-C)^2 + 0.10*(1-F)^2 + 0.45*(1-A)^2)
Hiện dùng: LOQV = sqrt(((1-C)^2 + 0.1*(1-F)^2 + (1-A)^2)/2.1)
            C=COELO, F=AFELO_proxy, A=ACELO

q_uc = mean(LOQV của các chương quan sát)
       chỉ có nếu mọi chương quan sát của cặp đều đủ ba chỉ số
y = E nếu q<0.2; G nếu 0.2<=q<0.4; A nếu q>=0.4
```

Nhãn hiện tại có raw weights `(1,0.1,1)`, normalized weights `(10/21,1/21,10/21)`, xấp xỉ `(0,476190; 0,047619; 0,476190)`. Alpha=0,1 không phải normalized weight AFELO=0,1; bản cũ 0,45–0,10–0,45 tương đương alpha=2/9. Chỉ alpha=1 khôi phục công thức LOQV nguyên bản Review. Giá trị q/LOQV càng nhỏ càng gần điểm lý tưởng; E/G/A là lớp chất lượng vận hành theo proxy, không phải nhãn chuẩn do chuyên gia gán.

Nhãn được tính trên `[enroll_time,course_end]` và cố định cho cả P1–P4. Bảng gốc khớp hàm tham chiếu Review trên 832.379 dòng có LOQV, sai số tối đa 2,22×10⁻¹⁶, theo [kiểm tra công thức](final_quality_tables/review_loqv_formula_audit.json).

''')
    old_dist={r['class_label']:r['n_weighted'] for r in old['class_distribution']}
    pieces.append(md_table(['Lớp','Review đều','0,45–0,10–0,45','Alpha=0,1 hiện dùng'],[
        [r['class_label'],fmt(r['n_equal']),fmt(old_dist[r['class_label']]),fmt(r['n_weighted'])] for r in alpha['class_distribution']]))
    pieces.append('\n\nĐiều kiện có nhãn không thay đổi giữa ba phiên bản. Master vẫn giữ mọi cặp:\n\n')
    pieces.append(md_table(['Trạng thái cuối kỳ','Số cặp'],[[r['label_status_weighted'],fmt(r['n_rows'])] for r in alpha['label_status']]))
    pieces.append('''

### 3.6. Inference sentiment để tính AFELO

''')
    pieces.append(md_table(['Thuộc tính','Giá trị'],[
        ['Model',s['model']['model']['model_id']],['Revision cố định',f'`{s["model"]["model"]["revision"]}`'],
        ['Chạy thực tế',f'{fmt(sent["unique_model_texts"])} nội dung sau tiền xử lý, ánh xạ về đủ {fmt(sent["comments"])} comment'],
        ['Thiết bị/precision',s['model']['inference']['device']+' / '+s['model']['inference']['dtype']],
        ['Đầu ra','p_neg, p_neu, p_pos; nhãn dự đoán và confidence/entropy'],
        ['id2label','0=negative, 1=neutral, 2=positive'],
        ['Tiền xử lý','Theo model card: @name→@user; URL bắt đầu bằng http→http'],
        ['Comment dài',f'{fmt(sent["truncated_comments"])} dòng bị cắt đầu đến 512 token gồm special tokens, gắn was_truncated'],
        ['Comment chứa CJK',fmt(sent['comments_with_cjk'])+'; giữ và gắn has_cjk'],
        ['Fine-tuning trên dataset','Không; frozen pretrained weights'],
        ['Kiểm tra thủ công','manual_audit_sample.parquet có 90 mẫu; chưa có nhãn sentiment do người gán']]))
    pieces.append('''

Model được thiết kế cho tiếng Anh và được huấn luyện trên tweet, theo [model card Cardiff](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest). Dùng cho comment dịch trong giáo dục là chuyển miền; confidence không phải accuracy đã kiểm định. Mỗi bản ghi comment có xác suất riêng; cache text trùng chỉ giảm số lần inference, không làm mất người học/comment khi aggregate.

''')
    pieces.append(md_table(['Sentiment dự đoán','Số comment'],[[r['predicted_label'],fmt(r['n_comments'])] for r in sent['label_distribution']]))
    pieces.append('''

### 3.7. Feature, encoder và tiền xử lý

Feature gồm hành vi video, làm bài, comment/reply, thống kê hoạt động và context tĩnh; định nghĩa từng cột ở mục 4. Các chỉ số COELO/AFELO/ACELO, q/y, trạng thái nhãn, cờ thiếu, course_end, course_id không nằm trong allowlist X. `phase_fraction` là feature chỉ báo tiến độ cho Pool B.

Số đếm/thời lượng không có log được biểu diễn 0 trong phạm vi nguồn quan sát. Mean/rate/recency chưa xác định giữ null ở `features/` và `supervised_raw/`; `model_ready/` dùng median rồi standard scale `(x_imputed-mean)/scale` fit trên train. Std gần 0 dùng scale=1. Không thêm missing indicators; hiện không có cột all-null-train bị loại.

Vocabulary người học và tập giảng viên/trường/lĩnh vực được sort, fit trên các cặp train có nhãn; unknown/missing→0. Gender giữ mã nguồn 0/1/2, giá trị khác/thiếu→-1; chưa diễn giải 0/1/2 thành tên giới tính khi chưa có mapping xác nhận. Tập teacher/school/field được mã hóa theo toàn bộ chữ ký tập ID đã sort, chưa khớp encoding gốc TEMPO-DQ. Các mã là category, không phải số đo thứ bậc; classifier cần xử lý categorical hoặc one-hot fit train. Selector hình học/kernel dùng riêng 50 cột `distance_feature_columns`, không dùng mã ID làm tọa độ.

Encoder và imputer/scaler chỉ phụ thuộc cohort train và X; không dùng giá trị q/y để tính thống kê. Nếu đổi nhãn mà cùng cohort, feature log giữ nguyên; cần ghép y mới và chạy lại các bước học có dùng y. Nếu đổi cohort, cutoff, nguồn hoặc quy tắc feature thì cần rà/refit các bước liên quan.

Chuỗi tuần gồm `week_index=floor((event_time-enroll_time)/(7 ngày))` và 8 lượng hành vi; chỉ tạo tuần có sự kiện, sort theo week_index. Có thể khôi phục khoảng trống từ week_index; không tự nội suy tuần trống. Lượng segment được gán vào tuần bắt đầu sự kiện, không phải phép phân bổ thời gian liên tục qua từng ngày. Cohort supervised giữ chuỗi rỗng cho cặp chưa có sự kiện.

### 3.8. Train/dev/test và phân bố nhãn

Sắp course theo observed course_start rồi course_id, chia 70/10/20 theo **số course** trước khi fit P95, encoder, imputer và scaler. Mọi snapshot của một course/user–course ở cùng split. Không yêu cầu user_id rời nhau giữa các split. Không nhóm bằng tên hoặc course_root_id chưa xác minh.

''')
    pieces.append(md_table(['Split','Course đủ điều kiện','Tổng cặp','Course có nhãn','User có nhãn','Cặp có nhãn','Snapshot có nhãn'],[
        [r['split'],fmt(r['n_courses']),fmt(r['n_pairs']),fmt(r['n_labeled_courses']),fmt(r['n_labeled_users']),fmt(r['n_labeled_pairs']),fmt(4*r['n_labeled_pairs'])] for r in s['split_population']]))
    by_split={(r['split'],r['y_weighted']):r['n_rows'] for r in alpha['class_by_split']}
    pieces.append('\n\nPhân bố nhãn theo cặp user–course, giống nhau ở cả bốn phase:\n\n')
    pieces.append(md_table(['Split','E','G','A','Tỷ lệ G'],[
        [r['split'],fmt(by_split.get((r['split'],'E'),0)),fmt(by_split.get((r['split'],'G'),0)),fmt(by_split.get((r['split'],'A'),0)),
            percent(by_split.get((r['split'],'G'),0),r['n_labeled_pairs'])] for r in s['split_population']]))
    pieces.append(f'''

Pool B train có **{fmt(pre['fit_rows'])} snapshot**. Split theo số khóa không bảo đảm tỷ lệ số dòng 70/10/20; các khóa train có quy mô lớn hơn đáng kể. Có **{overlap['n_train_ending_after_first_test_start']}/{overlap['n_train_courses']} course train ({percent(overlap['n_train_ending_after_first_test_start'],overlap['n_train_courses'])})** kết thúc sau ngày mở sớm nhất của test. Đây là split theo course/thứ tự mở khóa, **chưa phải đánh giá vận hành với nhãn train luôn có sẵn trước lúc dự đoán test**; phương án NOLFO chưa chạy.

Giá trị q trung bình theo lớp trên cặp train của nhãn alpha=0,1 (thống kê mô tả của data card, chưa phải kết quả đánh giá mô hình):

''')
    pieces.append(md_table(['Lớp train','Số cặp','Mean q'],[[r['y'],fmt(r['n_pairs']),num(r['mean_q'])] for r in s['q_train_class_means']]))
    pieces.append('''

Không có mean q cho E vì train không có E; không được tự tạo giá trị μ_E=0. Chất lượng cấp course theo Review yêu cầu tổng hợp từ người học phù hợp định nghĩa L_c. Bản hiện tại chỉ có nhãn cho một phần ghi danh, nên trung bình q trên cohort có nhãn không được tự gọi là chất lượng thật của toàn course. Phân bố số cặp có nhãn/course là thống kê **độ phủ**, không phải tổng số người ghi danh:

''')
    nc=s['labeled_per_course']
    pieces.append(md_table(['Thống kê cặp có nhãn/course','Giá trị'],[
        ['Course có nhãn',fmt(nc['n_courses'])],['Min / Q1 / median / Q3 / max',
            ' / '.join(fmt(x) for x in [nc['min_n'],*nc['quartiles'],nc['max_n']])],
        ['Course có dưới 30 cặp có nhãn',fmt(nc['n_courses_under_30'])],
        ['Course có hai lớp người học',fmt(sum(r['n_courses'] for r in s['within_course_label_variation'] if r['n_classes']>1))]]))
    pieces.append('''

### 3.9. Kiểm tra kỹ thuật và tái lập

''')
    pieces.append(md_table(['Nhóm kiểm tra','Kết quả','Bằng chứng'],[
        ['Chỉ số cuối kỳ','7 kiểm tra nền + 6 fixture; 12 kiểm tra final','final_quality_tables/validation.json'],
        ['Sentiment','7 kiểm tra xác suất/khóa + 9 kiểm tra sau chạy','afelo_sentiment_cardiff/validation.json và validation_postrun.json'],
        ['LOQV alpha=0,1','15/15','weighted_loqv_alpha_0p1/validation.json'],
        ['Feature phase','28/28 kiểm tra dữ liệu + 4 fixture','phase_features/validation.json'],
        ['Chèn log tương lai','9/9; toàn bộ 55 feature P1 giữ nguyên','phase_features/temporal_invariance_validation.json']]))
    pieces.append('''

Các kiểm tra xác nhận khóa/số dòng, cửa sổ timestamp, nhãn cố định, phép tổng hợp, cutoff khớp notebook 05 và tiền xử lý train-only. Chúng không đo accuracy sentiment, tính hợp lệ giáo dục của proxy, fairness, chất lượng dự đoán hoặc khả năng vận hành thật. Các probe/hoán vị nhãn và thí nghiệm đầy đủ ở Review chưa được thực hiện bởi bước tạo feature/data card.

Môi trường chạy đã ghi nhận: Python 3.9.6, DuckDB 1.3.2, PyTorch 2.8.0, Transformers 4.57.6, tokenizers 0.22.2, huggingface-hub 0.36.2, NumPy 2.0.2. Giữ DuckDB 1.3.2 để tái lập hash-shard. Model được pin revision; tham số chuẩn hóa, split, source fingerprint và checksum ở các manifest. Ngày tạo feature ghi ở manifest là 17/10/2026; đây không phải ngày thu thập log.

Các notebook 01–02 do người dùng đã chạy là đầu vào. Thứ tự tái dựng sau đó:

```bash
python 04_eda_engagement_features_comparison.py
python 05_build_review_time_phases.py
python 06_build_final_quality_tables.py
python 07_compute_afelo_sentiment.py
python 08_build_weighted_loqv.py --alpha 0.1
python 09_build_phase_features.py
python tests/test_phase_feature_cutoff.py
python 10_build_data_card.py --as-of 2026-10-17
```

Chạy từ `working_13-10` với đúng Python environment, hoặc chỉnh đường dẫn/config khi chạy ở root khác. Chạy lại 06 đưa AFELO về deferred; phải chạy tiếp 07 rồi 08–09. Bước 09 hiện cần cache problem rows của EDA 04 và event stages của 06; fingerprint metadata phải khớp input. Đường dẫn, vocabulary và nguồn local không tự trở thành URI tải công khai; khi chuyển máy cần mang đủ input/cache hoặc dựng lại bước nguồn. Kiểm tra checksum/evidence của card ở [data_card/run_manifest.json](data_card/run_manifest.json).

## 4. Cấu trúc dữ liệu

### 4.1. Các bảng sử dụng chính

''')
    tables=[
        ('quality/coelo_components','final_quality_tables/tables/coelo_components.parquet','user–course–chapter'),
        ('quality/afelo_components','final_quality_tables/tables/afelo_components.parquet','user–course–chapter'),
        ('quality/acelo_components','final_quality_tables/tables/acelo_components.parquet','user–course–chapter'),
        ('quality/chapter_quality','final_quality_tables/tables/chapter_quality.parquet','user–course–chapter'),
        ('quality/user_course_final','final_quality_tables/tables/user_course_final/*.parquet','user–course'),
        ('sentiment/comment_sentiment','afelo_sentiment_cardiff/tables/comment_sentiment.parquet','comment'),
        ('weighted_loqv_alpha_0p1/chapter_loqv_weighted','weighted_loqv_alpha_0p1/tables/chapter_loqv_weighted.parquet','user–course–chapter'),
        ('weighted_loqv_alpha_0p1/user_course_loqv_weighted','weighted_loqv_alpha_0p1/tables/user_course_loqv_weighted.parquet','user–course'),
        ('phase/user_course_targets','phase_features/tables/user_course_targets.parquet','user–course'),
        ('phase/features/P1','phase_features/features/P1/*.parquet','user–course–phase, P1; tương tự P2–P4'),
        ('phase/supervised_raw/P1','phase_features/supervised_raw/P1.parquet','user–course–phase, P1; tương tự P2–P4'),
        ('phase/model_ready/P1','phase_features/model_ready/P1.parquet','user–course–phase, P1; tương tự P2–P4'),
        ('phase/supervised_sequences/P1','phase_features/supervised_sequences/P1.parquet','user–course–phase, P1; tương tự P2–P4')]
    pieces.append(md_table(['Đường dẫn tương đối','Đơn vị','Dòng','Cột','File'],[
        [f'`{path}`',grain,fmt(inv[name]['n_rows']),inv[name]['n_columns'],inv[name]['n_files']] for name,path,grain in tables]))
    pieces.append(f'''

Các thư mục chia shard là **một bảng logic**, không cộng số bảng/shard thành số mẫu. Kho X thô bốn phase có 128 file, {fmt(n_snap)} dòng; các bảng model_ready có 4 file, {fmt(n_super)} dòng. Card kiểm kê **{len(inventory)} bảng logic** gồm raw/model/EDA/time/quality/sentiment/weighted/phase/encoder tại [schema_inventory.json](data_card/schema_inventory.json); `_stage` và `_scratch` là cache trung gian, không thuộc bộ bảng dùng trực tiếp để huấn luyện.

`chapter_details` của master gốc là `LIST<STRUCT>` chứa đầy đủ feature con/chỉ số từng chương. Bảng weighted user–course giữ feature scalar và q/y cũ/mới; chi tiết chương còn ở bảng chapter tương ứng và master gốc. Dòng chỉ số null vẫn được giữ qua LEFT JOIN. Bảng chapter ít dòng hơn master vì là các chapter quan sát của một phần user–course; không phải vì mất toàn bộ cặp sau inner join.

### 4.2. Metadata và target của bảng phase

''')
    pieces.append(md_table(['Cột','Kiểu model_ready','Vai trò'],[
        ['user_id',ready_schema['user_id'],'Khóa ghép; bản thô không dùng làm X, learner_code là category riêng'],
        ['course_id',ready_schema['course_id'],'Khóa/split/group; không là predictor'],
        ['phase',ready_schema['phase'],'Khóa snapshot; phase_fraction là feature tiến độ riêng'],
        ['split',ready_schema['split'],'Metadata train/dev/test, không là predictor'],
        ['enroll_time',ready_schema['enroll_time'],'Đầu cửa sổ, không là predictor trong allowlist'],
        ['cutoff_time',ready_schema['cutoff_time'],'Mốc cắt, không là predictor trong allowlist'],
        ['q',ready_schema['q'],'Target liên tục cuối kỳ alpha=0,1; không đưa vào X'],
        ['y',ready_schema['y'],'Nhãn A/G/E cuối kỳ; hiện chỉ A/G có mẫu'],
        ['target_alpha',ready_schema['target_alpha'],'Metadata phiên bản nhãn'],
        ['target_formula_version',ready_schema['target_formula_version'],'Metadata phiên bản nhãn']]))
    pieces.append('''

Bảng feature thô có 61 cột = 6 metadata + 55 X; `supervised_raw`/`model_ready` có 65 cột = 6 metadata + 55 X + 4 trường target/phiên bản. **65 cột không đồng nghĩa 65 feature gốc TEMPO-DQ.** Dùng [feature_columns.json](phase_features/feature_columns.json) làm allowlist thay vì chọn mọi cột số.

### 4.3. Từ điển 55 feature

Kiểu raw lấy từ `features/P1`, kiểu processed từ `model_ready/P1`; ba phase còn lại cùng schema. Cột số processed là DOUBLE sau impute/scale, mã danh mục giữ kiểu số nguyên nhưng phải được hiểu như category.

''')
    pieces.append(md_table(['Feature','Nhóm','Kiểu raw → processed','Định nghĩa'],[
        [f'`{r["name"]}`',r['group'],raw_schema[r['name']]+' → '+ready_schema[r['name']],r['definition']] for r in s['feature_dictionary']]))
    pieces.append('''

Danh sách và định nghĩa có thể đọc riêng tại [tu_dien_features.md](phase_features/tu_dien_features.md); các tham số số học train thực tế ở [preprocessing.json](phase_features/preprocessing.json). Các cột như `has_observed_activity`, min/max event time và n_out_of_window_events chỉ có ở audit, không là feature X.

### 4.4. Khối chuỗi và quan sát phase trống

`behavior_sequence` là `LIST<STRUCT>` gồm `week_index INTEGER` và 8 lượng DOUBLE: n_video_segments, video_media_seconds, n_problem_submissions, problem_attempts, n_correct_problems, n_comments, n_replies_authored, n_replies_received. `supervised_sequences` có 4 cột khóa/metadata và cột sequence; ghép theo user_id, course_id, phase. Không coi mỗi tuần hay mỗi chapter là một mẫu nhãn độc lập.

''')
    empty={p:sum(r['n_no_observed_activity'] for r in phase['supervised_phase_rows'] if r['phase']==p) for p in ['P1','P2','P3','P4']}
    pieces.append(md_table(['Phase','Cặp có nhãn','Chưa có sự kiện quan sát','Tỷ lệ','Chuỗi rỗng'],[
        [p,fmt(lt['n_pairs']),fmt(empty[p]),percent(empty[p],lt['n_pairs']),fmt(next(r['n_empty_sequences'] for r in phase['supervised_sequence_rows'] if r['phase']==p))] for p in ['P1','P2','P3','P4']]))
    pieces.append('''

Các dòng trống được giữ; không lọc cohort theo việc đã có video/bài tập/comment ở P1. Mean/rate không xác định ở raw và chuỗi rỗng là các cách biểu diễn khác nhau, không tự suy diễn chất lượng kém.

## 5. Mục đích sử dụng dự kiến

Mục tiêu chính là huấn luyện và so sánh mô hình **phân loại sớm ở cấp user–course** từ feature P1–P4, dự đoán nhãn cuối kỳ đã định nghĩa. Có thể dùng logistic regression, cây/boosting, MLP hoặc learner phù hợp dữ liệu bảng; các mã danh mục cần encode đúng. Chuỗi tuần cho phép thí nghiệm mô hình chuỗi khi có thiết kế padding/biểu diễn rõ ràng. Không có mô hình nào trong các nhóm này đã được benchmark trong bước tạo card.

Thiết kế nghiên cứu chính theo Review là đánh giá **coreset trên Pool B**: chọn mẫu snapshot thuộc course train, giữ feature tiền xử lý chung, so sánh với FullTrain-B ở cùng learner/config/ngân sách; đánh giá trên course test chưa xuất hiện trong train. Với selector dựa trên khoảng cách, chỉ lấy `distance_feature_columns`; với learner, lấy `model_feature_columns`, y và sample weight nếu có. Khi đánh giá theo course, bootstrap/group bằng course_id theo lựa chọn hiện tại, không lấy snapshot là mẫu độc lập.

Endpoint dự kiến: Macro-F1 theo phase và trung bình P1–P4; báo cáo thêm chỉ số từng lớp, đặc biệt việc lớp E không có hỗ trợ dữ liệu. Bài toán ba lớp đầy đủ chưa thể được xác nhận thực nghiệm với bản hiện tại. Không dùng accuracy tổng thể để che phân bố A/G lệch và không tự sinh nhãn E.

Endpoint phụ tổng hợp cấp course cần xử lý rõ độ phủ người học có nhãn, μ_y, các course nhỏ và điều kiện thời gian trước khi dùng làm kết quả chính thức. Mean q trong cohort quan sát chưa là ground truth toàn bộ ghi danh. Card không công bố Q_c/Y_c, ICC(1), oracle floor, missing-mask probe hoặc ID-only probe đã hoàn thành nếu chưa có kết quả tương ứng.

Nhãn hiện tại mô tả proxy chất lượng vận hành theo hành vi/điểm, không phải điểm kiểm định thiết kế course, chứng nhận năng lực cá nhân hay kết luận chuyên gia về người học/giảng viên. Giới hạn miền dữ liệu và nguồn phải được giữ khi tái sử dụng ngoài nghiên cứu này.

## 6. Hạn chế và thiên kiến

### 6.1. Độ phủ và thiên lệch chọn mẫu

''')
    pieces.append(md_table(['Nguồn','Tổng đơn vị EDA','Nối duy nhất được','Tỷ lệ','Đơn vị'],[
        [r['source'],fmt(r['total_events']),fmt(r['uniquely_mapped']),percent(r['uniquely_mapped'],r['total_events']),
            'segment' if r['source']=='video_segments' else 'dòng nguồn'] for r in s['source_mapping_coverage']]))
    pieces.append(f'''

Các tỷ lệ trên là độ phủ mapping ở EDA nguồn, trước mọi lọc cuối kỳ; không trộn mẫu số với số event thực dùng ở phase. Chỉ {fmt(lt['n_pairs'])}/{fmt(ft['n_pairs'])} cặp đủ điều kiện thời gian có nhãn ({percent(lt['n_pairs'],ft['n_pairs'])}). Cohort học có thể nghiêng mạnh về course/người học có log nối được và đủ mẫu số chỉ số. Giữ đủ master và feature không xóa thiên lệch của subset supervised. Độ phủ khác nhau giữa nền tảng, tài nguyên, học phần và nguồn hành vi làm việc khái quát sang toàn bộ người đăng ký chưa được bảo đảm.

Nguồn thuộc MOOC XuetangX và giai đoạn lịch sử 2019–2020; không có thiết kế lấy mẫu đại diện mọi quốc gia, nhóm tuổi hay loại hình giáo dục. Raw metadata có ít hơn quy mô công bố và các namespace ID/bridge không luôn một–một. Chưa xác minh đầy đủ nguyên nhân thiếu catalog và chưa nhóm các đợt mở cùng khóa gốc; cùng nội dung có thể tồn tại dưới course_id khác ở các split.

### 6.2. Tính hợp lệ của nhãn và độ phủ cảm xúc

{fmt(afdiag['n_with_sentiment'])}/{fmt(afdiag['n_chapters'])} chương quan sát có sentiment ({percent(afdiag['n_with_sentiment'],afdiag['n_chapters'])}); {fmt(afdiag['n_afelo_zero'])} chương có AFELO_proxy=0 ({percent(afdiag['n_afelo_zero'],afdiag['n_chapters'])}) do quy ước vắng comment/reply. Với alpha=0,1, ngay cả C=A=1 nhưng F=0 thì `LOQV=sqrt(0.1/2.1)` ≈ 0,218218, vẫn không đạt E. Đây là đặc tính thước đo, không chứng minh cảm xúc tiêu cực hay chất lượng giáo dục thật thấp.

Nhãn alpha=0,1 hiện có E=0, G={fmt(next(r['n_weighted'] for r in alpha['class_distribution'] if r['class_label']=='G'))}, A={fmt(next(r['n_weighted'] for r in alpha['class_distribution'] if r['class_label']=='A'))}; min q={num(alpha['q_distribution']['min_q_weighted'])}. Nhãn không do chuyên gia gán, chưa được xác thực bằng tiêu chuẩn bên ngoài. ACELO của 376.983 chương chỉ có CFA; tính đủ nhãn về mặt code không đồng nghĩa mọi chiều được đo đầy đủ hoặc đồng nhất.

Giảm alpha làm thay đổi cách đo và phân lớp; không tự nâng chất lượng thực tế của course/người học. Các phiên bản được xây dựng trong quá trình đã xem thống kê phân bố nhãn; chưa có hồ sơ khóa giao thức/đăng ký trước cuối cùng. Không có tối ưu alpha tự động theo score mô hình, nhưng cũng chưa đủ cơ sở tuyên bố test chưa được thăm dò hoặc giao thức đã đăng ký trước. Cần giữ lịch sử phiên bản khi lập thí nghiệm.

### 6.3. Missing data và mã danh mục chưa thấy

''')
    pieces.append(md_table(['Phase','Tuổi thiếu','Mean video thiếu','Problem rate thiếu','Score ratio thiếu','Sentiment thiếu'],[
        [r['phase'],percent(r['missing_age'],r['n_rows'],4),percent(r['missing_video_mean'],r['n_rows']),
            percent(r['missing_problem_rate'],r['n_rows']),percent(r['missing_attempted_score'],r['n_rows']),percent(r['missing_sentiment'],r['n_rows'])]
        for r in s['raw_missing_by_phase']]))
    pieces.append('''

Tỷ lệ trong bảng là trên raw cohort supervised của từng phase, trước imputation. Tuổi chỉ có giá trị hợp lệ ở 58/164.429 cặp mỗi phase; không đủ cơ sở suy luận fairness theo tuổi. Nguồn gender chưa có mapping ý nghĩa mã được xác nhận. Median imputation giữ đủ dòng nhưng không khôi phục thông tin bị thiếu và có thể che độ bất định.

Mã 0 gộp danh mục thiếu hoặc không có trong vocabulary train. Với teacher_set_code, unknown là **chữ ký toàn bộ tập giảng viên chưa thấy**, không đồng nghĩa từng giảng viên đều mới. Độ phủ mã theo cặp của P1 (các mã tĩnh giữ nguyên qua phase):

''')
    pieces.append(md_table(['Split','Learner code=0','Teacher set=0','School set=0','Field set=0'],[
        [r['split'],percent(r['learner_unknown'],r['n_pairs']),percent(r['teacher_set_unknown'],r['n_pairs']),
            percent(r['school_set_unknown'],r['n_pairs']),percent(r['field_set_unknown'],r['n_pairs'])] for r in s['categorical_unknowns']]))
    pieces.append('''

Đặc biệt field_set_code của toàn bộ test bằng 0, teacher_set_code của gần toàn bộ test bằng 0; chưa thể giả định các cột này hữu ích hoặc encoding này tương đương TEMPO-DQ. Không thêm missing indicators không loại mọi tín hiệu về độ phủ: n_events, các số đếm nguồn và các giá trị sau impute vẫn có thể tương quan với tình trạng thiếu dữ liệu. Missing-mask/ID probe và phân tích fairness theo nhóm chưa được chạy.

### 6.4. Giới hạn thời gian, label source và khả năng vận hành

Course_end là mốc hoạt động cuối suy ra hồi cứu. Các phase theo tỷ lệ thời gian còn lại có thể không xác định được trước trong hệ thống thật nếu lịch đóng course không biết sẵn. Tỷ lệ khóa train kết thúc sau ngày mở đầu tiên của test rất cao; chưa chạy NOLFO và chưa bảo đảm availability của nhãn train trước mọi thời điểm dự đoán. Không gọi kết quả hiện tại là đánh giá triển khai thời gian thực.

Nguồn chuỗi timestamp có timezone chưa xác nhận; mapping sai giờ sẽ ảnh hưởng enrollment/event/cutoff. Catalog, teacher/school và user profile được coi là tĩnh vì không có lịch sử hiệu lực. Event-time checks chỉ kiểm tra các timestamp đã parse và giả định này, không chứng minh trạng thái metadata tồn tại đúng như vậy ở thời điểm gốc.

User-problem aggregate không cung cấp chuỗi từng attempt; một người làm bài trước phase cutoff rồi nộp lại sau cutoff có thể không có bản trạng thái sớm để quan sát. Caption duration chỉ là proxy, thiếu catalog duration/max score khiến một phần thành phần không tính được. Các quy tắc fallback và mean_non_null làm chất lượng đo lường giữa các chương khác nhau.

### 6.5. Sentiment, công bằng và những nội dung còn mở

Model tiếng Anh học từ tweet khác miền giáo dục và dữ liệu nguồn tiếng Trung được dịch. Quy trình dịch upstream chưa được ghi lại; 31 comment bị truncation, 41 chứa CJK, và comment ngoài nội dung học có thể còn. Một câu có sắc thái tiêu cực về nội dung thảo luận không nhất thiết là đánh giá tiêu cực course. Chưa có nhãn sentiment người gán hoặc đánh giá lỗi theo ngôn ngữ/độ dài/chủ đề.

Chưa có đánh giá công bằng theo giới tính, tuổi, trường, lĩnh vực hay người học cũ/mới. Sự khác nhau về hành vi viết comment, cấu trúc đánh giá, lịch mở course và độ phủ tài nguyên có thể ảnh hưởng proxy/nhãn; đây là nguy cơ cần kiểm chứng, không phải kết luận đã đo được. Chưa chạy mô hình học máy/coreset nên không có accuracy, Macro-F1, calibration hoặc lợi ích chi phí được xác nhận trong card này.

Những thông tin còn thiếu được ghi nhận rõ: đầu mối duy trì và giấy phép riêng bộ dẫn xuất; ngày/phiên bản tải dump gốc; timezone gốc; lịch sử hiệu lực metadata; khóa gốc/lần mở; danh sách/encoding đầy đủ TEMPO-DQ; thẩm định proxy/nhãn và sentiment; giao thức thời gian nghiêm ngặt; Q_c/Y_c trên đúng quần thể và các đánh giá hai cấp theo Review. Không điền các mục này bằng số liệu giả định.

Các nguồn số liệu cục bộ và checksum đầy đủ nằm trong [dataset_statistics.json](data_card/dataset_statistics.json), [schema_inventory.json](data_card/schema_inventory.json), [source_validations.json](data_card/source_validations.json) và [run_manifest.json](data_card/run_manifest.json). Card mô tả snapshot ngày chốt; khi thay nguồn, alpha, cohort, split, feature hoặc encoder cần tạo phiên bản card mới.
''')
    import re
    return re.sub(r'\n{3,}', '\n\n', '\n\n'.join(pieces).strip())+'\n'


def validate_card(base,card,s,inventory,source_hashes):
    import re
    inv={r['name']:r for r in inventory};cfg=s['phase_feature_contract'];phase=s['summaries']['phase_features']
    available={r['name'] for r in inv['phase/model_ready/P1']['columns']}
    local_links=[]
    for target in re.findall(r'\]\(([^)]+)\)',card):
        if target.startswith(('https://','http://','#')):continue
        local_links.append(target.split('#')[0])
    checks={
        'six_requested_sections_present':all(f'## {i}.' in card for i in range(1,7)),
        'every_local_link_resolves':all((base/target).exists() for target in local_links if target not in ['data_card/run_manifest.json']),
        'all_55_features_described_with_real_types':len(cfg['model_feature_columns'])==55 and all('`'+name+'`' in card and name in available for name in cfg['model_feature_columns']),
        'all_inventory_files_have_schema':all(r['n_columns']==len(r['columns']) and r['n_files']>0 for r in inventory),
        'master_and_targets_row_counts_match':inv['quality/user_course_final']['n_rows']==inv['phase/user_course_targets']['n_rows']==s['full_targets']['n_pairs'],
        'all_phases_have_same_full_cohort':all(inv[f'phase/features/{p}']['n_rows']==s['full_targets']['n_pairs'] for p in ['P1','P2','P3','P4']),
        'all_supervised_phases_have_same_labeled_cohort':all(inv[f'phase/model_ready/{p}']['n_rows']==s['labeled_targets']['n_pairs'] for p in ['P1','P2','P3','P4']),
        'time_statuses_partition_source_pairs':sum(r['n_pairs'] for r in s['summaries']['review_time_phases']['eligibility'])==s['summaries']['review_time_phases']['statistics']['n_input_pairs'],
        'class_counts_partition_labeled_pairs':sum(r['n_weighted'] for r in s['summaries']['weighted_loqv_alpha_0p1']['class_distribution'])==s['labeled_targets']['n_pairs'],
        'all_current_phase_data_checks_pass':all(v for k,v in phase['validation'].items() if k!='fixtures') and all(phase['validation']['fixtures'].values()),
        'source_artifacts_unchanged_while_building_card':all(sha256(base/path)==value for path,value in source_hashes.items()),
    }
    return checks


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root');parser.add_argument('--as-of',default=AS_OF)
    args=parser.parse_args()
    base=Path(args.root or (Path.cwd()/'working_13-10' if (Path.cwd()/'working_13-10').is_dir() else Path.cwd())).resolve()
    evidence=base/'data_card';evidence.mkdir(exist_ok=True)
    stats,inventory,source_hashes,validations=collect(base)
    card=render_card(stats,inventory,args.as_of)
    card_path=base/'DATA_CARD.md';card_path.write_text(card,encoding='utf-8')
    write_json(evidence/'dataset_statistics.json',stats)
    write_json(evidence/'schema_inventory.json',inventory)
    write_json(evidence/'source_validations.json',validations)
    write_json(evidence/'source_hashes.json',source_hashes)
    checks=validate_card(base,card,stats,inventory,source_hashes)
    write_json(evidence/'validation.json',checks)
    assert all(checks.values()),checks
    manifest={'dataset_id':DATASET_ID,'card_version':'1.0','as_of':args.as_of,'date_timezone':'Asia/Ho_Chi_Minh',
        'engine_source_sha256':sha256(__file__),'card_sha256':sha256(card_path),
        'sources_sha256':source_hashes,'inventory_scope':'Persisted Parquet tables in raw/model/EDA/time/quality/sentiment/weighted/phase outputs; excludes _stage and _scratch caches.',
        'source_data_mutation':'none','external_verification_date':AS_OF,
        'external_sources':['https://github.com/THU-KEG/MOOCCubeX',
          'https://github.com/THU-KEG/MOOCCubeX/blob/main/docs/user-en.md',
          'https://github.com/THU-KEG/MOOCCubeX/blob/main/LICENSE',
          'https://lfs.aminer.cn/misc/moocdata/publications/mooccubex.pdf',
          'https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest']}
    write_json(evidence/'run_manifest.json',manifest)
    write_json(evidence/'_SUCCESS.json',{'manifest_sha256':sha256(evidence/'run_manifest.json'),'card_sha256':sha256(card_path)})
    print('DATA_CARD_COMPLETE',card_path,len(inventory),'tables',len(card.split()),'words',checks)


if __name__=='__main__':
    main()
