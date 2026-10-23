"""Summarize elapsed days between saved cutoffs, one row per enrollment.

Read-only on the dataset; write the JSON and Markdown report beside this file.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import duckdb

OUT = Path(__file__).resolve().parent
BASE = OUT.parents[1]
OUT.mkdir(parents=True, exist_ok=True)
con = duckdb.connect()
con.execute('SET threads=2')
con.execute("SET memory_limit='2GB'")
con.execute("SET TimeZone='UTC'")
con.execute("SET temp_directory='"+str(OUT/'_scratch').replace("'","''")+"'")


def qs(x):
    return "'"+str(x).replace("'","''")+"'"


def rows(sql):
    cur=con.execute(sql)
    return [dict(zip([d[0] for d in cur.description],r)) for r in cur.fetchall()]


con.execute(f'''CREATE VIEW times AS SELECT * FROM read_parquet(
    {qs(BASE/'review_time_phases/tables/user_course_phase_times.parquet')}) WHERE time_eligible''')
con.execute(f'''CREATE TABLE labels AS SELECT user_id,course_id,split FROM read_parquet(
    {qs(BASE/'phase_features/tables/user_course_targets.parquet')}) WHERE y IS NOT NULL''')
con.execute('CREATE VIEW labeled AS SELECT t.*,l.split FROM times t JOIN labels l USING(user_id,course_id)')
gaps=[('P1→P2','cutoff_P1','cutoff_P2',0.25),
      ('P2→P3','cutoff_P2','cutoff_P3',0.25),
      ('P3→P4','cutoff_P3','cutoff_P4',0.15),
      ('P1→P4','cutoff_P1','cutoff_P4',0.65)]
result={'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
    'client_date':'2026-10-19','unit':'elapsed days, 86400 seconds per day; not calendar-date differences',
    'weighting':'one row per unique user_id/course_id enrollment; not equal course weights',
    'duckdb_version':duckdb.__version__,
    'formula':'delta_days = (epoch_us(cutoff_to)-epoch_us(cutoff_from))/86400000000',
    'quantiles':'exact quantile_cont',
    'statistics':[],'checks':{}}
for scope,source in [('all_time_eligible','times'),('labeled','labeled'),
                     ('labeled_train',"(SELECT * FROM labeled WHERE split='train')"),
                     ('labeled_dev',"(SELECT * FROM labeled WHERE split='dev')"),
                     ('labeled_test',"(SELECT * FROM labeled WHERE split='test')")]:
    columns=[]
    for i,(_,a,b,_) in enumerate(gaps):
        x=f'(epoch_us({b})-epoch_us({a}))::DOUBLE/86400000000.0'
        columns.extend([f'min({x}) g{i}_min',f'avg({x}) g{i}_mean',
            f'stddev_pop({x}) g{i}_std',f'max({x}) g{i}_max',
            f'quantile_cont({x},[0.01,0.05,0.10,0.25,0.5,0.75,0.95,0.99]) g{i}_quantiles',
            f'count(*) FILTER(WHERE {x}<1) g{i}_lt1day',
            f'count(*) FILTER(WHERE {x}<0) g{i}_negative'])
    data=rows('SELECT count(*) n,'+','.join(columns)+' FROM '+source)[0]
    for i,(name,a,b,fraction) in enumerate(gaps):
        item={'scope':scope,'gap':name,'n':data['n'],'fraction_of_enrollment_to_end':fraction,
            'min':data[f'g{i}_min'],'mean':data[f'g{i}_mean'],'std_population':data[f'g{i}_std'],
            'max':data[f'g{i}_max'],'n_lt1day':data[f'g{i}_lt1day'],
            'n_negative':data[f'g{i}_negative']}
        item.update(dict(zip(['p01','p05','p10','p25','median','p75','p95','p99'],data[f'g{i}_quantiles'])))
        result['statistics'].append(item)
    print(scope,data['n'],flush=True)
result['checks']['no_negative_gaps']=all(x['n_negative']==0 for x in result['statistics'])
result['checks']['ordered_quantiles']=all(
    all(r[a]<=r[b] for a,b in zip(
        ['min','p01','p05','p10','p25','median','p75','p95','p99'],
        ['p01','p05','p10','p25','median','p75','p95','p99','max']))
    for r in result['statistics'])
result['checks']['eligible_count_matches_master']=con.execute(f'''SELECT (SELECT count(*) FROM times)=
    (SELECT count(*) FROM read_parquet({qs(BASE/'phase_features/tables/user_course_targets.parquet')}))''').fetchone()[0]
result['checks']['labeled_count_matches_targets']=con.execute('SELECT (SELECT count(*) FROM labeled)=(SELECT count(*) FROM labels)').fetchone()[0]
result['checks']['adjacent_gaps_sum_to_P1_P4']=con.execute('''SELECT count(*)=0 FROM times
    WHERE (epoch_us(cutoff_P2)-epoch_us(cutoff_P1))+(epoch_us(cutoff_P3)-epoch_us(cutoff_P2))
    +(epoch_us(cutoff_P4)-epoch_us(cutoff_P3))<>(epoch_us(cutoff_P4)-epoch_us(cutoff_P1))''').fetchone()[0]
result['checks']['max_rounding_error_microseconds']=rows('SELECT '+','.join(
    f'max(abs((epoch_us({b})-epoch_us({a}))-(epoch_us(course_end)-epoch_us(enroll_time))*{f})) g{i}'
    for i,(_,a,b,f) in enumerate(gaps))+' FROM times')[0]
result['source_sha256']={p:hashlib.sha256((BASE/p).read_bytes()).hexdigest()
    for p in ['review_time_phases/run_manifest.json','phase_features/run_manifest.json']}
assert result['checks']['no_negative_gaps']
assert result['checks']['ordered_quantiles']
assert result['checks']['eligible_count_matches_master']
assert result['checks']['labeled_count_matches_targets']
assert result['checks']['adjacent_gaps_sum_to_P1_P4']
(OUT/'statistics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')


def fmt(v):
    return f'{v:,.2f}'.replace(',','_').replace('.',',').replace('_','.')


def table(scope):
    head=['| Khoảng | Trung bình | Độ lệch chuẩn | Nhỏ nhất | P25 | Trung vị | P75 | P95 | Lớn nhất |',
          '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in result['statistics']:
        if r['scope']==scope:
            head.append('| '+r['gap']+' | '+' | '.join(
                fmt(r[k]*86400)+' giây' if k=='min' and r[k]<60/86400 else fmt(r[k]) for k in
                ['mean','std_population','min','p25','median','p75','p95','max'])+' |')
    return '\n'.join(head)


def lower_percentiles(scope):
    head=['| Khoảng | P5 (ngày) | P10 (ngày) |',
          '| --- | ---: | ---: |']
    for r in result['statistics']:
        if r['scope']==scope:
            head.append('| '+r['gap']+' | '+fmt(r['p05'])+' | '+fmt(r['p10'])+' |')
    return '\n'.join(head)


text=['# Thống kê số ngày giữa các phase',
    '**Ngày thống kê:** 19/10/2026. **Đơn vị:** ngày thực trôi qua, mỗi ngày = 86.400 giây. Riêng giá trị nhỏ nhất dưới một phút được ghi bằng giây để tránh làm tròn thành 0 ngày.',
    'Mỗi cặp user–course đóng góp một quan sát cho mỗi khoảng. Chênh lệch lấy trực tiếp từ cutoff đã lưu, không chỉ trừ phần ngày lịch và không lấy trung bình đều theo course.',
    '## Toàn bộ cặp đủ điều kiện thời gian',
    '**N = 10.574.107 cặp.**',table('all_time_eligible'),lower_percentiles('all_time_eligible'),
    '## Tập có nhãn dùng cho huấn luyện và đánh giá',
    '**N = 164.429 cặp.**',table('labeled'),lower_percentiles('labeled')]
for scope,label in [('labeled_train','Train — 113.036 cặp'),('labeled_dev','Dev — 27.255 cặp'),('labeled_test','Test — 24.138 cặp')]:
    text.extend(['## '+label,table(scope),lower_percentiles(scope)])
text.extend(['## Diễn giải',
    'Đặt L = course_end − enroll_time. Theo công thức phase, P1→P2 và P2→P3 đều bằng khoảng 0,25L; P3→P4 bằng 0,15L; P1→P4 bằng 0,65L. Sai khác rất nhỏ có thể do làm tròn xuống microsecond.',
    'Khoảng cách phase của mỗi người phụ thuộc thời điểm ghi danh và mốc cuối course, nên không phải mọi người đều cách phase một số ngày cố định. Người ghi danh trước ngày hoạt động đầu của course vẫn được giữ theo chính sách hiện tại, có thể làm khoảng cách dài.',
    'Đây là thống kê mô tả cutoff hồi cứu hiện tại. course_end được suy ra từ hoạt động cuối khóa; các con số này không chứng minh mốc phase đã biết được trong vận hành thời gian thực.',
    'P5 và P10 là các phân vị 5% và 10%: xấp xỉ 5% hoặc 10% số cặp có khoảng cách không lớn hơn giá trị tương ứng; quantile_cont có nội suy. Đây không phải mức trung bình của nhóm 5% hoặc 10% nhỏ nhất.',
    'P01/P05/P10/P99, số khoảng dưới một ngày và kết quả kiểm tra nằm trong [statistics.json](statistics.json). Độ lệch chuẩn trong bảng là stddev_pop; các phân vị tính bằng quantile_cont chính xác.',
    'Mã tái lập: [summarize_phase_gaps.py](summarize_phase_gaps.py). Các bảng nguồn chỉ được đọc, không thay đổi.'])
(OUT/'bao_cao_khoang_cach_phase.md').write_text('\n\n'.join(text)+'\n',encoding='utf-8')
con.close()
print(json.dumps({'statistics':result['statistics'][:8],'checks':result['checks']},ensure_ascii=False,default=str))
