"""Run pinned Cardiff sentiment locally, then finalize persisted quality tables.

No video/problem rescan, no fitting on dev/test, no hosted comment inference.
Each prediction is checkpointed by unique preprocessed text, then joined back
to every accepted comment_id. Re-running resumes the same pinned revision.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import time

os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import duckdb

MODEL_ID = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
PREPROCESS_VERSION = 'cardiff-model-card-mentions-http-v1'
MAX_LENGTH = 512


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump_json(path, value):
    p = Path(path)
    temporary = p.with_suffix(p.suffix + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    temporary.replace(p)


def sql_string(value):
    return "'" + str(value).replace("'", "''") + "'"


def preprocess(text):
    # Exactly the preprocessing published on the selected model card.
    words = []
    for word in text.split(' '):
        word = '@user' if word.startswith('@') and len(word) > 1 else word
        word = 'http' if word.startswith('http') else word
        words.append(word)
    return ' '.join(words)


def resolve_model(output_root, cache_dir, allow_download=True):
    from huggingface_hub import HfApi, snapshot_download
    out = Path(output_root)
    out.mkdir(parents=True, exist_ok=True)
    lock_path = out / 'model_lock.json'
    if lock_path.exists():
        lock = json.loads(lock_path.read_text())
        assert lock['model_id'] == MODEL_ID
    else:
        if not allow_download:
            raise RuntimeError('Model has not been downloaded; run once with allow_download=True.')
        info = HfApi().model_info(MODEL_ID)
        names = {s.rfilename for s in info.siblings}
        weight = 'model.safetensors' if 'model.safetensors' in names else 'pytorch_model.bin'
        wanted = [x for x in ['config.json', 'tokenizer.json', 'tokenizer_config.json',
            'special_tokens_map.json', 'vocab.json', 'merges.txt', weight, 'README.md'] if x in names]
        lock = {'model_id': MODEL_ID, 'revision': info.sha, 'files': wanted}
    path = Path(lock.get('snapshot_path', ''))
    if not path.is_dir() or not all((path / f).is_file() for f in lock['files']):
        path = Path(snapshot_download(MODEL_ID, revision=lock['revision'],
            allow_patterns=lock['files'], cache_dir=str(cache_dir), local_files_only=not allow_download))
    lock['snapshot_path'] = str(path.resolve())
    lock['file_sha256'] = {f: file_hash(path / f) for f in lock['files']}
    dump_json(lock_path, lock)
    return lock


def prepare_comments(con, source, out):
    stage = out / '_stage'
    stage.mkdir(exist_ok=True)
    signature = file_hash(source)
    checkpoint = stage / 'preprocessing.json'
    paths = [stage / 'unique_texts.parquet', stage / 'comment_text_keys.parquet']
    expected = {'source_sha256': signature, 'preprocess_version': PREPROCESS_VERSION}
    if checkpoint.exists() and json.loads(checkpoint.read_text()) == expected and all(p.exists() for p in paths):
        return paths, signature
    rows = con.execute(f'SELECT comment_id,text FROM read_parquet({sql_string(source)}) ORDER BY comment_id').fetchall()
    assert len(rows) == len({r[0] for r in rows}), 'Input comment IDs must be unique'
    unique = {}
    key_json = stage / 'comment_text_keys.jsonl'
    text_json = stage / 'unique_texts.jsonl'
    with key_json.open('w', encoding='utf-8') as keys:
        for comment_id, text in rows:
            assert isinstance(text, str) and text.strip(), 'Input must contain filtered nonempty comments'
            processed = preprocess(text)
            digest = hashlib.sha256(processed.encode('utf-8')).hexdigest()
            if digest in unique:
                assert unique[digest] == processed, 'Text hash collision'
            unique[digest] = processed
            keys.write(json.dumps({'comment_id': comment_id, 'text_hash': digest}) + '\n')
    with text_json.open('w', encoding='utf-8') as texts:
        for digest, text in sorted(unique.items()):
            texts.write(json.dumps({'text_hash': digest, 'model_text': text,
                'has_cjk': bool(re.search(r'[\u3400-\u9fff]', text))}, ensure_ascii=False) + '\n')
    for json_path, parquet_path in zip([text_json, key_json], paths):
        con.execute(f'COPY (SELECT * FROM read_json_auto({sql_string(json_path)})) TO {sql_string(parquet_path)} (FORMAT PARQUET, COMPRESSION ZSTD)')
        json_path.unlink()
    dump_json(checkpoint, expected)
    print(f'Prepared {len(rows):,} comments; {len(unique):,} unique model texts', flush=True)
    return paths, signature


def infer_unique(con, unique_path, out, lock, source_hash, device='auto', threads=4,
                 batch_size=32, token_budget=2048, checkpoint_size=1024):
    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    model_path = lock['snapshot_path']
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    labels = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    assert labels == {0: 'negative', 1: 'neutral', 2: 'positive'}, labels
    model.eval().to(device)
    rows = con.execute(f'SELECT text_hash,model_text,has_cjk FROM read_parquet({sql_string(unique_path)}) ORDER BY text_hash').fetchall()
    texts = [r[1] for r in rows]
    encoded = tokenizer(texts, truncation=False, padding=False, verbose=False)['input_ids']
    items = []
    for row, ids in zip(rows, encoded):
        n_tokens = len(ids)
        if n_tokens > MAX_LENGTH:
            ids = tokenizer(row[1], truncation=True, max_length=MAX_LENGTH)['input_ids']
        items.append({'text_hash': row[0], 'has_cjk': row[2], 'n_tokens': n_tokens, 'ids': ids})
    items.sort(key=lambda r: (len(r['ids']), r['text_hash']))
    signature_payload = {'source_sha256': source_hash, 'model_revision': lock['revision'],
        'preprocess_version': PREPROCESS_VERSION, 'max_length': MAX_LENGTH,
        'torch': torch.__version__, 'transformers': transformers.__version__, 'device': device,
        'batch_size': batch_size, 'token_budget': token_budget, 'checkpoint_size': checkpoint_size}
    signature = hashlib.sha256(json.dumps(signature_payload, sort_keys=True).encode()).hexdigest()
    predictions = out / 'text_predictions'
    predictions.mkdir(exist_ok=True)
    info_path = predictions / 'inference_config.json'
    if info_path.exists():
        assert json.loads(info_path.read_text())['signature'] == signature, 'Inference config changed; use a new output directory'
    dump_json(info_path, dict(signature_payload, signature=signature, labels=labels, dtype='float32'))
    started = time.monotonic()
    n_parts = math.ceil(len(items) / checkpoint_size)
    for part in range(n_parts):
        path = predictions / f'part-{part:05d}.parquet'
        chunk = items[part * checkpoint_size:(part + 1) * checkpoint_size]
        if path.exists():
            assert con.execute(f'SELECT count(*) FROM read_parquet({sql_string(path)})').fetchone()[0] == len(chunk)
            print(f'Sentiment checkpoint {part + 1}/{n_parts} reused', flush=True)
            continue
        result = []
        pos = 0
        tick = time.monotonic()
        while pos < len(chunk):
            length = len(chunk[min(pos + batch_size, len(chunk)) - 1]['ids'])
            n = max(1, min(batch_size, token_budget // max(length, 1)))
            batch = chunk[pos:pos + n]
            inputs = tokenizer.pad({'input_ids': [r['ids'] for r in batch]}, padding=True, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.inference_mode():
                probabilities = torch.softmax(model(**inputs).logits.float(), dim=-1).cpu().double().tolist()
            for item, values in zip(batch, probabilities):
                # Normalize float32 rounding into a double probability simplex.
                total = sum(values)
                negative, neutral, positive = [v / total for v in values]
                p = [negative, neutral, positive]
                result.append({'text_hash': item['text_hash'], 'p_neg': negative, 'p_neu': neutral,
                    'p_pos': positive, 'predicted_label': labels[max(range(3), key=lambda i: p[i])],
                    'confidence': max(p), 'entropy': -sum(v * math.log(v) for v in p if v > 0),
                    'n_tokens': item['n_tokens'], 'n_tokens_used': len(item['ids']),
                    'was_truncated': item['n_tokens'] > MAX_LENGTH, 'has_cjk': item['has_cjk']})
            pos += len(batch)
        temporary = path.with_suffix('.jsonl')
        with temporary.open('w') as f:
            for record in result:
                f.write(json.dumps(record) + '\n')
        partial = path.with_suffix('.parquet.partial')
        con.execute(f'COPY (SELECT * FROM read_json_auto({sql_string(temporary)})) TO {sql_string(partial)} (FORMAT PARQUET,COMPRESSION ZSTD)')
        partial.replace(path)
        temporary.unlink()
        print(f'Sentiment {min((part+1)*checkpoint_size,len(items)):,}/{len(items):,}; '
              f'checkpoint {part+1}/{n_parts}, {time.monotonic()-tick:.1f}s, elapsed {time.monotonic()-started:.1f}s', flush=True)
    return predictions, dict(signature_payload, signature=signature, dtype='float32', labels=labels)


def load_finalizer(path):
    if Path(path).is_file():
        spec = importlib.util.spec_from_file_location('quality_finalizer', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.finalize_quality, file_hash(path)
    if 'FINALIZER_SOURCE' in globals():
        source = globals()['FINALIZER_SOURCE']
        namespace = {'__name__': 'quality_finalizer', 'ENGINE_SOURCE_SHA256': hashlib.sha256(source.encode()).hexdigest()}
        exec(compile(source, 'embedded_quality_finalizer.py', 'exec'), namespace)
        return namespace['finalize_quality'], namespace['ENGINE_SOURCE_SHA256']
    raise FileNotFoundError(path)


def run_sentiment(quality_root=None, output_root=None, cache_dir=None, engine_path=None,
                  allow_download=True, device='auto', threads=4, finalize=True):
    default_base = Path.cwd() / 'working_13-10' if (Path.cwd() / 'working_13-10').is_dir() else Path.cwd()
    quality = Path(quality_root or default_base / 'final_quality_tables').resolve()
    out = Path(output_root or quality.parent / 'afelo_sentiment_cardiff').resolve()
    cache = Path(cache_dir or os.environ.get('HF_HOME', str(Path.home() / '.cache/huggingface/hub')))
    source = quality / 'tables/afelo_comment_links.parquet'
    assert source.is_file(), source
    out.mkdir(parents=True, exist_ok=True)
    for folder in [out / 'tables', out / '_scratch']:
        folder.mkdir(exist_ok=True)
    success = out / '_SUCCESS.json'
    if success.exists():
        success.unlink()
    lock = resolve_model(out, cache, allow_download)
    con = duckdb.connect()
    con.execute("SET memory_limit='2GB'")
    con.execute('SET threads=2')
    con.execute(f'SET temp_directory={sql_string(out / "_scratch")}')
    (unique, keys), source_hash = prepare_comments(con, source, out)
    predictions, inference = infer_unique(con, unique, out, lock, source_hash, device=device, threads=threads)
    for name, path in [('source_comments', source), ('comment_keys', keys), ('text_predictions', predictions / '*.parquet')]:
        con.execute(f'CREATE VIEW {name} AS SELECT * FROM read_parquet({sql_string(path)})')
    result_path = out / 'tables/comment_sentiment.parquet'
    partial = result_path.with_suffix('.parquet.partial')
    con.execute(f'''COPY (SELECT c.comment_id,c.user_id,c.course_id,c.lo_key,c.event_time,
        p.*,{sql_string(MODEL_ID)} model_id,{sql_string(lock['revision'])} model_revision
        FROM source_comments c LEFT JOIN comment_keys k USING(comment_id)
        LEFT JOIN text_predictions p USING(text_hash)) TO {sql_string(partial)} (FORMAT PARQUET,COMPRESSION ZSTD)''')
    partial.replace(result_path)
    con.execute(f'CREATE VIEW sentiment AS SELECT * FROM read_parquet({sql_string(result_path)})')
    checks = {
        'all_source_comments_preserved': con.execute('SELECT (SELECT count(*) FROM source_comments)=(SELECT count(*) FROM sentiment)').fetchone()[0],
        'unique_comment_ids': con.execute('SELECT count(*)=count(DISTINCT comment_id) FROM sentiment').fetchone()[0],
        'no_missing_predictions': con.execute('SELECT count(*)=0 FROM sentiment WHERE p_pos IS NULL OR p_neu IS NULL OR p_neg IS NULL OR text_hash IS NULL').fetchone()[0],
        'valid_probabilities': con.execute('''SELECT count(*)=0 FROM sentiment WHERE NOT isfinite(p_pos) OR NOT isfinite(p_neu) OR NOT isfinite(p_neg)
            OR p_pos<0 OR p_neu<0 OR p_neg<0 OR p_pos>1 OR p_neu>1 OR p_neg>1 OR abs(p_pos+p_neu+p_neg-1)>1e-6''').fetchone()[0],
        'events_inside_user_final_window': con.execute('SELECT count(*)=0 FROM source_comments WHERE event_time<enroll_time OR event_time>course_end').fetchone()[0],
        'prediction_keys_match_source': con.execute('SELECT count(*)=0 FROM source_comments ANTI JOIN sentiment USING(comment_id)').fetchone()[0],
        'preprocessing_fixture': preprocess('@alice hello https://example.com') == '@user hello http',
    }
    assert all(checks.values()), checks
    def records(sql):
        cursor = con.execute(sql)
        names = [x[0] for x in cursor.description]
        return [dict(zip(names, row)) for row in cursor.fetchall()]
    summary = {'comments': con.execute('SELECT count(*) FROM sentiment').fetchone()[0],
        'unique_model_texts': con.execute('SELECT count(*) FROM text_predictions').fetchone()[0],
        'truncated_comments': con.execute('SELECT count(*) FROM sentiment WHERE was_truncated').fetchone()[0],
        'comments_with_cjk': con.execute('SELECT count(*) FROM sentiment WHERE has_cjk').fetchone()[0],
        'label_distribution': records('SELECT predicted_label,count(*) n_comments,avg(confidence) mean_confidence FROM sentiment GROUP BY 1 ORDER BY 1'),
        'checks': checks}
    audit_path = out / 'tables/manual_audit_sample.parquet'
    con.execute(f'''COPY (SELECT * EXCLUDE(sample_rank) FROM (
        SELECT c.text,s.*,row_number() OVER(PARTITION BY predicted_label ORDER BY hash(comment_id)) sample_rank
        FROM sentiment s JOIN source_comments c USING(comment_id)) WHERE sample_rank<=30)
        TO {sql_string(audit_path)} (FORMAT PARQUET,COMPRESSION ZSTD)''')
    con.close()
    import importlib.metadata
    packages = {name: importlib.metadata.version(name) for name in ['torch','transformers','tokenizers','huggingface-hub','duckdb','numpy']}
    manifest = {'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'source': str(source),
        'source_sha256': source_hash, 'model': lock, 'inference': inference, 'packages': packages,
        'sentiment_path': str(result_path), 'sentiment_sha256': file_hash(result_path),
        'script_sha256': globals().get('SENTIMENT_SOURCE_SHA256') or file_hash(__file__),
        'preprocessing': PREPROCESS_VERSION, 'training': 'Frozen pretrained model, no fitting or threshold tuning on dataset',
        'limitations': ['English Twitter sentiment model on translated educational comments; no domain accuracy claim without annotated evaluation.',
            'Mixed-language/CJK comments are retained and flagged; sentiment remains a proxy.',
            'Long comments use the first 512 tokens including special tokens; truncation is flagged.',
            'Source filter removes empty/exact duplicate comments; it does not identify all off-topic/UI/check-in comments.',
            'No-comment chapters keep sentiment null; AFELO can still use observed social participation components.']}
    if finalize:
        backup = out / 'pre_sentiment_state'
        backup.mkdir(exist_ok=True)
        for name in ['run_manifest.json','summary.json','validation.json','_SUCCESS.json','bao_cao_final_quality.md']:
            p = quality / name
            if p.exists() and not (backup / name).exists():
                (backup / name).write_bytes(p.read_bytes())
        finalizer, engine_hash = load_finalizer(engine_path or quality.parent / '06_build_final_quality_tables.py')
        summary['quality'] = finalizer(quality, sentiment_path=result_path, afelo_mode='sentiment')
        manifest['finalizer_source_sha256'] = engine_hash
        quality_manifest = json.loads((quality / 'run_manifest.json').read_text())
        quality_manifest['finalization']['sentiment_model'] = {'model_id': MODEL_ID, 'revision': lock['revision'],
            'sentiment_manifest': str(out / 'run_manifest.json')}
        dump_json(quality / 'run_manifest.json', quality_manifest)
        dump_json(quality / '_SUCCESS.json', {'afelo_mode': 'sentiment', 'manifest_sha256': file_hash(quality / 'run_manifest.json')})
    dump_json(out / 'summary.json', summary)
    dump_json(out / 'run_manifest.json', manifest)
    dump_json(out / 'validation.json', checks)
    dump_json(success, {'manifest_sha256': file_hash(out / 'run_manifest.json'), 'finalized': finalize})
    write_sentiment_report(out, quality, summary, lock)
    print('SENTIMENT_COMPLETE', json.dumps(summary, ensure_ascii=False, default=str), flush=True)
    return summary


def write_sentiment_report(out, quality, summary, lock):
    fmt = lambda n: f'{n:,}'.replace(',', '.')
    lines = ['# AFELO với Cardiff sentiment', '',
        f'Model: [{MODEL_ID}](https://huggingface.co/{MODEL_ID}), revision `{lock["revision"]}`. '
        'Chạy tại máy, frozen weights, không huấn luyện hoặc fit tham số trên comment của dataset.', '',
        f'- Đã suy luận và ghép lại đủ **{fmt(summary["comments"])} comment** từ '
        f'**{fmt(summary["unique_model_texts"])} nội dung sau tiền xử lý**.',
        f'- {fmt(summary["truncated_comments"])} comment dài hơn 512 token: lấy phần đầu và gắn cờ `was_truncated`.',
        f'- {fmt(summary["comments_with_cjk"])} comment có ký tự CJK: giữ và gắn cờ `has_cjk`.',
        '- Đầu vào đã lọc theo `[enroll_time, course_end]`, loại text rỗng và bản ghi trùng chính xác. '
        'Đây chưa phải bộ lọc ngữ nghĩa cho comment học tập.', '',
        '| Nhãn sentiment dự đoán | Số comment |', '|---|---:|']
    lines += [f'| {r["predicted_label"]} | {fmt(r["n_comments"])} |' for r in summary['label_distribution']]
    lines += ['', '## Công thức và dữ liệu thiếu', '',
        '`sentiment_valence = ((sum(p_pos) - sum(p_neg)) / n_sentiment_comments + 1) / 2`.', '',
        '`AFELO_proxy = mean_non_null(comment_participation, reply_participation, response_received_rate, sentiment_valence)`.', '',
        'Không có comment thì sentiment và response rate là null. Các thành phần tham gia xã hội còn lại vẫn được tính '
        'theo quy ước notebook 06; AFELO vì thế không đồng nghĩa chỉ với cảm xúc dự đoán. '
        'Chương hoàn toàn chưa quan sát không có dòng chapter và không được tự gán chất lượng kém.', '',
        'Model học từ tweet tiếng Anh; comment dịch trong giáo dục có khác biệt miền dữ liệu. Xác suất/confidence '
        'không phải độ chính xác đã kiểm định. `manual_audit_sample.parquet` chứa 30 mẫu mỗi nhãn để kiểm tra thủ công. '
        'AFELO vẫn là proxy do nguồn không có like/dislike/rating gốc.', '']
    if 'quality' in summary:
        q = summary['quality']
        lines += ['## Bảng cuối kỳ', '',
            f'Giữ đủ **{fmt(q["user_course_rows"])} user–course** và **{fmt(q["chapter_rows"])} chương quan sát**. '
            'Các bảng được cập nhật bằng LEFT JOIN; không bỏ dòng do thiếu chỉ số.', '',
            '| Trạng thái | Số user–course |', '|---|---:|']
        lines += [f'| {r["label_status"]} | {fmt(r["n_rows"])} |' for r in q['label_status']]
        lines += ['', '| Split | Lớp | Số user–course |', '|---|---|---:|']
        lines += [f'| {r["split"]} | {r["y"]} | {fmt(r["n_rows"])} |' for r in q['class_distribution']]
        lines += ['', 'q chỉ có khi tất cả chương quan sát có đủ ba chỉ số; q là trung bình LOQV từng chương. '
            'Các cặp chưa quan sát hoặc thiếu COELO/ACELO vẫn giữ q/y null.', '',
            'Bảng cuối: `../final_quality_tables/tables/user_course_final/*.parquet` (32 file, một bảng logic). '
            'Ba bảng thành phần và chapter_quality trong cùng thư mục tables đã được cập nhật.', '',
            'Các bảng cuối kỳ phục vụ nhãn/kiểm toán. Feature P1–P4 phải tính lại đến đúng cutoff; '
            'không đưa feature cuối kỳ hoặc chỉ số tạo nhãn vào X. Split course hiện vẫn sơ bộ và còn chồng thời gian '
            'như báo cáo notebook 06; chạy sentiment không giải quyết giới hạn này.', '']
    lines += ['## Chạy lại', '',
        'Notebook 07 / script 07 tiếp tục từ checkpoint và pin revision trong model_lock.json. '
        'Notebook 06 dựng dữ liệu nền ở mode deferred; nếu chạy lại notebook 06, cần chạy tiếp notebook 07 '
        'để điền lại sentiment và nhãn. Không chạy lại dữ liệu thô chỉ để đổi sentiment.', '']
    (out / 'bao_cao_afelo_sentiment.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quality-root')
    parser.add_argument('--output-root')
    parser.add_argument('--cache-dir')
    parser.add_argument('--engine-path')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--no-finalize', action='store_true')
    args = parser.parse_args()
    run_sentiment(args.quality_root, args.output_root, args.cache_dir, args.engine_path,
        allow_download=not args.offline, device=args.device, threads=args.threads, finalize=not args.no_finalize)
