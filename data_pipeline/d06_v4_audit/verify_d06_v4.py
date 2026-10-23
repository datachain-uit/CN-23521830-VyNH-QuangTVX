"""Verify the D06 private dossier independently, without touching source data."""
from __future__ import annotations

import argparse
from collections import defaultdict
import datetime as dt
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def moment(value):
    return dt.datetime.fromisoformat(value) if value is not None else None


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.articles = []
        self.default_checked = 0
        self.remote_resources = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'article':
            self.articles.append(attrs)
        if tag == 'input' and 'checked' in attrs:
            self.default_checked += 1
        if any(attrs.get(k, '').startswith(('http:', 'https:')) for k in ['src','href']):
            self.remote_resources.append(attrs)


def verify(private):
    private = Path(private).resolve()
    root = Path(__file__).resolve().parents[1]
    comparisons = json.loads((private/'comparison.json').read_text())
    by_row = {s['row_id']:s for s in comparisons}
    ledger = [json.loads(line) for line in (private/'events.jsonl').read_text().splitlines()]
    checks = {}
    mismatches = defaultdict(int)
    accepted = defaultdict(list)
    for e in ledger:
        s = by_row[e['row_id']]
        start, end = moment(s['enroll_time']), moment(s['cutoff_time'])
        if e['event_kind'] == 'video':
            epoch = float(e['raw_timestamp']) if e['raw_timestamp'] is not None else None
            raw_time = dt.datetime(1970,1,1)+dt.timedelta(seconds=epoch) if epoch is not None else None
        elif e['event_kind'] == 'comment':
            raw_time = dt.datetime(1970,1,1)+dt.timedelta(milliseconds=int(e['raw_timestamp']))
        else:
            raw_time = moment(e['raw_timestamp'])
        event_time = moment(e['event_time'])
        if raw_time != event_time:
            mismatches['raw_timestamp_conversion'] += 1
        include = (event_time is not None and bool(e['source_window_ok']) and start <= event_time <= end)
        if e['event_kind'] == 'received':
            parent = moment(e['parent_time'])
            include = include and bool(e['parent_comment_valid']) and parent is not None and start <= parent <= end
        if include != e['included']:
            mismatches['inclusion'] += 1
        if include:
            observed = event_time
            if e['event_kind'] == 'video' and e['valid_video']:
                media = min(e['end_point']-e['start_point'],max(0,(end-event_time).total_seconds())*e['speed'])
                observed += dt.timedelta(seconds=media/e['speed'])
            if abs((observed-moment(e['observed_end_time'])).total_seconds())>1e-6:
                mismatches['observed_end'] += 1
            accepted[e['row_id']].append((e['event_kind'],event_time,observed))
    fractions = {'P1':(1,4),'P2':(1,2),'P3':(3,4),'P4':(9,10)}
    for s in comparisons:
        start,end = moment(s['enroll_time']),moment(s['course_end'])
        span = end-start
        microseconds = (span.days*86400+span.seconds)*1000000+span.microseconds
        n,d = fractions[s['phase']]
        expected_cutoff = start+dt.timedelta(microseconds=microseconds*n//d)
        if expected_cutoff != moment(s['cutoff_time']):
            mismatches['rational_cutoff'] += 1
        rows = accepted[s['row_id']]
        count = sum(kind!='received' for kind,_,_ in rows)
        minimum = min((t for _,t,_ in rows),default=None)
        maximum = max((end for _,_,end in rows),default=None)
        if count != s['n_events'] or minimum != moment(s['saved_min']) or maximum != moment(s['saved_max']):
            mismatches['saved_audit'] += 1
    checks['independent_python_recalculation_80'] = not mismatches and len(comparisons)==80
    checks['only_train_dev_rows'] = all(s['split'] in ['train','dev'] for s in comparisons)
    checks['human_status_not_fabricated'] = json.loads((private/'MANIFEST.json').read_text())['human_reviewed_pairs']==0
    failures = []
    hash_count = 0
    for line in (private/'SHA256SUMS.txt').read_text().splitlines():
        expected,relative = line.split('  ',1)
        hash_count += 1
        if sha(private/relative) != expected:
            failures.append(relative)
    checks['private_file_checksums_match'] = not failures
    sources = json.loads((private/'MANIFEST.json').read_text())['source_files']
    checks['all_source_files_unchanged'] = all(sha(root/s['path'])==s['sha256'] for s in sources)
    original = {}
    for line in (root/'phase_features_v4/SHA256SUMS.txt').read_text().splitlines():
        checksum,path = line.split('  ',1)
        original[path] = checksum
    protected = ['audit/cutoff_sample_20_pairs.parquet','audit/cutoff_sample_20_source_comparison.parquet','run_manifest.json']
    checks['original_sample_and_manifest_still_match_delivery_checksums'] = all(
        sha(root/'phase_features_v4'/p)==original[p] for p in protected)
    parser = Page()
    parser.feed((private/'index.html').read_text())
    checks['viewer_has_80_correct_snapshot_forms'] = (
        len(parser.articles)==80 and {a['data-row'] for a in parser.articles}==set(by_row))
    checks['manual_checkboxes_initially_empty'] = parser.default_checked==0
    checks['private_viewer_has_no_remote_resources'] = not parser.remote_resources
    # Inspect exactly the article-to-case hash binding used by the human export.
    checks['viewer_binds_current_case_evidence_hashes'] = all(
        a['data-evidence-hash']==sha(private/'cases'/(a['data-case']+'.json')) for a in parser.articles)
    result = {'scope':'Independent Python D06 dossier verification, not human review',
              'checks':checks,'all_passed':all(checks.values()),'mismatch_counts':dict(mismatches),
              'private_files_hashed':hash_count,'source_files_hashed':len(sources),'human_reviewed_pairs':0,
              'test_feature_or_label_rows_used':0}
    (Path(__file__).parent/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))
    assert result['all_passed'],result


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private-output',default='/private/tmp/d06_cq_v4_2026_10_23')
    args = parser.parse_args()
    verify(args.private_output)
