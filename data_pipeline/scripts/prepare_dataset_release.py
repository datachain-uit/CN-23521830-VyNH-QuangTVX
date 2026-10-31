"""Materialize the four approved datasets without modifying local symlink targets."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

DATASETS = ('input/moocubex_model', 'final_quality_v4', 'phase_features_v4', 'phase_features_v4_2')
SKIP_PARTS = {'_stage', '_scratch', '__pycache__', '.pytest_cache', '.ipynb_checkpoints'}
SOURCE_FILES = [
    '01_moocubex_raw_to_parquet.ipynb', '02_build_dimensions_bridges.ipynb',
    '04_eda_engagement_features_comparison.py', '05_build_review_time_phases.py',
    '06_build_final_quality_tables.py', '07_compute_afelo_sentiment.py',
    '08_build_weighted_loqv.py', '09_build_phase_features.py',
    '15_build_final_quality_v2.py', '16_build_phase_features_v2.py',
    '18_build_final_quality_v3.py', '19_build_phase_features_v3.py', '20_build_fullA.py',
    '21_build_final_quality_v4.py', '22_build_phase_features_v4.py', '23_finalize_data_card_v4.py',
    '25_build_phase_features_v4_2.py', '26_verify_phase_features_v4_2.py',
    '27_build_test_features_v4_2.py', '28_build_data_card_v4_2.py', '29_build_teacher_subgroup_v4_2.py',
    'cq_v4_common.py', 'cq_tempo_features.py', 'requirements-v4.txt', 'requirements-v4_2.txt',
    'afelo_sentiment_cardiff/requirements-sentiment.txt', 'test_cq_tempo_features.py',
    'tests/test_cq_v4_2_builder.py', 'tests/test_cq_v4_protocol.py', 'tests/test_phase_feature_cutoff.py',
    'scripts/clear_notebook_outputs.py', 'DATA_CARD_V4.md', 'DATA_CARD_V4_2.md',
    'BANG_GIAI_TRINH_CONG_THUC_FEATURE_VA_NHAN_CQ_V4_2026_10_23.md',
]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def resolve_dataset(source_root, links, name):
    local = source_root/name
    matches = [r for r in links['links'] if 'data_pipeline/'+name == r['path'] or
               ('data_pipeline/'+name).startswith(r['path']+'/')]
    if matches:
        record = max(matches, key=lambda r:len(r['path']))
        suffix = ('data_pipeline/'+name)[len(record['path']):].lstrip('/')
        mapped = (Path(record['source'])/suffix).resolve()
        if local.resolve() != mapped:
            raise ValueError('Symlink disagrees with .local_data_links.json: '+name)
    else:
        mapped = local.resolve()
    if not mapped.is_dir():
        raise FileNotFoundError(name)
    return mapped


def release_files(source):
    expected = {}
    index = source/'SHA256SUMS.txt'
    if index.exists():
        for line in index.read_text().splitlines():
            digest, name = line.split('  ', 1)
            q = Path(name)
            if q.is_absolute() or '..' in q.parts:
                raise ValueError('Invalid checksum path')
            expected[name] = digest
    selected = []
    for p in sorted(source.rglob('*')):
        if p.is_symlink():
            raise ValueError('Nested symlink must be resolved explicitly: '+str(p.relative_to(source)))
        if not p.is_file():
            continue
        rel = p.relative_to(source).as_posix()
        transient = bool(set(p.relative_to(source).parts)&SKIP_PARTS) or p.suffix in {'.pyc','.log'} or p.name.endswith('.partial') or p.name=='.DS_Store'
        # Historical test logs and one loader bytecode file are indexed in the
        # frozen bundles. Preserve indexed artifacts so their signatures remain valid.
        if not transient or rel in expected:
            selected.append((rel, p, transient))
    if not set(expected) <= {r for r,_,_ in selected}:
        raise FileNotFoundError('An indexed artifact is missing from the source')
    return selected, expected


def clone_or_copy(source, target, expected=None):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if not target.is_symlink() and target.is_file() and sha256(target)==(expected or sha256(source)):
            return 'reused'
        raise FileExistsError(target)
    if os.uname().sysname == 'Darwin':
        copied = subprocess.run(['cp','-c','-p',str(source),str(target)],capture_output=True).returncode==0
        if copied:
            return 'apfs_clone'
    shutil.copy2(source, target)
    return 'copy'


def source_closure(root):
    selected = set(SOURCE_FILES)
    pending = list(selected)
    while pending:
        name = pending.pop()
        if not name.endswith('.py'):
            continue
        for node in ast.walk(ast.parse((root/name).read_text())):
            if isinstance(node,ast.Constant) and isinstance(node.value,str):
                candidate = node.value
                if candidate.endswith('.py') and Path(candidate).name==candidate and (root/candidate).is_file() and candidate not in selected:
                    selected.add(candidate)
                    pending.append(candidate)
    return sorted(selected)


def prepare(source_root, links_path, output_root):
    source_root, output_root = Path(source_root).resolve(), Path(output_root).resolve()
    if source_root==output_root or source_root in output_root.parents or output_root in source_root.parents:
        raise ValueError('Use a separate clean checkout for the release')
    links = json.loads(Path(links_path).read_text())
    records, summaries = {}, {}
    for name in DATASETS:
        source = resolve_dataset(source_root, links, name)
        selected, expected = release_files(source)
        for rel,p,indexed_transient in selected:
            before = p.stat()
            digest = sha256(p)
            if rel in expected and digest!=expected[rel]:
                raise AssertionError('Frozen source checksum mismatch: '+name+'/'+rel)
            target = output_root/name/rel
            method = clone_or_copy(p,target,digest)
            after = p.stat()
            if sha256(target)!=digest or (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
                raise AssertionError('Source/copy changed: '+name+'/'+rel)
            records[name+'/'+rel] = {'sha256':digest,'bytes':before.st_size,
                'lfs':p.suffix=='.parquet','dataset':name,
                'preserved_indexed_history':indexed_transient}
        summaries[name] = {'files':len(selected),'bytes':sum(records[name+'/'+r]['bytes'] for r,_,_ in selected),
                           'source_checksum_entries_verified':len(expected)}
        print(name, summaries[name], flush=True)
    for name in source_closure(source_root):
        target = output_root/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source_root/name,target)
    generated = ['README.md','REPRODUCING.md','.gitignore','.gitattributes',
                 'scripts/prepare_dataset_release.py','scripts/check_github_payload.py',
                 'tests/test_dataset_publication.py']
    code_files = sorted(set(source_closure(source_root)+generated))
    manifest = {'schema_version':'dataset-publication-v1','publication_date':'2026-10-31',
        'source_code_commit':'7c5477410b9afb51894eeca4d45a27b63c85a82d',
        'source_link_map_sha256':sha256(links_path),'datasets':summaries,'files':records,
        'code_files':code_files,'source_data_modified':False,
        'policy':'Four approved datasets materialized as regular files; Parquet uses Git LFS; indexed historical artifacts retain their original bytes'}
    output_root.mkdir(parents=True,exist_ok=True)
    (output_root/'PUBLICATION_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    rules = ['# Curated source and exact artifacts in PUBLICATION_MANIFEST.json.','*','!*/','!*.py','!*.ipynb',
             '!requirements*.txt','!.gitignore','!/.gitattributes','!/README.md','!/REPRODUCING.md',
             '!/PUBLICATION_MANIFEST.json','!/DATA_CARD_V4.md','!/DATA_CARD_V4_2.md',
             '!/BANG_GIAI_TRINH_CONG_THUC_FEATURE_VA_NHAN_CQ_V4_2026_10_23.md',
             '# Exact dataset allowlist; new runtime outputs stay ignored.']
    rules += ['!/'+name for name in sorted(records)]
    rules += ['.env','.env.*','.local_data_links.json','.DS_Store','.aws/','.ssh/','.venv/',
              '.pytest_cache/','.ipynb_checkpoints/','*.partial']
    (output_root/'.gitignore').write_text('\n'.join(rules)+'\n')
    print('Materialized',len(records),'artifacts;',round(sum(r['bytes'] for r in records.values())/1024**3,3),'GiB',flush=True)
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--links',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    args=parser.parse_args()
    prepare(args.source_root,args.links,args.output_root)
