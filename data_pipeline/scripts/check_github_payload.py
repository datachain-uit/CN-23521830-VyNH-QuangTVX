"""Validate curated source and approved dataset hashes/LFS pointers in Git."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]
DATASETS=('input/moocubex_model','final_quality_v4','phase_features_v4','phase_features_v4_2')
MAX_SOURCE_BYTES=2*1024**2
LFS_POINTER=re.compile(rb'version https://git-lfs.github.com/spec/v1\noid sha256:([a-f0-9]{64})\nsize ([0-9]+)\n')
SYNTHETIC={'test_cq_tempo_features.py':{1,2},'tests/test_cq_v4_2_builder.py':{1,4}}
SECRETS={
    'private-key marker':re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    'AWS access-key pattern':re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
    'GitHub token pattern':re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b'),
    'API-key pattern':re.compile(r'\bsk-(?:proj-)?[A-Za-z0-9_-]{30,}\b')}


def git(*args,input=None):
    return subprocess.check_output(['git',*args],cwd=ROOT,input=input)


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):
            h.update(block)
    return h.hexdigest()


def pointer_problem(payload,entry):
    match=LFS_POINTER.fullmatch(payload)
    if not match:
        return 'Parquet must be a canonical Git LFS pointer in the index'
    if match[1].decode()!=entry['sha256'] or int(match[2])!=entry['bytes']:
        return 'LFS pointer differs from the approved artifact'


def text_problems(name,payload,dataset=False):
    try:
        text=payload.decode('utf-8')
    except UnicodeDecodeError:
        return [] if dataset else ['not UTF-8 source text']
    problems=[label for label,p in SECRETS.items() if p.search(text)]
    if not dataset:
        allowed={f'U_{n}' for n in SYNTHETIC.get(name,set())}
        if set(re.findall(r'\bU_\d+\b',text))-allowed:
            problems.append('literal learner ID in source/documentation')
        if '\0' in text:
            problems.append('binary source content')
    if name.endswith('.ipynb'):
        try:
            n=json.loads(text)
            if any(c.get('outputs') or c.get('execution_count') is not None for c in n['cells'] if c.get('cell_type')=='code'):
                problems.append('notebook has outputs/execution counts')
            if any(c.get('attachments') for c in n['cells']) or n.get('metadata',{}).get('widgets'):
                problems.append('notebook has attachments/widgets')
        except (ValueError,KeyError,TypeError):
            problems.append('invalid notebook')
    return problems


def validate(working_tree=False):
    names=set(git('ls-files','--cached',*(['--others','--exclude-standard'] if working_tree else []),'-z').decode().split('\0'))-{''}
    if working_tree:
        names={n for n in names if (ROOT/n).exists() or (ROOT/n).is_symlink()}
    read=lambda n:(ROOT/n).read_bytes() if working_tree else git('show',':./'+n)
    manifest=json.loads(read('PUBLICATION_MANIFEST.json'))
    if manifest['schema_version']!='dataset-publication-v1' or set(manifest['datasets'])!=set(DATASETS):
        raise ValueError('Unexpected publication scope')
    artifacts=manifest['files']
    code=set(manifest['code_files'])|{'PUBLICATION_MANIFEST.json'}
    errors=[]
    for name in sorted(set(artifacts)-names):
        errors.append((name,'approved artifact missing from Git candidates/index'))
    for name in sorted(code-names):
        errors.append((name,'required source/control file missing'))
    attrs=git('check-attr',*(['--cached'] if not working_tree else []),'-z','--stdin','filter',
              input=b'\0'.join(n.encode() for n,e in artifacts.items() if e['lfs'])+b'\0').decode().split('\0')
    filters={attrs[i]:attrs[i+2] for i in range(0,len(attrs)-2,3)}
    for name in sorted(names):
        p=PurePosixPath(name)
        entry=artifacts.get(name)
        if p.is_absolute() or '..' in p.parts or any(x in {'.env','.aws','.ssh','.local_data_links.json'} for x in p.parts):
            errors.append((name,'forbidden local/credential path'));continue
        if entry and not any(name.startswith(root+'/') for root in DATASETS):
            errors.append((name,'artifact outside approved roots'));continue
        if not entry and name not in code:
            errors.append((name,'outside selected publication files'));continue
        if working_tree:
            target=ROOT/name
            if target.is_symlink() or any(q.is_symlink() for q in target.parents if q!=ROOT and ROOT in q.parents):
                errors.append((name,'symlink is not a materialized artifact'));continue
            size=target.stat().st_size
        else:
            stage=git('ls-files','--stage','--',name).decode()
            if not stage.startswith(('100644 ','100755 ')):
                errors.append((name,'non-regular or unmerged index entry'));continue
            size=int(git('cat-file','-s',':./'+name))
        if entry:
            if entry['lfs'] and filters.get(name)!='lfs':
                errors.append((name,'missing LFS filter attribute'));continue
            if working_tree:
                if size!=entry['bytes'] or sha256(ROOT/name)!=entry['sha256']:
                    errors.append((name,'materialized file differs from publication manifest'))
                if not entry['lfs']:
                    errors.extend((name,e) for e in text_problems(name,read(name),True))
            elif entry['lfs']:
                if size>1024:
                    errors.append((name,'large binary was staged without LFS'));continue
                problem=pointer_problem(read(name),entry)
                if problem:errors.append((name,problem))
            else:
                payload=read(name)
                if len(payload)!=entry['bytes'] or hashlib.sha256(payload).hexdigest()!=entry['sha256']:
                    errors.append((name,'indexed artifact differs from publication manifest'))
                errors.extend((name,e) for e in text_problems(name,payload,True))
        elif size>MAX_SOURCE_BYTES:
            errors.append((name,'source exceeds 2 MiB'))
        else:
            errors.extend((name,e) for e in text_problems(name,read(name)))
    return {'status':'FAIL' if errors else 'PASS','files':len(names),'artifacts':len(artifacts),
            'lfs_files':sum(e['lfs'] for e in artifacts.values()),'errors':errors}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--working-tree',action='store_true')
    mode.add_argument('--staged',action='store_true')
    args=parser.parse_args()
    report=validate(args.working_tree)
    for name,reason in report['errors']:
        print('BLOCKED',name+':',reason)
    print(json.dumps({k:v for k,v in report.items() if k!='errors'}))
    raise SystemExit(bool(report['errors']))
