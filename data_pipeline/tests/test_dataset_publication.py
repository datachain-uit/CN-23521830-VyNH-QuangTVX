"""Check actual Git LFS staging and release integrity, without network access."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import duckdb
import pytest

PIPELINE=Path(__file__).resolve().parents[1]


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_pointer_must_match_approved_bytes():
    check=module('publication_check',PIPELINE/'scripts/check_github_payload.py')
    entry={'sha256':'a'*64,'bytes':123}
    pointer=f'version https://git-lfs.github.com/spec/v1\noid sha256:{entry["sha256"]}\nsize 123\n'.encode()
    assert check.pointer_problem(pointer,entry) is None
    assert check.pointer_problem(b'PAR1'+b'x'*123,entry)
    assert check.pointer_problem(pointer,{**entry,'bytes':124})
    assert check.pointer_problem(pointer+b'extra\n',entry)


def test_link_map_must_match_actual_symlink(tmp_path):
    prepare=module('publication_prepare',PIPELINE/'scripts/prepare_dataset_release.py')
    source=tmp_path/'source';source.mkdir()
    actual=tmp_path/'actual';(actual/'moocubex_model').mkdir(parents=True)
    (source/'input').symlink_to(actual,target_is_directory=True)
    links={'links':[{'path':'data_pipeline/input','source':str(actual)}]}
    assert prepare.resolve_dataset(source,links,'input/moocubex_model')==actual/'moocubex_model'
    links['links'][0]['source']=str(tmp_path/'wrong')
    with pytest.raises(ValueError,match='disagrees'):
        prepare.resolve_dataset(source,links,'input/moocubex_model')


def test_unindexed_staging_is_omitted_but_frozen_history_kept(tmp_path):
    prepare=module('publication_prepare_files',PIPELINE/'scripts/prepare_dataset_release.py')
    (tmp_path/'_stage').mkdir();(tmp_path/'_stage/cache.parquet').write_bytes(b'cache')
    (tmp_path/'frozen.log').write_text('historical tests passed')
    (tmp_path/'data.parquet').write_bytes(b'PAR1')
    digest=hashlib.sha256((tmp_path/'frozen.log').read_bytes()).hexdigest()
    (tmp_path/'SHA256SUMS.txt').write_text(digest+'  frozen.log\n')
    files,_=prepare.release_files(tmp_path)
    assert {r for r,_,_ in files}=={'frozen.log','data.parquet','SHA256SUMS.txt'}


def test_real_lfs_index_and_unapproved_file_are_checked(tmp_path,monkeypatch):
    repo=tmp_path/'repo';root=repo/'data_pipeline';root.mkdir(parents=True)
    def git(*args):
        return subprocess.check_output(['git',*args],cwd=repo,stderr=subprocess.STDOUT)
    git('init','-q');git('lfs','install','--local')
    (root/'.gitattributes').write_text('*.parquet filter=lfs diff=lfs merge=lfs -text\n')
    name='phase_features_v4_2/raw_features/small.parquet'
    p=root/name;p.parent.mkdir(parents=True)
    con=duckdb.connect()
    con.execute('COPY (SELECT 1 n) TO ? (FORMAT PARQUET)',[str(p)]);con.close()
    digest=hashlib.sha256(p.read_bytes()).hexdigest()
    manifest={'schema_version':'dataset-publication-v1','datasets':{k:{} for k in
        ['input/moocubex_model','final_quality_v4','phase_features_v4','phase_features_v4_2']},
        'code_files':['.gitattributes'],'files':{name:{'sha256':digest,'bytes':p.stat().st_size,'lfs':True}}}
    (root/'PUBLICATION_MANIFEST.json').write_text(json.dumps(manifest))
    git('add','data_pipeline')
    check=module('publication_real_index',PIPELINE/'scripts/check_github_payload.py')
    monkeypatch.setattr(check,'ROOT',root)
    assert check.validate()['status']=='PASS'
    assert check.validate(True)['status']=='PASS'
    p.write_bytes(b'changed')
    assert check.validate(True)['status']=='FAIL'
    assert check.validate()['status']=='PASS'  # Actual index still contains approved pointer.
    (root/'unapproved.json').write_text('{}');git('add','data_pipeline/unapproved.json')
    assert any('outside selected' in reason for _,reason in check.validate()['errors'])
