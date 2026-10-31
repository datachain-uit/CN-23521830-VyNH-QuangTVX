"""Exercise the real checker against staged blobs and the withdrawn release."""
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

PIPELINE = Path(__file__).resolve().parents[1]
CHECKER = PIPELINE / "scripts/check_github_payload.py"
SPEC = importlib.util.spec_from_file_location("github_payload", CHECKER)
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)
BAD_RELEASE = "8c6a52c15f749182b00240d32266cae6cc242d03"
POINTER = (b"version https://git-lfs.github.com/spec/v1\noid sha256:"
           + b"a" * 64 + b"\nsize 133\n")


def git(repo, *args, input=None):
    return subprocess.check_output(["git", *args], cwd=repo, input=input)


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    pipeline = root / "data_pipeline"
    (pipeline / "scripts").mkdir(parents=True)
    git(root, "init", "--quiet")
    git(root, "config", "core.hooksPath", "/dev/null")
    git(root, "config", "user.name", "Synthetic fixture")
    git(root, "config", "user.email", "fixture@example.invalid")
    shutil.copyfile(CHECKER, pipeline / "scripts/check_github_payload.py")
    shutil.copyfile(PIPELINE / "cq_patch_1031.py", pipeline / "cq_patch_1031.py")
    git(root, "add", "data_pipeline")
    return root


def stage_blob(repo, name, data):
    oid = git(repo, "hash-object", "-w", "--stdin", input=data).decode().strip()
    git(repo, "update-index", "--add", "--cacheinfo", "100644", oid, name)


def run_checker(repo, *args):
    return subprocess.run([sys.executable, str(repo / "data_pipeline/scripts/check_github_payload.py"),
                           *args], cwd=repo, capture_output=True, text=True)


def test_clean_code_and_commented_lfs_rule_pass(repo):
    stage_blob(repo, ".gitattributes", b"# *.parquet filter=lfs\n*.py text eol=lf\n")
    result = run_checker(repo, "--staged")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS:" in result.stdout


def test_staged_root_rule_cannot_be_hidden_by_clean_working_tree(repo):
    stage_blob(repo, ".gitattributes", b"*.parquet filter=lfs diff=lfs -text\n")
    (repo / ".gitattributes").write_text("*.py text\n")
    result = run_checker(repo, "--staged")
    assert result.returncode == 1
    assert "BLOCKED .gitattributes:1: luật Git LFS" in result.stdout


def test_pointer_outside_pipeline_and_force_added_data_are_blocked(repo):
    stage_blob(repo, "experiments/disguised.py", POINTER)
    stage_blob(repo, "data_pipeline/fixture.parquet", b"private-placeholder")
    result = run_checker(repo, "--staged")
    assert result.returncode == 1
    assert "BLOCKED experiments/disguised.py: con trỏ Git LFS" in result.stdout
    assert "fixture.parquet" in result.stdout
    assert "private-placeholder" not in result.stdout
    assert "oid sha256" not in result.stdout


def test_tree_mode_reads_commit_instead_of_clean_index(repo):
    stage_blob(repo, "data_pipeline/helper.py", POINTER)
    stage_blob(repo, "experiments/.gitattributes", b"*.bin filter=lfs\n")
    git(repo, "commit", "--quiet", "-m", "Synthetic withdrawn release")
    stage_blob(repo, "data_pipeline/helper.py", b"print('clean')\n")
    stage_blob(repo, "experiments/.gitattributes", b"*.py text\n")
    assert run_checker(repo, "--staged").returncode == 0
    result = run_checker(repo, "--tree", "HEAD")
    assert result.returncode == 1
    assert "BLOCKED data_pipeline/helper.py: con trỏ Git LFS" in result.stdout
    assert "BLOCKED experiments/.gitattributes:1: luật Git LFS" in result.stdout


def test_working_tree_detects_root_rule_and_untracked_pointer(repo):
    (repo / ".gitattributes").write_text("*.parquet filter=lfs\n")
    (repo / "hidden.py").write_bytes(POINTER)
    result = run_checker(repo, "--working-tree")
    assert result.returncode == 1
    assert "BLOCKED .gitattributes:1: luật Git LFS" in result.stdout
    assert "BLOCKED hidden.py: con trỏ Git LFS" in result.stdout


def test_withdrawn_release_8c6a52c_is_blocked():
    exists = subprocess.run(["git", "cat-file", "-e", BAD_RELEASE + "^{commit}"],
                            cwd=PIPELINE, capture_output=True)
    if exists.returncode:
        pytest.skip("Withdrawn commit was purged; synthetic regression remains mandatory")
    result = subprocess.run([sys.executable, str(CHECKER), "--tree", BAD_RELEASE],
                            cwd=PIPELINE, capture_output=True, text=True)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "con trỏ Git LFS" in result.stdout
    assert "luật Git LFS" in result.stdout
    assert "input/moocubex_model/" in result.stdout
    assert "FAILED:" in result.stdout
