"""Publishing guard rails (no network)."""

from __future__ import annotations

import json

import pytest

from processing import publish


def _release(tmp_path, commit="abc"):
    (tmp_path / "data" / "ami").mkdir(parents=True)
    (tmp_path / "data" / "ami" / "train.parquet").write_bytes(b"x")
    (tmp_path / "README.md").write_text("card")
    (tmp_path / "dataset-metadata.json").write_text("{}")
    (tmp_path / "stats.json").write_text(json.dumps({"provenance": {"git_commit": commit}}))
    return tmp_path


def test_incomplete_release_is_refused(tmp_path):
    with pytest.raises(publish.PublishError):
        publish.check_release(tmp_path, require_clean_commit=False)


def test_stale_commit_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(publish, "_git", lambda *a: "" if a[0] == "status" else "def")
    with pytest.raises(publish.PublishError, match="rebuild"):
        publish.check_release(_release(tmp_path, commit="abc"))


def test_dirty_tree_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(publish, "_git", lambda *a: " M file.py" if a[0] == "status" else "abc")
    with pytest.raises(publish.PublishError, match="uncommitted"):
        publish.check_release(_release(tmp_path))


def test_dry_run_publishes_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(publish, "publish_hf", lambda *a: pytest.fail("must not upload"))
    monkeypatch.setattr(publish.subprocess, "run", lambda *a, **k: pytest.fail("must not run"))
    assert (
        publish.main(["--release", str(_release(tmp_path)), "--hf", "--kaggle", "--dry-run"]) == 0
    )
    out = capsys.readouterr().out
    assert "nothing published" in out and "kaggle datasets version" in out


def test_kaggle_command_create_vs_version(tmp_path):
    assert publish.kaggle_command(tmp_path, new=True, tag="v1.0")[2] == "create"
    assert "version" in publish.kaggle_command(tmp_path, new=False, tag="v1.0")
