"""Release settings: committed metadata + IDs and links from .env or the environment."""

from __future__ import annotations

import pytest

from processing.release_config import (
    DEFAULT_RELEASE_CONFIG,
    ReleaseConfigError,
    load_release_settings,
    read_env_file,
)
from tests.conftest import TEST_RELEASE_ENV


def test_env_file_parsing(tmp_path):
    env = tmp_path / ".env"
    env.write_text("# comment\n\nexport A=1\nB = \"two words\"\nC='x=y'\n", encoding="utf-8")
    assert read_env_file(env) == {"A": "1", "B": "two words", "C": "x=y"}
    assert read_env_file(tmp_path / "missing") == {}


def test_env_file_rejects_bad_line(tmp_path):
    env = tmp_path / ".env"
    env.write_text("NOT_A_PAIR\n", encoding="utf-8")
    with pytest.raises(ReleaseConfigError, match="KEY=VALUE"):
        read_env_file(env)


def test_values_come_from_env_file(tmp_path):
    env = tmp_path / ".env"
    env.write_text("\n".join(f"{k}={v}" for k, v in TEST_RELEASE_ENV.items()), encoding="utf-8")
    settings = load_release_settings(env_file=env, environ={})
    assert settings.hf_repo_id == TEST_RELEASE_ENV["HF_REPO_ID"]
    assert settings.tag == "v1.0"


def test_environment_wins_over_env_file(tmp_path):
    env = tmp_path / ".env"
    env.write_text("\n".join(f"{k}={v}" for k, v in TEST_RELEASE_ENV.items()), encoding="utf-8")
    environ = {"HF_REPO_ID": "other/repo"}
    assert load_release_settings(env_file=env, environ=environ).hf_repo_id == "other/repo"


def test_missing_values_are_named(tmp_path):
    with pytest.raises(ReleaseConfigError, match="KAGGLE_DATASET_ID"):
        load_release_settings(env_file=tmp_path / "none", environ={"HF_REPO_ID": "a/b"})


def test_placeholders_are_rejected(tmp_path):
    environ = {**TEST_RELEASE_ENV, "HF_REPO_ID": "<hf-user>/meeting-notes-extractive-v1"}
    with pytest.raises(ReleaseConfigError, match="invalid"):
        load_release_settings(env_file=tmp_path / "none", environ=environ)


def test_ids_may_not_be_committed_in_the_config(tmp_path):
    config = tmp_path / "release.yaml"
    config.write_text(
        DEFAULT_RELEASE_CONFIG.read_text(encoding="utf-8") + "hf_repo_id: a/b\n", encoding="utf-8"
    )
    with pytest.raises(ReleaseConfigError, match=r"belong in \.env"):
        load_release_settings(config, env_file=tmp_path / "none", environ=TEST_RELEASE_ENV)
