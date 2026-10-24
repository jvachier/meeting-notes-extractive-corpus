"""Release settings: committed metadata plus account IDs and links kept out of git.

Fixed metadata (version, tag, titles, Kaggle keywords) lives in ``configs/release.yaml``.
The Hugging Face repo ID, Kaggle dataset ID and source repository URL come from the
environment or, failing that, from a git-ignored ``.env`` file (template: ``.env.example``).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from generation.config import REPO_ROOT

DEFAULT_RELEASE_CONFIG = REPO_ROOT / "configs" / "release.yaml"
DEFAULT_ENV_FILE = REPO_ROOT / ".env"
# Settings field -> environment variable.
ENV_KEYS: dict[str, str] = {
    "hf_repo_id": "HF_REPO_ID",
    "kaggle_dataset_id": "KAGGLE_DATASET_ID",
    "source_repo_url": "SOURCE_REPO_URL",
}
_REPO_ID = r"^[A-Za-z0-9][\w.-]*/[\w.-]+$"


class ReleaseConfigError(RuntimeError):
    """The release settings are missing or invalid."""


class ReleaseSettings(BaseModel):
    """Everything the dataset card, Kaggle metadata and publish step need."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str
    tag: str
    title: str
    subtitle: str
    kaggle_licence: str
    kaggle_keywords: tuple[str, ...]
    hf_repo_id: str = Field(pattern=_REPO_ID)
    kaggle_dataset_id: str = Field(pattern=_REPO_ID)
    source_repo_url: str = Field(pattern=r"^https://\S+$")


def read_env_file(path: Path) -> dict[str, str]:
    """Parse ``KEY=VALUE`` lines; skip blanks and ``#`` comments; strip matching quotes."""
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("export ").lstrip()
        key, sep, value = line.partition("=")
        if not sep or not key.strip():
            raise ReleaseConfigError(f"{path}:{number}: expected KEY=VALUE")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def load_release_settings(
    config: Path = DEFAULT_RELEASE_CONFIG,
    env_file: Path = DEFAULT_ENV_FILE,
    environ: Mapping[str, str] | None = None,
) -> ReleaseSettings:
    """Merge ``config`` with the IDs and links from ``environ`` (default ``os.environ``)
    and ``env_file``; the environment wins over the file."""
    environ = os.environ if environ is None else environ
    meta = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    committed = sorted(set(meta) & set(ENV_KEYS))
    if committed:
        raise ReleaseConfigError(f"{config}: {', '.join(committed)} belong in .env, not in git")
    file_values = read_env_file(env_file)
    links: dict[str, str] = {}
    missing: list[str] = []
    for field, key in ENV_KEYS.items():
        value = environ.get(key) or file_values.get(key)
        if value:
            links[field] = value
        else:
            missing.append(key)
    if missing:
        raise ReleaseConfigError(
            f"missing {', '.join(missing)}: copy .env.example to .env and fill it in"
        )
    try:
        return ReleaseSettings(**meta, **links)
    except ValidationError as exc:
        raise ReleaseConfigError(f"invalid release settings: {exc}") from exc
