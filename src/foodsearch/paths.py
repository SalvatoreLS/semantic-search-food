import os
from pathlib import Path

ROOT_ENV_VAR = "FOODSEARCH_ROOT"


def project_root() -> Path:
    return Path(os.environ.get(ROOT_ENV_VAR) or Path.cwd()).resolve()


def data_dir() -> Path:
    return project_root() / "data"


def artifacts_dir() -> Path:
    return project_root() / "artifacts"


def configs_dir() -> Path:
    return project_root() / "configs"


def items_csv() -> Path:
    return data_dir() / "5k_items_curated.csv"


def queries_csv() -> Path:
    return data_dir() / "queries.csv"


def items_parquet() -> Path:
    return artifacts_dir() / "items.parquet"


def runs_dir() -> Path:
    return artifacts_dir() / "runs"


def cache_dir() -> Path:
    return artifacts_dir() / "cache"


def cost_log() -> Path:
    return artifacts_dir() / "cost_log.jsonl"
