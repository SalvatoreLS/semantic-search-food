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


def query_types_csv() -> Path:
    return artifacts_dir() / "query_types.csv"


def human_label_queries_csv() -> Path:
    return artifacts_dir() / "human_label_queries.csv"


def pool_json() -> Path:
    return artifacts_dir() / "pool.json"


def judgments_dir() -> Path:
    return artifacts_dir() / "judgments"


def qrels_json() -> Path:
    return artifacts_dir() / "qrels.json"


def labels_dir() -> Path:
    return artifacts_dir() / "labels"


def eval_freeze_json() -> Path:
    return artifacts_dir() / "eval_freeze.json"


def reports_dir() -> Path:
    return project_root() / "reports"
