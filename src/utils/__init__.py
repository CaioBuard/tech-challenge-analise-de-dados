"""Utilitarios compartilhados."""

import os
import logging
from pathlib import Path
from datetime import datetime


def setup_logging(name: str = "tech_challenge") -> logging.Logger:
    """Configura logging estruturado."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def get_project_root() -> Path:
    """Retorna o diretório raiz do projeto."""
    return Path(__file__).parent.parent.parent


def get_data_dir(subdir: str = "") -> Path:
    """Retorna o diretório de dados."""
    return get_project_root() / "data" / "datasets" / subdir


def get_output_dir(subdir: str = "") -> Path:
    """Retorna o diretório de saída."""
    output = get_project_root() / "data" / "outputs" / subdir
    output.mkdir(parents=True, exist_ok=True)
    return output


def generate_report_filename(prefix: str = "report") -> str:
    """Gera nome de arquivo com timestamp."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}"


def get_env_bool(name: str, default: bool = False) -> bool:
    """Le flag booleana de ambiente com fallback seguro."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "sim"}
