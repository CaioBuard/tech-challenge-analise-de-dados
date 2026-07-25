"""Utilitarios compartilhados."""

import logging


def setup_logging(name: str = "tech_challenge") -> logging.Logger:
    """Configura um logger reutilizavel para a aplicacao."""
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
