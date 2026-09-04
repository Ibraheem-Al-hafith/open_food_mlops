"""Logging configuration loader using centralized environment settings."""

import os
import logging
import logging.config
import yaml

from open_food_mlops.config.settings import settings


def setup_logging(config_path: str | None = None) -> None:
    """Initializes the logging infrastructure using dynamic configuration.

    Args:
        config_path: Path to logging YAML configuration file. Defaults to `settings.log_config_path`.
    """
    resolved_path = config_path or settings.log_config_path

    if not os.path.exists(resolved_path):
        logging.basicConfig(level=logging.INFO)
        logging.warning(
            "Logging configuration file not found at %s. Fallback to basic configuration.",
            resolved_path,
        )
        return

    try:
        with open(resolved_path, "r", encoding="utf-8") as yaml_config_file:
            config = yaml.safe_load(yaml_config_file)

        handlers = config.get("handlers", {})
        for handler_config in handlers.values():
            if "filename" in handler_config:
                log_file = handler_config["filename"]
                log_dir = os.path.dirname(log_file)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)

        root_logger = logging.getLogger()
        if root_logger.handlers:
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)

        logging.config.dictConfig(config=config)
        logging.info(
            "Logging infrastructure successfully configured via %s.",
            resolved_path,
        )

    except Exception as e:
        logging.basicConfig(level=logging.INFO)
        logging.error(
            "Fatal failure initializing logging configuration: %s",
            str(e),
            exc_info=True,
        )