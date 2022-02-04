import logging.config


def configure_logging():
    DEFAULT_LOGGING_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "basic": {
                "format": "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
            }
        },
        "handlers": {
            "console": {
                "formatter": "basic",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            }
        },
        "loggers": {
            "jmetal": {"handlers": ["console"], "level": "DEBUG"},
        },
    }

    logging.config.dictConfig(DEFAULT_LOGGING_CONFIG)


def get_logger(module):
    return logging.getLogger(module)
