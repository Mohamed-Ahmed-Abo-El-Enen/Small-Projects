import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from typing import Optional
import json
from datetime import datetime

from app.core.config import settings


class ColoredFormatter(logging.Formatter):
    """Colored console formatter"""

    COLORS = {
        'DEBUG': '\033[36m',  # Cyan
        'INFO': '\033[32m',  # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',  # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'  # Reset
    }

    def format(self, record):
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"

        result = super().format(record)

        record.levelname = levelname

        return result


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""

    def format(self, record):
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }

        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        if hasattr(record, 'user_id'):
            log_data['user_id'] = record.user_id
        if hasattr(record, 'conversation_id'):
            log_data['conversation_id'] = record.conversation_id
        if hasattr(record, 'query'):
            log_data['query'] = record.query
        if hasattr(record, 'duration'):
            log_data['duration_ms'] = record.duration

        return json.dumps(log_data)


class LoggerManager:
    """Centralized logger management"""

    _loggers = {}
    _initialized = False

    @classmethod
    def setup_logging(cls,
                      log_level: str = "INFO",
                      log_dir: str = "logs",
                      enable_console: bool = True,
                      enable_file: bool = True,
                      enable_json: bool = False,
                      max_bytes: int = 10_000_000,
                      backup_count: int = 5):
        """ Setup centralized logging configuration """
        if cls._initialized:
            return

        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level.upper()))

        root_logger.handlers.clear()

        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.DEBUG)

            console_format = ColoredFormatter(
                '%(levelname)-8s | %(asctime)s | %(name)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(console_format)
            root_logger.addHandler(console_handler)

        if enable_file:
            file_handler = RotatingFileHandler(
                log_path / 'app.log',
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(logging.DEBUG)

            file_format = logging.Formatter(
                '%(levelname)-8s | %(asctime)s | %(name)s | %(module)s:%(funcName)s:%(lineno)d | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_format)
            root_logger.addHandler(file_handler)

            error_handler = RotatingFileHandler(
                log_path / 'error.log',
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(file_format)
            root_logger.addHandler(error_handler)

        if enable_json:
            json_handler = TimedRotatingFileHandler(
                log_path / 'app.json',
                when='midnight',
                interval=1,
                backupCount=30,
                encoding='utf-8'
            )
            json_handler.setLevel(logging.INFO)
            json_handler.setFormatter(JSONFormatter())
            root_logger.addHandler(json_handler)

        cls._initialized = True

        logger = cls.get_logger(__name__)
        logger.info("=" * 60)
        logger.info("Logging system initialized")
        logger.info(f"Log level: {log_level}")
        logger.info(f"Log directory: {log_path.absolute()}")
        logger.info(f"Console logging: {enable_console}")
        logger.info(f"File logging: {enable_file}")
        logger.info(f"JSON logging: {enable_json}")
        logger.info("=" * 60)

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """ Get or create a logger with the given name """
        if name not in cls._loggers:
            cls._loggers[name] = logging.getLogger(name)

        return cls._loggers[name]

    @classmethod
    def set_level(cls, level: str, logger_name: Optional[str] = None):
        """ Change log level dynamically """
        level_obj = getattr(logging, level.upper())

        if logger_name:
            logger = cls.get_logger(logger_name)
            logger.setLevel(level_obj)
        else:
            logging.getLogger().setLevel(level_obj)

    @classmethod
    def add_context(cls, logger: logging.Logger, **kwargs):
        """ Add context to logger (creates a LoggerAdapter """
        return logging.LoggerAdapter(logger, kwargs)


def get_logger(name: str) -> logging.Logger:
    """ Get a logger instance """
    return LoggerManager.get_logger(name)


LoggerManager.setup_logging(
    log_level=getattr(settings, 'LOG_LEVEL', 'INFO'),
    log_dir=getattr(settings, 'LOG_DIR', 'logs'),
    enable_console=True,
    enable_file=True,
    enable_json=getattr(settings, 'LOG_JSON', False)
)