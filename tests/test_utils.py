import logging

from src.utils.logger import setup_logger


def test_setup_logger_creates_logger(tmp_path):
    """Test that setup_logger creates a logger correctly."""

    log_file = tmp_path / "test.log"

    logger = setup_logger(
        name="test_logger",
        log_file=str(log_file),
    )

    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_logger"
    assert logger.level == logging.INFO


def test_setup_logger_creates_log_file(tmp_path):
    """Test that the logger creates the log file."""

    log_file = tmp_path / "test.log"

    logger = setup_logger(
        name="test_file_logger",
        log_file=str(log_file),
    )

    logger.info("Test log message")

    # Make sure logging handlers have written the message.
    for handler in logger.handlers:
        handler.flush()

    assert log_file.exists()

    content = log_file.read_text(encoding="utf-8")

    assert "Test log message" in content


def test_setup_logger_does_not_duplicate_handlers(tmp_path):
    """Test that calling setup_logger twice does not add duplicate handlers."""

    log_file = tmp_path / "test_duplicate.log"

    logger = setup_logger(
        name="test_duplicate_logger",
        log_file=str(log_file),
    )

    first_handler_count = len(logger.handlers)

    logger_again = setup_logger(
        name="test_duplicate_logger",
        log_file=str(log_file),
    )

    second_handler_count = len(logger_again.handlers)

    assert logger_again is logger
    assert first_handler_count == second_handler_count
