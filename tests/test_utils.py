# tests/test_utils.py

import pytest
from unittest.mock import patch, MagicMock
from utils import display_progress, log_info, log_error


@patch("utils.tqdm")
def test_display_progress(mock_tqdm):
    """Test display_progress initializes and returns a progress bar."""
    mock_pbar = MagicMock()
    mock_tqdm.return_value = mock_pbar

    total = 10
    description = "Testing Progress"
    pbar = display_progress(total, description)

    # Assertions
    mock_tqdm.assert_called_once_with(total=total, desc=description)
    assert pbar == mock_pbar


@patch("utils.logging.info")
def test_log_info(mock_logging_info):
    """Test log_info logs an info-level message."""
    message = "This is an info log message."
    log_info(message)

    # Assertions
    mock_logging_info.assert_called_once_with(message)


@patch("utils.logging.error")
def test_log_error(mock_logging_error):
    """Test log_error logs an error-level message."""
    message = "This is an error log message."
    log_error(message)

    # Assertions
    mock_logging_error.assert_called_once_with(message)