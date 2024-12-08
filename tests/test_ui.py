# tests/test_ui.py

import pytest
from unittest.mock import patch
from ui import get_user_input, get_yes_no_input, prompt_add_documents, prompt_continue_conversation


# Helper function to mock input calls
def mock_inputs(inputs):
    return patch("builtins.input", side_effect=inputs)

# Helper function to mock sys.exit
def mock_sys_exit():
    return patch("sys.exit", side_effect=SystemExit)


@mock_sys_exit()
@patch("ui.log_info")  # Mock log_info correctly
def test_get_user_input_exit(mock_log_info, mock_exit):
    """Test get_user_input exits when 'exit' is entered."""
    with mock_inputs(["exit"]):
        with pytest.raises(SystemExit):
            get_user_input("Enter something: ", exit_message="Goodbye!")
        mock_log_info.assert_called_once_with("User exited via input.")
        mock_exit.assert_called_once_with(0)


@mock_sys_exit()
@patch("ui.log_info")  # Mock log_info correctly
def test_get_user_input_quit(mock_log_info, mock_exit):
    """Test get_user_input exits when 'quit' is entered."""
    with mock_inputs(["quit"]):
        with pytest.raises(SystemExit):
            get_user_input("Enter something: ", exit_message="Goodbye!")
        mock_log_info.assert_called_once_with("User exited via input.")
        mock_exit.assert_called_once_with(0)


def test_get_user_input_valid():
    """Test get_user_input returns valid input."""
    with mock_inputs(["Hello"]):
        result = get_user_input("Enter something: ")
        assert result == "Hello"


@mock_sys_exit()
@patch("ui.log_info")  # Mock log_info correctly
def test_get_yes_no_input_exit(mock_log_info, mock_exit):
    """Test get_yes_no_input exits when 'exit' or 'quit' is entered."""
    with mock_inputs(["exit"]):
        with pytest.raises(SystemExit):
            get_yes_no_input("Confirm? (yes/no): ")
        mock_log_info.assert_called_once_with("User exited via yes/no prompt.")
        mock_exit.assert_called_once_with(0)


def test_get_yes_no_input_valid():
    """Test get_yes_no_input with valid responses."""
    # Test yes input
    with mock_inputs(["yes"]):
        result = get_yes_no_input("Confirm? (yes/no): ")
        assert result is True

    # Test no input
    with mock_inputs(["no"]):
        result = get_yes_no_input("Confirm? (yes/no): ")
        assert result is False


def test_get_yes_no_input_invalid_then_valid():
    """Test get_yes_no_input handles invalid input before valid input."""
    with mock_inputs(["invalid", "yes"]), patch("builtins.print") as mock_print:
        result = get_yes_no_input("Confirm? (yes/no): ")
        assert result is True
        mock_print.assert_called_with("Please enter 'yes' or 'no'.")


@patch("ui.get_yes_no_input", return_value=True)
def test_prompt_add_documents_yes(mock_get_yes_no_input):
    """Test prompt_add_documents returns True."""
    result = prompt_add_documents()
    assert result is True
    mock_get_yes_no_input.assert_called_once_with("Do you want to add documents to the vector database? (yes/no): ")


@patch("ui.get_yes_no_input", return_value=False)
def test_prompt_add_documents_no(mock_get_yes_no_input):
    """Test prompt_add_documents returns False."""
    result = prompt_add_documents()
    assert result is False
    mock_get_yes_no_input.assert_called_once_with("Do you want to add documents to the vector database? (yes/no): ")


@patch("ui.get_yes_no_input", return_value=True)
def test_prompt_continue_conversation_yes(mock_get_yes_no_input):
    """Test prompt_continue_conversation returns True."""
    result = prompt_continue_conversation()
    assert result is True
    mock_get_yes_no_input.assert_called_once_with("Do you want to continue the conversation? (yes/no): ")


@patch("ui.get_yes_no_input", return_value=False)
def test_prompt_continue_conversation_no(mock_get_yes_no_input):
    """Test prompt_continue_conversation returns False."""
    result = prompt_continue_conversation()
    assert result is False
    mock_get_yes_no_input.assert_called_once_with("Do you want to continue the conversation? (yes/no): ")