import sys
import time
from utils import log_info

def get_user_input(prompt_message, exit_message=None):
    """
    Get input from the user with an optional exit message.
    """
    while True:
        user_input = input(prompt_message).strip()
        if user_input.lower() in ['exit', 'quit']:
            if exit_message:
                print(exit_message)
            log_info("User exited via input.")
            sys.exit(0)
        return user_input

def get_yes_no_input(prompt_message):
    """
    Get a yes/no response from the user with input validation.
    """
    while True:
        choice = input(prompt_message).strip().lower()
        if choice in ['yes', 'y']:
            return True
        elif choice in ['no', 'n']:
            return False
        elif choice in ['exit', 'quit']:
            print("Exiting...")
            log_info("User exited via yes/no prompt.")
            sys.exit(0)
        else:
            print("Please enter 'yes' or 'no'.")

def prompt_add_documents():
    """
    Prompt the user to decide if they want to add documents.
    """
    return get_yes_no_input("Do you want to add documents to the vector database? (yes/no): ")

def prompt_fallback_mode():
    """
    Inform the user about fallback mode when no documents are in the database.
    """
    print(
        "No documents found in the vector database. The AI will answer questions using general knowledge only. "
        "To improve responses, consider adding documents."
    )
    log_info("User informed about fallback mode.")

def prompt_continue_conversation():
    """
    Prompt the user to decide if they want to continue the conversation.
    """
    return get_yes_no_input("Do you want to continue the conversation? (yes/no): ")

def show_loading_message(message, duration=2):
    """
    Display a loading message for a specific duration.
    """
    print(message, end="", flush=True)
    for _ in range(duration):
        time.sleep(1)
        print(".", end="", flush=True)
    print()  # End the loading line