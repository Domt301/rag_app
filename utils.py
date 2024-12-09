from tqdm import tqdm
import time
import logging

# Configure logging
logging.basicConfig(
    filename='app.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def display_progress(total, description="Processing"):
    """
    Displays a progress bar for long-running processes.
    """
    pbar = tqdm(total=total, desc=description)
    return pbar

def log_info(message):
    """
    Logs informational messages to the log file.
    """
    logging.info(message)

def log_error(message):
    """
    Logs error messages to the log file.
    """
    logging.error(message)

def log_conversation(user_query, agent_response, conversation_history=None):
    """
    Logs the conversation between the user and the agent, including the context if provided.

    Args:
        user_query (str): The user's input query.
        agent_response (str): The agent's response.
        conversation_history (list, optional): The conversation history, if available.
    """
    history_str = ""
    if conversation_history:
        history_str = "\n".join(
            [f"User: {turn['query']} | Agent: {turn['response']}" for turn in conversation_history]
        )
    
    logging.info(
        f"User query: {user_query}\n"
        f"Agent response: {agent_response}\n"
        f"Conversation history:\n{history_str if history_str else 'No prior context'}\n"
    )