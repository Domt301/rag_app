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
    pbar = tqdm(total=total, desc=description)
    return pbar

def log_info(message):
    logging.info(message)

def log_error(message):
    logging.error(message)