import os
from pypdf import PdfReader  # Updated import
import docx
import math
from utils import log_error, log_info

def read_pdf(file_path):
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
        return text
    except Exception as e:
        log_error(f"Error reading PDF {file_path}: {e}")
        return ""

def read_docx(file_path):
    try:
        doc = docx.Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
        log_error(f"Error reading DOCX {file_path}: {e}")
        return ""

def read_txt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        log_error(f"Error reading TXT {file_path}: {e}")
        return ""

def chunk_text(text, max_length=1000):
    try:
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            # Ensure the sentence ends with a single period
            sentence = sentence.strip()
            if not sentence.endswith('.'):
                sentence += '.'
            
            # Check if adding this sentence exceeds the max_length
            if len(current_chunk) + len(sentence) + 1 <= max_length:
                current_chunk += sentence + ' '
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + ' '
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks
    except Exception as e:
        log_error(f"Error chunking text: {e}")
        return []

def process_files(directory, batch_size=100):
    try:
        supported_extensions = ['.pdf', '.docx', '.txt']
        all_chunks = []
        for filename in os.listdir(directory):
            ext = os.path.splitext(filename)[1].lower()
            if ext not in supported_extensions:
                log_info(f"Skipping unsupported file type: {filename}")
                continue
            file_path = os.path.join(directory, filename)
            if ext == '.pdf':
                text = read_pdf(file_path)
            elif ext == '.docx':
                text = read_docx(file_path)
            elif ext == '.txt':
                text = read_txt(file_path)
            if text:
                chunks = chunk_text(text)
                all_chunks.extend(chunks)
        log_info(f"Processed {len(all_chunks)} chunks from directory {directory}.")
        return all_chunks
    except Exception as e:
        log_error(f"Error processing files in directory {directory}: {e}")
        return []