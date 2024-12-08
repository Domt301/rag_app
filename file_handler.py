import os
from pypdf import PdfReader
import docx
from langchain.text_splitter import CharacterTextSplitter  
from utils import log_error, log_info

def read_pdf(file_path):
    """
    Reads and extracts text from a PDF file.
    """
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
        log_info(f"Successfully read PDF: {file_path}")
        return text
    except Exception as e:
        log_error(f"Error reading PDF {file_path}: {e}")
        return ""

def read_docx(file_path):
    """
    Reads and extracts text from a DOCX file.
    """
    try:
        doc = docx.Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        log_info(f"Successfully read DOCX: {file_path}")
        return text
    except Exception as e:
        log_error(f"Error reading DOCX {file_path}: {e}")
        return ""

def read_txt(file_path):
    """
    Reads and extracts text from a TXT file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        log_info(f"Successfully read TXT: {file_path}")
        return text
    except Exception as e:
        log_error(f"Error reading TXT {file_path}: {e}")
        return ""

def chunk_text(text, max_length=1000, chunk_overlap=100):
    """
    Splits text into chunks using CharacterTextSplitter, ensuring that chunk_overlap < max_length.
    """
    try:
        if chunk_overlap >= max_length:
            log_error(f"chunk_overlap ({chunk_overlap}) >= max_length ({max_length}). Adjusting chunk_overlap to {max_length - 1}.")
            chunk_overlap = max_length - 1  # Adjust to ensure overlap is less than chunk size

        splitter = CharacterTextSplitter(
            chunk_size=max_length,
            chunk_overlap=chunk_overlap,
            separator=". "  # Split on period-space to separate sentences
        )
        chunks = splitter.split_text(text)
        log_info(f"Successfully chunked text into {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        log_error(f"Error chunking text: {e}")
        return []

def process_files(directory, chunk_size=1000, chunk_overlap=100):
    """
    Processes supported files in a directory by reading and chunking their content.
    """
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
            else:
                text = ""
            if text:
                chunks = chunk_text(text, max_length=chunk_size, chunk_overlap=chunk_overlap)
                all_chunks.extend(chunks)
        log_info(f"Processed {len(all_chunks)} chunks from directory '{directory}'.")
        return all_chunks
    except Exception as e:
        log_error(f"Error processing files in directory '{directory}': {e}")
        return []