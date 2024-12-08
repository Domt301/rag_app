import pinecone
from config import (
    PINECONE_API_KEY,
    PINECONE_ENVIRONMENT,
    PINECONE_INDEX,
    OPENAI_API_KEY
)
from langchain_community.embeddings import OpenAIEmbeddings
from utils import log_error

def initialize_pinecone():
    """
    Initializes the Pinecone client and ensures that the specified index exists.
    
    Returns:
        pinecone.Index: An instance of the Pinecone index.
    """
    try:
        pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
        if PINECONE_INDEX not in pinecone.list_indexes():
            pinecone.create_index(PINECONE_INDEX, dimension=768)  # Adjust dimension as needed
            log_error(f"Created new Pinecone index: {PINECONE_INDEX}")
        index = pinecone.Index(PINECONE_INDEX)
        log_error(f"Pinecone index '{PINECONE_INDEX}' initialized successfully.")
        return index
    except Exception as e:
        log_error(f"Error initializing Pinecone: {e}")
        raise e

def get_embeddings():
    """
    Initializes the OpenAIEmbeddings instance with the provided OpenAI API key.
    
    Returns:
        OpenAIEmbeddings: An instance of OpenAIEmbeddings.
    """
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        log_error("OpenAIEmbeddings initialized successfully.")
        return embeddings
    except Exception as e:
        log_error(f"Error initializing OpenAIEmbeddings: {e}")
        raise e

def add_chunks_to_pinecone(index, chunks, embeddings, pbar=None):
    """
    Adds document chunks to the Pinecone index with their corresponding embeddings.
    
    Args:
        index (pinecone.Index): The Pinecone index instance.
        chunks (list of str): List of text chunks to be embedded and added.
        embeddings (OpenAIEmbeddings): The embeddings instance to generate embeddings.
        pbar (tqdm.tqdm, optional): Progress bar instance for visual feedback.
    """
    try:
        vectors = []
        embed = embeddings.embed_documents(chunks)
        for i, chunk in enumerate(chunks):
            vector = {
                "id": f"chunk-{i}",
                "values": embed[i],
                "metadata": {"text": chunk}
            }
            vectors.append(vector)
            if pbar:
                pbar.update(1)
        index.upsert(vectors)
        log_error(f"Added {len(vectors)} chunks to Pinecone successfully.")
    except Exception as e:
        log_error(f"Error adding chunks to Pinecone: {e}")
        raise e

def retrieve_chunks(index, query, embeddings, top_k=5):
    """
    Retrieves the top_k most relevant chunks from Pinecone based on the query.
    
    Args:
        index (pinecone.Index): The Pinecone index instance.
        query (str): The user's query string.
        embeddings (OpenAIEmbeddings): The embeddings instance to generate query embeddings.
        top_k (int, optional): Number of top results to retrieve. Defaults to 5.
    
    Returns:
        list of str: List of retrieved text chunks.
    """
    try:
        query_vector = embeddings.embed_query(query)
        results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
        retrieved_chunks = [match['metadata']['text'] for match in results['matches']]
        log_error(f"Retrieved {len(retrieved_chunks)} chunks from Pinecone for the query.")
        return retrieved_chunks
    except Exception as e:
        log_error(f"Error retrieving chunks from Pinecone: {e}")
        return []