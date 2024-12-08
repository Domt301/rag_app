import pinecone
from config import PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX, OPENAI_API_KEY
from langchain.embeddings.openai import OpenAIEmbeddings
from utils import log_error

def initialize_pinecone():
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    if PINECONE_INDEX not in pinecone.list_indexes():
        pinecone.create_index(PINECONE_INDEX, dimension=768)  # Adjust dimension as needed
    index = pinecone.Index(PINECONE_INDEX)
    return index

def get_embeddings():
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    return embeddings

def add_chunks_to_pinecone(index, chunks, embeddings, pbar=None):
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
    except Exception as e:
        log_error(f"Error adding chunks to Pinecone: {e}")
        raise e

def retrieve_chunks(index, query, embeddings, top_k=5):
    try:
        query_vector = embeddings.embed_query(query)
        results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
        retrieved_chunks = [match['metadata']['text'] for match in results['matches']]
        return retrieved_chunks
    except Exception as e:
        log_error(f"Error retrieving chunks from Pinecone: {e}")
        return []