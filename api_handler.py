import os
from config import OPENAI_API_KEY
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA, ConversationChain
from langchain.memory import ConversationSummaryMemory
from langchain_pinecone import PineconeVectorStore
from db_connector import retrieve_chunks, get_embeddings
from utils import log_error, log_info

def create_rag_agent(index, embeddings, model="gpt-4", return_sources=False):
    """
    Creates a Retrieval-Augmented Generation (RAG) agent with summary memory for context.

    Args:
        index (pinecone.Index or None): The Pinecone index instance or None for fallback.
        embeddings (OpenAIEmbeddings): The embeddings instance.
        model (str): The OpenAI model to use (default: "gpt-4").
        return_sources (bool): Whether to return source documents (default: False).

    Returns:
        ConversationChain: A conversation chain with or without retrieval capabilities.
    """
    if index and not hasattr(index, "query"):
        raise ValueError("Invalid Pinecone index provided.")
    if not embeddings or not hasattr(embeddings, "embed_query"):
        raise ValueError("Invalid embeddings object provided.")

    try:
        log_info(f"Creating RAG agent with model='{model}', return_sources={return_sources}.")

        llm = OpenAI(model=model, openai_api_key=OPENAI_API_KEY)

        if index:
            # Retrieval-based RAG agent
            vector_store = PineconeVectorStore(
                index=index,
                embedding=embeddings,
                text_key="text"
            )
            log_info(f"Initialized Pinecone vector store successfully for index: {index.name}.")

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(),
                return_source_documents=return_sources
            )
            log_info("RetrievalQA chain created successfully.")
        else:
            # Fallback to conversation-only agent with memory
            memory = ConversationSummaryMemory(llm=llm)  # Initialize memory only for fallback
            log_info("No index provided; creating a conversation-only agent with summary memory.")
            qa_chain = ConversationChain(llm=llm, memory=memory)

        return qa_chain
    except Exception as e:
        log_error(f"Error creating RAG agent: {e}")
        raise RuntimeError("Failed to create RAG agent.") from e
def generate_response_rag(qa_chain, query):
    """
    Generates a response from the RAG agent based on the user's query.

    Args:
        qa_chain (RetrievalQA or ConversationChain): The chain instance with or without retrieval.
        query (str): The user's query.

    Returns:
        str: The generated response.

    Raises:
        ValueError: If query is invalid.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("The query must be a non-empty string.")

    try:
        response = qa_chain.run({"query": query})
        log_info(f"Generated response for query: '{query}'")
        return response
    except Exception as e:
        log_error(f"Error generating response for query '{query}': {e}")
        return "I'm sorry, something went wrong. Please try again later."