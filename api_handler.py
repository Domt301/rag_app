from config import OPENAI_API_KEY
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Pinecone
from db_connector import retrieve_chunks, get_embeddings
from utils import log_error, log_info

def create_rag_agent(index, embeddings):
    """
    Creates a Retrieval-Augmented Generation (RAG) agent using LangChain's RetrievalQA.

    Args:
        index (pinecone.Index): The Pinecone index instance.
        embeddings (OpenAIEmbeddings): The embeddings instance.

    Returns:
        RetrievalQA: An instance of the RetrievalQA chain.
    """
    try:
        # Initialize LangChain's Pinecone vector store
        vector_store = Pinecone(
            index=index,
            embedding_function=embeddings.embed_query,
            text_key="text"
        )
        log_info("Initialized Pinecone vector store successfully.")

        # Create a RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=OpenAI(model="gpt-4", openai_api_key=OPENAI_API_KEY),
            chain_type="stuff",
            retriever=vector_store.as_retriever(),
            return_source_documents=False
        )
        log_info("RetrievalQA chain created successfully.")
        return qa_chain
    except Exception as e:
        log_error(f"Error creating RAG agent: {e}")
        raise e

def generate_response_rag(qa_chain, query):
    """
    Generates a response from the RAG agent based on the user's query.

    Args:
        qa_chain (RetrievalQA): The RetrievalQA chain instance.
        query (str): The user's query.

    Returns:
        str: The generated response.
    """
    try:
        response = qa_chain.run(query)
        log_info(f"Generated response for query: '{query}'")
        return response
    except Exception as e:
        log_error(f"Error generating response for query '{query}': {e}")
        return "I'm sorry, I couldn't generate a response at this time."