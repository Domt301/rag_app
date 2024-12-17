import os
from config import OPENAI_API_KEY, PINECONE_INDEX
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from db_connector import retrieve_chunks, get_embeddings
from utils import log_error, log_info
from ui import show_loading_message


def create_rag_agent(index, embeddings, model="gpt-4", return_sources=False):
    """
    Creates a Retrieval-Augmented Generation (RAG) agent with buffer memory for context.

    Args:
        index (pinecone.Index or None): The Pinecone index instance or None for fallback.
        embeddings (OpenAIEmbeddings): The embeddings instance.
        model (str): The OpenAI model to use (default: "gpt-4").
        return_sources (bool): Whether to return source documents (default: False).

    Returns:
        RetrievalQA or LLMChain: A chain with retrieval capabilities or memory-based fallback.
    """
    if index and not hasattr(index, "query"):
        raise ValueError("Invalid Pinecone index provided.")
    if not embeddings or not hasattr(embeddings, "embed_query"):
        raise ValueError("Invalid embeddings object provided.")

    try:
        log_info(f"Creating RAG agent with model='{model}', return_sources={return_sources}.")

        # Initialize ChatOpenAI
        llm = ChatOpenAI(model=model, openai_api_key=OPENAI_API_KEY)

        if index:
            # Retrieval-based RAG agent
            vector_store = Pinecone(
                index=index,
                embedding=embeddings,
                text_key="text"  # Ensure this matches the metadata key in Pinecone
            )
            log_info(f"Initialized Pinecone vector store successfully for index: '{PINECONE_INDEX}'.")

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(),
                return_source_documents=return_sources
            )
            log_info("RetrievalQA chain created successfully.")
        else:
            # Fallback to memory-based conversation with LLMChain and ConversationBufferMemory
            log_info("No Pinecone index detected; defaulting to memory-based responses.")

            # Prompt template for fallback responses
            prompt = PromptTemplate(
                input_variables=["history", "input"],
                template="""
                You are a helpful AI assistant. Use the conversation history below to maintain context.
                Conversation history: {history}
                User: {input}
                AI:"""
            )
            memory = ConversationBufferMemory(memory_key="history", return_messages=True)
            qa_chain = LLMChain(prompt=prompt, llm=llm, memory=memory)

        return qa_chain
    except Exception as e:
        log_error(f"Error creating RAG agent: {e}")
        raise RuntimeError("Failed to create RAG agent.") from e


def generate_response_rag(qa_chain, query):
    """
    Generates a response from the RAG agent based on the user's query.

    Args:
        qa_chain (RetrievalQA or LLMChain): The chain instance with or without retrieval.
        query (str): The user's query.

    Returns:
        str: The generated response.

    Raises:
        ValueError: If query is invalid.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("The query must be a non-empty string.")

    try:
        show_loading_message("Processing your query, please wait")
        
        # Check the type of qa_chain and use the correct input key
        if isinstance(qa_chain, RetrievalQA):
            response = qa_chain.invoke({"query": query})  # Use 'query' for RetrievalQA
        elif isinstance(qa_chain, LLMChain):
            response = qa_chain.invoke({"input": query})  # Use 'input' for LLMChain
        else:
            raise ValueError("Unsupported chain type provided to generate_response_rag.")

        log_info(f"Generated response for query: '{query}'")
        return response
    except Exception as e:
        log_error(f"Error generating response for query '{query}': {e}")
        return "I'm sorry, something went wrong. Please try again later."
   