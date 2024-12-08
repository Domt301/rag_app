# tests/test_api_handler.py

import pytest
from unittest.mock import patch, MagicMock
from api_handler import create_rag_agent, generate_response_rag


# Mocked constants and objects
MOCK_INDEX = MagicMock()
MOCK_EMBEDDINGS = MagicMock()
MOCK_QUERY = "What is the capital of France?"
MOCK_RESPONSE = "Paris is the capital of France."


@patch("api_handler.PineconeVectorStore")
@patch("api_handler.RetrievalQA")
@patch("api_handler.OpenAI")
def test_create_rag_agent_success(mock_openai, mock_retrievalqa, mock_pineconevectorstore):
    """Test successful creation of a RAG agent."""
    mock_openai.return_value = MagicMock()
    mock_retrievalqa.from_chain_type.return_value = MagicMock()

    result = create_rag_agent(MOCK_INDEX, MOCK_EMBEDDINGS)

    # Assertions
    mock_pineconevectorstore.assert_called_once_with(
        index=MOCK_INDEX,
        embedding=MOCK_EMBEDDINGS,
        text_key="text"
    )
    mock_retrievalqa.from_chain_type.assert_called_once()
    assert result == mock_retrievalqa.from_chain_type.return_value


def test_create_rag_agent_invalid_index():
    """Test creation of RAG agent with invalid index."""
    with pytest.raises(ValueError, match="Invalid Pinecone index provided."):
        create_rag_agent(None, MOCK_EMBEDDINGS)


def test_create_rag_agent_invalid_embeddings():
    """Test creation of RAG agent with invalid embeddings."""
    with pytest.raises(ValueError, match="Invalid embeddings object provided."):
        create_rag_agent(MOCK_INDEX, None)


@patch("api_handler.PineconeVectorStore")
@patch("api_handler.RetrievalQA")
@patch("api_handler.OpenAI")
def test_create_rag_agent_failure(mock_openai, mock_retrievalqa, mock_pineconevectorstore):
    """Test failure in creating a RAG agent."""
    mock_pineconevectorstore.side_effect = Exception("Initialization failed.")

    with pytest.raises(RuntimeError, match="Failed to create RAG agent."):
        create_rag_agent(MOCK_INDEX, MOCK_EMBEDDINGS)


@patch("api_handler.RetrievalQA")
def test_generate_response_rag_success(mock_retrievalqa):
    """Test successful response generation."""
    mock_chain = MagicMock()
    mock_chain.run.return_value = MOCK_RESPONSE

    result = generate_response_rag(mock_chain, MOCK_QUERY)

    # Assertions
    mock_chain.run.assert_called_once_with(MOCK_QUERY)
    assert result == MOCK_RESPONSE


def test_generate_response_rag_invalid_query():
    """Test response generation with an invalid query."""
    mock_chain = MagicMock()

    with pytest.raises(ValueError, match="The query must be a non-empty string."):
        generate_response_rag(mock_chain, "")


@patch("api_handler.RetrievalQA")
def test_generate_response_rag_failure(mock_retrievalqa):
    """Test response generation failure."""
    mock_chain = MagicMock()
    mock_chain.run.side_effect = Exception("Query processing failed.")

    result = generate_response_rag(mock_chain, MOCK_QUERY)

    # Assertions
    assert result == "I'm sorry, something went wrong. Please try again later."