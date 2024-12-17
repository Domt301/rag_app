import pytest
from unittest.mock import patch, MagicMock
from api_handler import create_rag_agent, generate_response_rag
from langchain.chains import RetrievalQA, LLMChain

# Mocked constants and objects
MOCK_INDEX = MagicMock()
MOCK_EMBEDDINGS = MagicMock()
MOCK_QUERY = "What is the capital of France?"
MOCK_RESPONSE = "Paris is the capital of France."


@patch("api_handler.Pinecone")
@patch("api_handler.RetrievalQA")
@patch("api_handler.ConversationBufferMemory")
@patch("api_handler.ChatOpenAI")
def test_create_rag_agent_with_retrieval(
    mock_chatopenai, mock_memory, mock_retrievalqa, mock_pinecone
):
    """Test successful creation of a RAG agent with retrieval."""
    mock_chatopenai.return_value = MagicMock()
    mock_retrievalqa.from_chain_type.return_value = MagicMock()

    result = create_rag_agent(MOCK_INDEX, MOCK_EMBEDDINGS)

    # Assertions
    mock_pinecone.assert_called_once_with(
        index=MOCK_INDEX,
        embedding=MOCK_EMBEDDINGS,
        text_key="text"
    )
    mock_retrievalqa.from_chain_type.assert_called_once()
    mock_memory.assert_not_called()  # Ensure ConversationBufferMemory is not used in retrieval mode
    assert result == mock_retrievalqa.from_chain_type.return_value


@patch("api_handler.LLMChain")
@patch("api_handler.ConversationBufferMemory")
@patch("api_handler.ChatOpenAI")
def test_create_rag_agent_without_retrieval(
    mock_chatopenai, mock_memory, mock_llmchain
):
    """Test successful creation of a conversation-only RAG agent without retrieval."""
    mock_chatopenai.return_value = MagicMock()
    mock_llmchain.return_value = MagicMock()
    mock_memory.return_value = MagicMock()

    result = create_rag_agent(None, MOCK_EMBEDDINGS)

    # Assertions
    mock_memory.assert_called_once_with(memory_key="history", return_messages=True)
    mock_llmchain.assert_called_once()
    assert result == mock_llmchain.return_value


def test_create_rag_agent_invalid_index():
    """Test creation of RAG agent with invalid index."""
    with pytest.raises(ValueError, match="Invalid embeddings object provided."):
        create_rag_agent(None, None)


def test_create_rag_agent_invalid_embeddings():
    """Test creation of RAG agent with invalid embeddings."""
    with pytest.raises(ValueError, match="Invalid embeddings object provided."):
        create_rag_agent(MOCK_INDEX, None)


@patch("api_handler.Pinecone")
@patch("api_handler.RetrievalQA")
def test_create_rag_agent_failure(mock_retrievalqa, mock_pinecone):
    """Test failure in creating a RAG agent."""
    mock_pinecone.side_effect = Exception("Initialization failed.")

    with pytest.raises(RuntimeError, match="Failed to create RAG agent."):
        create_rag_agent(MOCK_INDEX, MOCK_EMBEDDINGS)







def test_generate_response_rag_invalid_query():
    """Test response generation with an invalid query."""
    mock_chain = MagicMock()

    with pytest.raises(ValueError, match="The query must be a non-empty string."):
        generate_response_rag(mock_chain, "")


@patch("api_handler.RetrievalQA")
def test_generate_response_rag_failure(mock_retrievalqa):
    """Test response generation failure."""
    # Mock RetrievalQA chain
    mock_chain = MagicMock(spec=RetrievalQA)
    mock_chain.invoke.side_effect = Exception("Query processing failed.")

    # Test the function
    result = generate_response_rag(mock_chain, MOCK_QUERY)

    # Assertions
    assert result == "I'm sorry, something went wrong. Please try again later."