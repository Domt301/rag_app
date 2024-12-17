import unittest
from unittest.mock import patch, MagicMock
from db_connector import initialize_pinecone, get_embeddings, add_chunks_to_pinecone, retrieve_chunks

class TestDBConnector(unittest.TestCase):

    @patch('db_connector.Pinecone.Index')
    @patch('db_connector.Pinecone.create_index')
    @patch('db_connector.Pinecone.list_indexes')
    def test_initialize_pinecone(self, mock_list_indexes, mock_create_index, mock_index):
        # Mock Pinecone responses
        mock_list_indexes.return_value.names.return_value = []
        mock_create_index.return_value = None
        mock_index.return_value = MagicMock()

        # Call the function
        index = initialize_pinecone()

        # Assertions
        mock_list_indexes.assert_called_once()
        # mock_create_index.assert_called_once_with(
        #     name='your_index_name',  # Replace 'your_index_name' with your test PINECONE_INDEX
        #     dimension=768,
        #     metric='cosine'
        # )
        # mock_index.assert_called_once_with(name='your_index_name')
        self.assertIsNotNone(index)

    @patch('db_connector.OpenAIEmbeddings')
    def test_get_embeddings(self, mock_embeddings):
        # Mock OpenAIEmbeddings instance
        mock_instance = MagicMock()
        mock_embeddings.return_value = mock_instance

        # Call the function
        embeddings = get_embeddings()

        # Assertions
        mock_embeddings.assert_called_once()
        self.assertEqual(embeddings, mock_instance)

    @patch('db_connector.Pinecone.Index')
    @patch('db_connector.OpenAIEmbeddings')
    def test_add_chunks_to_pinecone(self, mock_embeddings, mock_index):
        # Mock embeddings and index
        mock_embeddings_instance = MagicMock()
        mock_embeddings_instance.embed_documents.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_embeddings.return_value = mock_embeddings_instance

        mock_index_instance = MagicMock()
        mock_index.return_value = mock_index_instance

        chunks = ["Chunk 1", "Chunk 2"]

        # Call the function
        add_chunks_to_pinecone(mock_index_instance, chunks, mock_embeddings_instance)

        # Assertions
        mock_embeddings_instance.embed_documents.assert_called_once_with(chunks)
        mock_index_instance.upsert.assert_called_once_with([
            {"id": "chunk-0", "values": [0.1, 0.2, 0.3], "metadata": {"text": "Chunk 1"}},
            {"id": "chunk-1", "values": [0.4, 0.5, 0.6], "metadata": {"text": "Chunk 2"}}
        ])

    @patch('db_connector.Pinecone.Index')
    def test_retrieve_chunks(self, mock_index):
    # Mock embeddings and query results
        mock_embeddings = MagicMock()
        mock_embeddings.embed_query.return_value = [0.1, 0.2, 0.3]

        mock_index_instance = MagicMock()
        mock_index_instance.query.return_value = MagicMock(
            matches=[
                {'metadata': {'text': 'Sample chunk 1'}},
                {'metadata': {'text': 'Sample chunk 2'}}
            ]
        )
        mock_index.return_value = mock_index_instance

    # Call the function
        result = retrieve_chunks(mock_index_instance, "Test query", mock_embeddings, top_k=2)

    # Assertions
        mock_embeddings.embed_query.assert_called_once_with("Test query")
        mock_index_instance.query.assert_called_once_with(
            vector=[0.1, 0.2, 0.3], top_k=2, include_metadata=True
        )
        self.assertEqual(result, ['Sample chunk 1', 'Sample chunk 2'])

if __name__ == '__main__':
    unittest.main()