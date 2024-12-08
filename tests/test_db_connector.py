import unittest
from unittest.mock import patch, MagicMock
from db_connector import initialize_pinecone, get_embeddings, add_chunks_to_pinecone, retrieve_chunks

class TestDBConnector(unittest.TestCase):

    @patch('db_connector.pinecone.init')
    @patch('db_connector.pinecone.list_indexes')
    @patch('db_connector.pinecone.create_index')
    @patch('db_connector.pinecone.Index')
    def test_initialize_pinecone(self, mock_index, mock_create_index, mock_list_indexes, mock_init):
        mock_list_indexes.return_value = []
        index = initialize_pinecone()
        mock_init.assert_called_once()
        mock_create_index.assert_called_once()
        mock_index.assert_called_once()
        self.assertIsNotNone(index)

    @patch('db_connector.OpenAIEmbeddings')
    def test_get_embeddings(self, mock_embeddings):
        mock_instance = MagicMock()
        mock_embeddings.return_value = mock_instance
        embeddings = get_embeddings()
        mock_embeddings.assert_called_once()
        self.assertEqual(embeddings, mock_instance)

    @patch('db_connector.pinecone.Index')
    def test_retrieve_chunks(self, mock_index):
        mock_embeddings = MagicMock()
        mock_embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_index.query.return_value = {
            'matches': [
                {'metadata': {'text': 'Sample chunk 1'}},
                {'metadata': {'text': 'Sample chunk 2'}}
            ]
        }
        result = retrieve_chunks(mock_index, "Test query", mock_embeddings, top_k=2)
        self.assertEqual(result, ['Sample chunk 1', 'Sample chunk 2'])

if __name__ == '__main__':
    unittest.main()