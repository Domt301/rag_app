# tests/test_main.py

import pytest
from unittest import mock
from unittest.mock import patch, MagicMock

# Helper function to mock input() calls
def mock_inputs(inputs):
    return patch('builtins.input', side_effect=inputs)

# Helper function to mock sys.exit to prevent the test runner from exiting
def mock_sys_exit():
    return patch('sys.exit')

# Test when user chooses to add documents and everything works fine
def test_main_add_documents_success():
    with mock_inputs([
        'yes',                          # User chooses to add documents
        '/valid/directory/path',        # User provides a valid directory
        'What is the capital of France?',  # User enters a query
        'exit'                          # User decides to exit after one query
    ]), patch('os.path.isdir', return_value=True), \
       patch('file_handler.process_files', return_value=['chunk1', 'chunk2']), \
       patch('main.initialize_pinecone') as mock_init_pinecone, \
       patch('main.get_embeddings') as mock_get_embeddings, \
       patch('main.add_chunks_to_pinecone') as mock_add_chunks, \
       patch('main.create_rag_agent') as mock_create_rag, \
       patch('main.generate_response_rag', return_value='Paris is the capital of France.') as mock_generate_response_rag, \
       patch('main.display_progress') as mock_display_progress, \
       patch('main.log_info') as mock_log_info, \
       patch('main.log_error') as mock_log_error, \
       mock_sys_exit() as mock_exit:
        
        # Import main after patches
        import main

        # Run main
        main.main()

        # Assertions
        mock_init_pinecone.assert_called_once()
        mock_get_embeddings.assert_called_once()
        mock_display_progress.assert_called_once_with(2, description="Adding chunks to Pinecone")
        mock_add_chunks.assert_called_once_with(
            mock_init_pinecone.return_value, 
            ['chunk1', 'chunk2'], 
            mock_get_embeddings.return_value, 
            mock_display_progress.return_value
        )
        mock_create_rag.assert_called_once_with(
            mock_init_pinecone.return_value, 
            mock_get_embeddings.return_value
        )
        mock_generate_response_rag.assert_called_once_with(
            mock_create_rag.return_value, 
            'What is the capital of France?'
        )
        mock_log_info.assert_any_call("Initialized Pinecone and embeddings successfully.")
        mock_log_info.assert_any_call("Added chunks to Pinecone successfully.")
        mock_log_info.assert_any_call("RAG agent created successfully.")
        mock_log_info.assert_any_call("User exited the application.")
        mock_log_error.assert_not_called()
        mock_exit.assert_not_called()

# Test when user chooses not to add documents and starts conversation immediately
def test_main_skip_add_documents():
    with mock_inputs([
        'no',                               # User chooses not to add documents
        'Tell me about the Eiffel Tower.',  # User enters a query
        'exit'                              # User decides to exit
    ]), \
       patch('main.create_rag_agent') as mock_create_rag, \
       patch('main.generate_response_rag', return_value='The Eiffel Tower is located in Paris.') as mock_generate_response_rag, \
       patch('main.log_info') as mock_log_info, \
       patch('main.log_error') as mock_log_error, \
       mock_sys_exit() as mock_exit:
        
        # Mock the dependencies that are not called when add_docs is False
        with patch('main.initialize_pinecone') as mock_init_pinecone, \
             patch('main.get_embeddings') as mock_get_embeddings:
            
            # Import main after patches
            import main

            # Run main
            main.main()

            # Assertions
            mock_init_pinecone.assert_called_once()
            mock_get_embeddings.assert_called_once()
            mock_create_rag.assert_called_once_with(
                mock_init_pinecone.return_value, 
                mock_get_embeddings.return_value
            )
            mock_generate_response_rag.assert_called_once_with(
                mock_create_rag.return_value, 
                'Tell me about the Eiffel Tower.'
            )
            mock_log_info.assert_any_call("Initialized Pinecone and embeddings successfully.")
            mock_log_info.assert_any_call("RAG agent created successfully.")
            mock_log_info.assert_any_call("User exited the application.")
            mock_log_error.assert_not_called()
            mock_exit.assert_not_called()

# Test when user provides an invalid directory path
def test_main_invalid_directory():
    with mock_inputs([
        'yes',                          # User chooses to add documents
        '/invalid/directory/path',      # User provides an invalid directory
        'What is the tallest mountain?',# User enters a query
        'exit'                          # User decides to exit after one query
    ]), patch('os.path.isdir', return_value=False), \
       patch('main.log_error') as mock_log_error, \
       patch('main.log_info') as mock_log_info, \
       patch('main.initialize_pinecone') as mock_init_pinecone, \
       patch('main.get_embeddings') as mock_get_embeddings, \
       patch('main.create_rag_agent') as mock_create_rag, \
       patch('main.generate_response_rag', return_value='Mount Everest is the tallest mountain.') as mock_generate_response_rag, \
       mock_sys_exit() as mock_exit:
        
        # Import main after patches
        import main

        # Run main
        main.main()

        # Assertions
        mock_init_pinecone.assert_called_once()
        mock_get_embeddings.assert_called_once()
        mock_create_rag.assert_called_once_with(
            mock_init_pinecone.return_value, 
            mock_get_embeddings.return_value
        )
        mock_generate_response_rag.assert_called_once_with(
            mock_create_rag.return_value, 
            'What is the tallest mountain?'
        )
        mock_log_info.assert_any_call("Initialized Pinecone and embeddings successfully.")
        mock_log_error.assert_any_call("Invalid directory path provided by user.")
        mock_log_info.assert_any_call("RAG agent created successfully.")
        mock_log_info.assert_any_call("User exited the application.")
        mock_exit.assert_not_called()

# Test when Pinecone initialization fails
# def test_main_pinecone_initialization_failure():
#     with patch('main.initialize_pinecone', side_effect=Exception("Pinecone initialization failed")), \
#          patch('main.log_error') as mock_log_error, \
#          patch('main.log_info') as mock_log_info, \
#          mock_sys_exit() as mock_exit:
        
#         # Import main after patches
#         import main

#         # Run main
#         with pytest.raises(SystemExit) as pytest_wrapped_e:
#             main.main()

#         # Assertions
#         mock_log_error.assert_any_call("Error initializing Pinecone: Pinecone initialization failed")
#         mock_exit.assert_called_once_with(1)
#         assert pytest_wrapped_e.value.code == 1

# Test when RAG agent creation fails
def test_main_rag_agent_creation_failure():
    with mock_inputs([
        'no',                               # User chooses not to add documents
        'Describe the Great Wall of China.',# User enters a query
        'exit'                              # User decides to exit
    ]), \
       patch('main.create_rag_agent', side_effect=Exception("RAG agent creation failed")) as mock_create_rag, \
       patch('main.log_error') as mock_log_error, \
       patch('main.log_info') as mock_log_info, \
       patch('main.initialize_pinecone') as mock_init_pinecone, \
       patch('main.get_embeddings') as mock_get_embeddings, \
       patch('main.generate_response_rag', return_value='The Great Wall of China is a series of fortifications.') as mock_generate_response_rag, \
       mock_sys_exit() as mock_exit:
        
        # Import main after patches
        import main

        # Run main
        main.main()

        # Assertions
        mock_init_pinecone.assert_called_once()
        mock_get_embeddings.assert_called_once()
        mock_create_rag.assert_called_once_with(
            mock_init_pinecone.return_value, 
            mock_get_embeddings.return_value
        )
        mock_log_error.assert_any_call("Error creating RAG agent: RAG agent creation failed")
        mock_exit.assert_called_once_with(1)

# Test user exits application immediately
def test_main_user_exits_immediately():
    with mock_inputs([
        'no',       # User chooses not to add documents
        'exit'      # User decides to exit immediately
    ]), \
       patch('main.create_rag_agent') as mock_create_rag, \
       patch('main.generate_response_rag', return_value='Mocked response') as mock_generate_response_rag, \
       patch('main.log_info') as mock_log_info, \
       patch('main.log_error') as mock_log_error, \
       mock_sys_exit() as mock_exit:
        
        # Mock the dependencies that are not called when add_docs is False
        with patch('main.initialize_pinecone') as mock_init_pinecone, \
             patch('main.get_embeddings') as mock_get_embeddings:
            
            # Import main after patches
            import main

            # Run main
            main.main()

            # Assertions
            mock_init_pinecone.assert_called_once()
            mock_get_embeddings.assert_called_once()
            mock_create_rag.assert_called_once_with(
                mock_init_pinecone.return_value, 
                mock_get_embeddings.return_value
            )
            mock_generate_response_rag.assert_not_called()  # No query entered before exit
            mock_log_info.assert_any_call("Initialized Pinecone and embeddings successfully.")
            mock_log_info.assert_any_call("RAG agent created successfully.")
            mock_log_info.assert_any_call("User exited the application.")
            mock_log_error.assert_not_called()
            mock_exit.assert_not_called()

# Test when an error occurs during adding chunks to Pinecone
def test_main_error_adding_chunks():
    with mock_inputs([
        'yes',                          # User chooses to add documents
        '/valid/directory/path',        # User provides a valid directory
        'What is the largest ocean?',   # User enters a query
        'exit'                          # User decides to exit after one query
    ]), patch('os.path.isdir', return_value=True), \
       patch('file_handler.process_files', return_value=['chunk1', 'chunk2']), \
       patch('main.initialize_pinecone') as mock_init_pinecone, \
       patch('main.get_embeddings') as mock_get_embeddings, \
       patch('main.add_chunks_to_pinecone', side_effect=Exception("Failed to add chunks")) as mock_add_chunks, \
       patch('main.create_rag_agent') as mock_create_rag, \
       patch('main.generate_response_rag', return_value='The Pacific Ocean is the largest.') as mock_generate_response_rag, \
       patch('main.display_progress') as mock_display_progress, \
       patch('main.log_info') as mock_log_info, \
       patch('main.log_error') as mock_log_error, \
       mock_sys_exit() as mock_exit:
        
        # Import main after patches
        import main

        # Run main
        main.main()

        # Assertions
        mock_init_pinecone.assert_called_once()
        mock_get_embeddings.assert_called_once()
        mock_display_progress.assert_called_once_with(2, description="Adding chunks to Pinecone")
        mock_add_chunks.assert_called_once_with(
            mock_init_pinecone.return_value, 
            ['chunk1', 'chunk2'], 
            mock_get_embeddings.return_value, 
            mock_display_progress.return_value
        )
        mock_log_info.assert_any_call("Initialized Pinecone and embeddings successfully.")
        mock_log_error.assert_any_call("Error processing files: Failed to add chunks")
        mock_exit.assert_not_called()