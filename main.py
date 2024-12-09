from ui import get_user_input, prompt_add_documents, show_loading_message
from file_handler import process_files
from db_connector import initialize_pinecone, get_embeddings, add_chunks_to_pinecone, retrieve_chunks
from api_handler import create_rag_agent, generate_response_rag
from utils import display_progress, log_info, log_error, log_conversation
import os
import sys

def initialize_application():
    """
    Initialize Pinecone and embeddings.
    """
    try:
        index = initialize_pinecone()
        embeddings = get_embeddings()
        log_info("Initialized Pinecone and embeddings successfully.")
        return index, embeddings
    except Exception as e:
        log_error(f"Error initializing Pinecone: {e}")
        print(f"Error initializing Pinecone: {e}")
        sys.exit(1)

def process_documents(index, embeddings):
    """
    Process and add documents to the vector database.
    """
    add_docs = prompt_add_documents()
    if add_docs:
        directory = get_user_input("Enter the directory path containing your documents (or type 'exit' to quit): ", exit_message="Exiting document processing.")
        if os.path.isdir(directory):
            try:
                chunks = process_files(directory)
                if not chunks:
                    print("No valid content found in the directory.")
                    log_info("No chunks processed; directory may not contain valid files.")
                    return
                num_chunks = len(chunks)
                print(f"Processing {num_chunks} chunks...")
                pbar = display_progress(num_chunks, description="Adding chunks to Pinecone")
                add_chunks_to_pinecone(index, chunks, embeddings, pbar)
                pbar.close()
                log_info("Added chunks to Pinecone successfully.")
                print("All documents have been added to the vector database.")
            except Exception as e:
                log_error(f"Error processing files: {e}")
                print(f"Error processing files: {e}")
        else:
            log_error("Invalid directory path provided by user.")
            print("Invalid directory path.")

def check_documents_in_database(index):
    """
    Check if there are any documents in the Pinecone database.
    """
    try:
        stats = index.describe_index_stats()
        return stats['total_vector_count'] > 0
    except Exception as e:
        log_error(f"Error checking Pinecone database: {e}")
        return False

def start_query_loop(qa_chain):
    """
    Start the query-response loop with the RAG agent, retaining context.
    """
    print("RAG agent with LangChain is ready for use!")
    conversation_history = []  # To log conversation history
    while True:
        user_query = get_user_input("\nEnter your query (or type 'exit' to quit): ", exit_message="Exiting the application.")
        try:
            show_loading_message("Processing your query, please wait")
            response = generate_response_rag(qa_chain, user_query)
            print(f"Agent: {response}")
            log_conversation(user_query, response, conversation_history)
            conversation_history.append({"query": user_query, "response": response})
        except Exception as e:
            log_error(f"Error generating response: {e}")
            print(f"Error generating response: {e}")

def main():
    print("Welcome to the RAG Application!")
    
    # Initialization
    index, embeddings = initialize_application()
    
    # Document Processing
    process_documents(index, embeddings)
    documents_exist = check_documents_in_database(index)

    if not documents_exist:
        print("No documents found in the database. The AI will still respond, but answers may rely solely on general knowledge.")
        log_info("No documents in Pinecone; proceeding without document retrieval.")

    # Create RAG Agent
    pinecone_index = index if documents_exist else None
    try:
        qa_chain = create_rag_agent(pinecone_index, embeddings, model="gpt-4")
        log_info("RAG agent created successfully.")
    except Exception as e:
        log_error(f"Error creating RAG agent: {e}")
        print(f"Error creating RAG agent: {e}")
        sys.exit(1)

    # Query Loop
    start_query_loop(qa_chain)

if __name__ == "__main__":
    main()