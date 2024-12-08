from ui import get_user_input, prompt_add_documents
from file_handler import process_files
from db_connector import initialize_pinecone, get_embeddings, add_chunks_to_pinecone, retrieve_chunks
from api_handler import create_rag_agent, generate_response_rag
from utils import display_progress, log_info, log_error
import os
import sys

def main():
    print("Welcome to the RAG Application!")
    
    try:
        index = initialize_pinecone()
        embeddings = get_embeddings()
        log_info("Initialized Pinecone and embeddings successfully.")
    except Exception as e:
        log_error(f"Error initializing Pinecone: {e}")
        print(f"Error initializing Pinecone: {e}")
        sys.exit(1)

    add_docs = prompt_add_documents()

    if add_docs:
        directory = get_user_input("Enter the directory path containing your documents: ")
        if os.path.isdir(directory):
            try:
                chunks = process_files(directory)
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

    # Create the RAG agent
    try:
        qa_chain = create_rag_agent(index, embeddings)
        log_info("RAG agent created successfully.")
        print("RAG agent with LangChain is ready for use!")
    except Exception as e:
        log_error(f"Error creating RAG agent: {e}")
        print(f"Error creating RAG agent: {e}")
        sys.exit(1)

    # Start the conversation loop
    while True:
        user_query = get_user_input("\nEnter your query (or type 'exit' to quit): ")
        if user_query.lower() in ['exit', 'quit']:
            print("Goodbye!")
            log_info("User exited the application.")
            break
        try:
            response = generate_response_rag(qa_chain, user_query)
            print(f"Agent: {response}")
            log_info(f"User query: {user_query} | Agent response: {response}")
        except Exception as e:
            log_error(f"Error generating response: {e}")
            print(f"Error generating response: {e}")

if __name__ == "__main__":
    main()