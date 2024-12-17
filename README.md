RAG Application README

Overview

This application is a Retrieval-Augmented Generation (RAG) system that combines document retrieval with generative AI to provide intelligent, context-aware responses. The app processes user documents, stores their embeddings in a vector database (Pinecone), and integrates with OpenAI’s models for answering user queries.

Prerequisites

Before running the application, ensure you have the following:
	1.	Python Installed (Python 3.8 or higher recommended)
	2.	API Keys and Environment Setup:
	•	Pinecone API Key and related environment details
	•	OpenAI API Key
	3.	Required Python Libraries

Getting Started

1. Set Up Pinecone

a. Create a Pinecone Account
	1.	Go to Pinecone’s website and sign up or log in.
	2.	Navigate to the API Keys section in the dashboard.

b. Retrieve Pinecone API Key
	1.	In the Pinecone dashboard, find your API key.
	2.	Copy it to use in your .env file.

c. Retrieve Pinecone Environment
	1.	In the API Keys section, you’ll also see your environment (e.g., us-west1-gcp).
	2.	Copy it for use in the .env file.

d. Define or Create a Pinecone Index
	1.	Navigate to the Indexes tab in Pinecone.
	2.	Create a new index with a name (e.g., rag-index) and set the dimension to 768 (compatible with OpenAI embeddings).
	3.	Note the index name for your .env file.

2. Set Up OpenAI API Key

a. Create an OpenAI Account
	1.	Go to OpenAI’s website and sign up or log in.
	2.	Navigate to the API Keys section in the dashboard.

b. Retrieve Your OpenAI API Key
	1.	Click Create New Secret Key.
	2.	Copy the generated key for use in your .env file.

3. Clone the Repository

git clone https://github.com/your-repository/rag-application.git
cd rag-application

4. Install Required Python Libraries

pip install -r requirements.txt

5. Create a .env File

In the root directory of the project, create a file named .env and populate it with the following:

OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX=your_pinecone_index_name

Replace the placeholders (your_openai_api_key, etc.) with the respective values obtained earlier.

Running the Application
	1.	Start the application by running the main.py file:

python main.py


	2.	Follow the on-screen prompts to:
	•	Add documents to the database.
	•	Query the system.
	pass a directory to the app like so
	/Users/your_username/Documents/my_files

Features

1. Document Processing
	•	Upload documents (PDF, DOCX, or TXT) to the vector database.
	•	Documents are chunked, embedded using OpenAI’s embeddings, and stored in Pinecone for fast retrieval.

2. Question Answering
	•	Ask the system questions based on uploaded documents.
	•	If no documents are available, fallback mode uses OpenAI’s general knowledge.

3. Conversation History
	•	Maintains a history of interactions within the session.
	•	Optionally truncates or clears history to avoid confusion.

Configuration Options

Modify Pinecone or OpenAI Model Settings
	•	Change the model used (e.g., gpt-3.5-turbo or gpt-4) in config.py.
	•	Adjust chunk size and overlap in file_handler.py to optimize document processing.

Troubleshooting

Common Issues
	1.	Invalid Pinecone API Key or Environment
	•	Ensure your .env file contains the correct PINECONE_API_KEY
	•	Double-check that your Pinecone index exists.
	2.	OpenAI API Errors
	•	Verify your OPENAI_API_KEY in .env.
	•	Ensure your OpenAI account has sufficient quota.
	3.	File Processing Errors
	•	Ensure the files are in a supported format (PDF, DOCX, or TXT).
	•	Check the directory path for validity.

Key Files

1. main.py
	•	Entry point for the application.
	•	Manages initialization, document processing, and query loop.

2. file_handler.py
	•	Handles document reading, chunking, and preparation for embedding.

3. api_handler.py
	•	Creates the RAG agent and generates responses.

4. db_connector.py
	•	Connects to Pinecone for storing and retrieving embeddings.

5. ui.py
	•	Handles user interaction and input.

6. utils.py
	•	Provides logging, progress display, and utility functions.

7. .env
	•	Stores API keys and configuration details.

Future Enhancements
	•	Persist conversation history across sessions.
	•	Add support for more file types (e.g., CSV, JSON).
	•	Implement advanced query analytics and result scoring.

License

This project is licensed under the MIT License.

Support

If you encounter issues or have questions, feel free to open an issue in the repository or contact the maintainer.