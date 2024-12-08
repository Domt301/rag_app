import openai
from config import OPENAI_API_KEY
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
from db_connector import retrieve_chunks, get_embeddings
import pinecone

openai.api_key = OPENAI_API_KEY

def create_rag_agent(index, embeddings):
    # Initialize LangChain's Pinecone vector store
    vector_store = Pinecone(index, embeddings.embed_query, "text")

    # Create a RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(model="gpt-4", openai_api_key=OPENAI_API_KEY),
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=False
    )
    return qa_chain

def generate_response_rag(qa_chain, query):
    response = qa_chain.run(query)
    return response