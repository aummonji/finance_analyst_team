
"""
Simple investment knowledge RAG using Chroma.
Uses OpenAI embeddings.
"""

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import tool
import logging

logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()



def _load_vector_store():
    """Load the investment principles into a vector store."""
    from pathlib import Path

 # Get path relative to this file's location
    file_path = Path(__file__).parent.parent / "investment_principles.txt"
   
    loader = TextLoader(str(file_path), encoding='utf-8')
    docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)
    
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"  
    )
    
    return Chroma.from_documents(chunks, embeddings) 
# Takes each chunk
# Converts to embedding using embeddings
# Stores in vector database (in-memory by default)
# Returns a Chroma object you can search


@tool
def get_investment_advice(query: str) -> str:
    """
    Get investment advice from knowledge base.
    
    Args:
        query: What to search for (e.g., "crypto allocation aggressive",
               "tech sector diversification", "conservative bonds")
    
    Returns:
        Relevant investment principles
    """
    try:
        vector_store = _load_vector_store()
        results = vector_store.similarity_search(query, k=2)
         # Finds chunks with most similar embeddings
        # Returns top k=2 matches

        advice = "INVESTMENT PRINCIPLES:\n\n"
        for doc in results:
            advice += doc.page_content + "\n\n"
                    
        return advice
    
        # Building response:
            # 1. Start with header: `"INVESTMENT PRINCIPLES:\n\n"`
            # 2. For each retrieved chunk (doc):
            #    - Add chunk text: `doc.page_content`
            #    - Add spacing: `\n\n`
            # 3. Return combined string

    except Exception as e:
        logger.error(f"Error retrieving advice: {e}")
        return "Error retrieving investment advice."