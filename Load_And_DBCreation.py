import os
import shutil
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from logger import setup_logger

logger = setup_logger(__name__)

DIRECTORY_PATH = "./Knowledge-base" 
CHROMA_PATH = "./chroma_db"


def load_document():
    """
    Loads text documents from the defined directory.
    """
    try:
        logger.info(f"Starting document loading from: {DIRECTORY_PATH}")
        
        if not os.path.exists(DIRECTORY_PATH):
            logger.error(f"Directory path does not exist: {DIRECTORY_PATH}")
            return None

        loader = DirectoryLoader(DIRECTORY_PATH, glob="**/*.md", loader_cls=TextLoader)
        documents = loader.load()

        if not documents:
            logger.warning("No documents were found in the specified directory.")
            return None
            
        logger.info(f"Successfully loaded {len(documents)} documents.")
        return documents

    except Exception as e:
        logger.error(f"Failed to load documents. Error: {str(e)}", exc_info=True)
        return None


def chunk_documents(documents):
    """
    Splits loaded documents into smaller chunks.
    """
    try:
        if not documents:
            logger.warning("Skipping chunking process because no documents were provided.")
            return []

        logger.info("Starting text splitting process.")
        
        text_split = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = text_split.split_documents(documents)
        logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks

    except Exception as e:
        logger.error(f"Failed to chunk documents. Error: {str(e)}", exc_info=True)
        return []


def create_vector_db(chunks):
    
    try:
        if not chunks:
            logger.warning("Skipping database creation because no chunks are available.")
            return None

        if os.path.exists(CHROMA_PATH):
            logger.info(f"Removing existing database at {CHROMA_PATH}")
            shutil.rmtree(CHROMA_PATH)
        
        logger.info("Initializing HuggingFace embeddings model.")
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        logger.info("Creating Chroma database from documents.")
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=CHROMA_PATH
        )
        
        logger.info(f"Database successfully saved to {CHROMA_PATH}")
        return db

    except Exception as e:
        logger.error(f"Failed to create vector database. Error: {str(e)}", exc_info=True)
        return None


if __name__ == "__main__":
    logger.info("--- Pipeline Execution Started ---")
    
    my_doc = load_document()
    
    if my_doc:
        my_chunk = chunk_documents(my_doc)
        
        if my_chunk:
            vector_db = create_vector_db(my_chunk)

            if vector_db:
                logger.info("Executing Test Query: 'What are the core hours?'")
                try:
                    query = "What are the core hours?"
                    results = vector_db.similarity_search(query, k=1)
                    
                    if results:
                        logger.info(f"Query Result: {results[0].page_content[:100]}...")
                    else:
                        logger.warning("Query returned no results.")
                        
                except Exception as e:
                    logger.error(f"Test query failed. Error: {str(e)}", exc_info=True)
    
    logger.info("--- Pipeline Execution Finished ---")