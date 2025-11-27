import os
from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain.retrievers import EnsembleRetriever
    from langchain.chains import RetrievalQA
except ImportError:
    from langchain_classic.retrievers import EnsembleRetriever
    from langchain_classic.chains import RetrievalQA

from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from logger import setup_logger

load_dotenv()

logger = setup_logger(__name__)

CHROMA_PATH = os.getenv("CHROMA_PATH")
DATA_PATH = os.getenv("DATA_PATH")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def setup_hybrid_chain():
    try:
        logger.info("Initializing Hybrid Search Engine.")

        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_db = Chroma(
            persist_directory=CHROMA_PATH, 
            embedding_function=embedding_model
        )
        vector_retriever = vector_db.as_retriever(search_kwargs={"k": 5})
        logger.info("Vector store loaded successfully.")

        logger.info("Building BM25 Keyword Index.")
        loader = DirectoryLoader(DATA_PATH, glob="**/*.md", loader_cls=TextLoader)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)
        
        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = 5
        logger.info("Keyword index built successfully.")

        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.5, 0.5]
        )

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", 
            google_api_key=GOOGLE_API_KEY, 
            temperature=0
        )

        prompt_template = """
        You are a helpful Ciklum HR Assistant. 
        Use the following pieces of retrieved context to answer the question. 
        If the context mentions a rule, assume it applies to the user.
        
        Context: {context}
        
        Question: {question}
        
        Answer (be direct and professional):
        """
        
        PROMPT = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question"]
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=ensemble_retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        return qa_chain

    except Exception as e:
        logger.error(f"Failed to setup hybrid chain: {e}", exc_info=True)
        return None

if __name__ == "__main__":
    try:
        chain = setup_hybrid_chain()
        
        if chain:
            print("\nCiklum Hybrid Agent is Ready! (Type 'exit' to stop)")
            
            while True:
                query = input("\nUser: ")
                if query.lower() == "exit":
                    break
                    
                try:
                    logger.info(f"Received query: {query}")
                    response = chain.invoke({"query": query})
                    
                    print(f"Agent: {response['result']}")
                    logger.info(f"Agent Response: {response['result']}")
                    
                    print("\n[Sources Found:]")
                    for doc in response['source_documents']:
                        source_name = doc.metadata.get('source', 'Unknown')
                        print(f"- {source_name}")
                        logger.info(f"Source used: {source_name}")

                except Exception as e:
                    logger.error(f"Error during query execution: {e}", exc_info=True)
                    print("An error occurred while processing your request.")
        else:
            logger.error("Failed to initialize agent chain.")
            print("System Error: Agent could not be started.")

    except Exception as e:
        logger.error(f"Critical Application Error: {e}", exc_info=True)
        print("Critical Error. Check logs.")