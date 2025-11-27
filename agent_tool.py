import os
import random
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain.retrievers import EnsembleRetriever
    from langchain.agents import AgentExecutor, create_tool_calling_agent
except ImportError:
    from langchain_classic.retrievers import EnsembleRetriever
    from langchain_classic.agents import AgentExecutor, create_tool_calling_agent

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from logger import setup_logger

load_dotenv()
logger = setup_logger(__name__)

CHROMA_PATH = os.getenv("CHROMA_PATH")
DATA_PATH = os.getenv("DATA_PATH")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def get_retriever():
    try:
        logger.info("Initializing knowledge base retriever.")
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_model)
        vector_retriever = vector_db.as_retriever(search_kwargs={"k": 3})

        loader = DirectoryLoader(DATA_PATH, glob="**/*.md", loader_cls=TextLoader)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)

        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = 3
        
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.5, 0.5]
        )
        return ensemble_retriever
    except Exception as e:
        logger.error(f"Failed to initialize retriever: {e}", exc_info=True)
        raise

global_retriever = get_retriever()

@tool
def lookup_policy(query: str) -> str:
    """Useful for answering questions about HR rules, remote work, holidays, and company policies.
    Input should be a specific question."""
    try:
        logger.info(f"Tool triggered: lookup_policy with query: {query}")
        docs = global_retriever.invoke(query)
        results = "\n\n".join([d.page_content for d in docs])
        return results
    except Exception as e:
        logger.error(f"Error in lookup_policy: {e}", exc_info=True)
        return "Error retrieving policy."

@tool
def check_leave_balance(employee_id: str) -> str:
    """Useful for checking how many vacation or sick days an employee has left.
    Input should be the Employee ID (e.g., 'EMP-123')."""
    try:
        logger.info(f"Tool triggered: check_leave_balance for ID: {employee_id}")
        clean_id = employee_id.replace("EMP", "").replace("-", "").strip()
        
        if clean_id == "123":
            return "You have 15 days of Annual Leave and 3 Sick Days remaining."
        elif clean_id == "999":
            return "You have 0 days remaining. Time to work!"
        else:
            return "Employee ID not found."
    except Exception as e:
        logger.error(f"Error in check_leave_balance: {e}", exc_info=True)
        return "Error checking leave balance."

@tool
def create_support_ticket(issue_description: str) -> str:
    """Useful for reporting IT problems, broken hardware, password resets, or access issues.
    Input should be a clear description of the problem."""
    try:
        logger.info(f"Tool triggered: create_support_ticket for issue: {issue_description}")
        ticket_id = f"INC-{random.randint(1000, 9999)}"
        return f"Success! Ticket #{ticket_id} has been created. IT Support will contact you within 24 hours."
    except Exception as e:
        logger.error(f"Error in create_support_ticket: {e}", exc_info=True)
        return "Error creating ticket."

my_tools = [lookup_policy, check_leave_balance, create_support_ticket]

def run_agent():
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY, temperature=0)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful HR Assistant."),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"), 
        ])
        agent = create_tool_calling_agent(llm, my_tools, prompt)
        return AgentExecutor(agent=agent, tools=my_tools, verbose=False)
    except Exception as e:
        logger.error(f"Error building run_agent: {e}", exc_info=True)
        return None

def run_agent_with_memory():
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY, temperature=0)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful HR Assistant."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        agent = create_tool_calling_agent(llm, my_tools, prompt)
        return AgentExecutor(agent=agent, tools=my_tools, verbose=True)
    except Exception as e:
        logger.error(f"Error building run_agent_with_memory: {e}", exc_info=True)
        return None

def run_agent_with_refine_prompt():
    try:
        logger.info("Building agent with refined prompt.")
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", 
            google_api_key=GOOGLE_API_KEY, 
            temperature=0
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a smart Ciklum HR Assistant.
            
            YOUR GOAL: Help with HR policies AND Leave Balances.
            
            RULES FOR CHOOSING TOOLS:
            1. POLICY QUESTIONS: If the user asks about rules, remote work, 'can I do X', or general info:
               - IGNORE Employee IDs.
               - IMMEDIATELY call `lookup_policy`.
               
            2. LEAVE BALANCES: If the user asks about personal time off or 'my balance':
               - CHECK HISTORY: Look for an ID (like '123' or 'EMP-123') in past messages.
               - CONNECT DOTS: If you have the ID and the intent, RUN `check_leave_balance` immediately.
               - If NO ID is found, ask for it politely.
            3. IT SUPPORT: If the user mentions a broken laptop, password issue, or bug or any issue:
               Ask user if user wants to create a IT ticket. If he acknowledge .Create the ticket
               - Call `create_support_ticket` with the details.  
            4. GENERAL: Be helpful and concise.
            """),
            
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        agent = create_tool_calling_agent(llm, my_tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=my_tools, verbose=False)
        return agent_executor
    except Exception as e:
        logger.error(f"Error building agent: {e}", exc_info=True)
        return None

if __name__ == "__main__":
    try:
        logger.info("Application starting.")
        bot = run_agent_with_refine_prompt()
        
        if bot:
            print("\nCiklum Agent is Ready! (Type 'exit' to stop)")
            chat_history = []

            while True:
                user_input = input("\nUser: ")
                if user_input.lower() == "exit": 
                    break
                
                try:
                    logger.info(f"User Input: {user_input}")
                    response = bot.invoke({
                        "input": user_input,
                        "chat_history": chat_history
                    })
                    
                    output_text = response['output']
                    print(f"Agent: {output_text}")
                    logger.info(f"Agent Response: {output_text}")

                    chat_history.append(HumanMessage(content=user_input))
                    chat_history.append(AIMessage(content=output_text))
                    
                except Exception as e:
                    error_msg = f"Interaction error: {e}"
                    print(error_msg)
                    logger.error(error_msg, exc_info=True)
        else:
            logger.error("Failed to initialize agent bot.")
            print("System Error: Agent could not be initialized.")
            
    except Exception as e:
        logger.error(f"Critical Application Error: {e}", exc_info=True)
        print("Critical Error. Check logs.")