Project Overview
The Ciklum HR Guardian is an Agentic RAG (Retrieval-Augmented Generation) system designed to automate internal HR support. Unlike standard chatbots, this system utilizes Hybrid Search (Semantic + Keyword) and Function Calling to differentiate between static policy queries and dynamic employee data requests.

Features
Hybrid Search Architecture: Combines BM25 (Keyword) and ChromaDB (Vector) retrieval to ensure high accuracy for specific acronyms (e.g., FCPA) and general semantic concepts.

Agentic Tool Calling: The system dynamically selects tools based on user intent:

Policy Lookup: Retrieves information from the knowledge base.

Leave Balance: Queries a mock SQL database for employee data.

IT Support: Generates support tickets for technical issues.

Contextual Memory: The agent retains conversation history to handle follow-up questions without requiring repeated context (e.g., remembering Employee IDs).

Automated Evaluation: Includes an LLM-as-a-Judge pipeline to grade agent responses against ground truth data.

Production Logging: Centralized logging system that writes to file rather than console.

Technical Stack
Language: Python 3.x

Orchestration: LangChain (Classic & Community)

LLM: Google Gemini 2.0 Flash

Vector Store: ChromaDB

Embeddings: HuggingFace (all-MiniLM-L6-v2)

Testing: Pandas (for reporting)

Project Structure
agent_tool.py: The main application entry point. Contains the Agent logic, tool definitions, and chat loop.

Load_And_DBCreation.py: The data ingestion pipeline. Loads Markdown files, chunks text, and builds the ChromaDB vector store.

Grade.py: The evaluation script. Runs the agent against a test dataset and uses an LLM to grade accuracy (1-5 scale).

logger.py: Centralized logging configuration file.

Knowledge-base/: Directory containing source HR policy documents (.md and .docx).

chroma_db/: Persisted vector database directory.

logs/: Directory where application logs (app.log) are stored.

Requirement.txt: List of Python dependencies.

Setup and Installation
Clone the repository Ensure all files from the project structure are present.

Install Dependencies Run the following command to install required libraries:

Bash

pip install -r Requirement.txt
Environment Configuration Create a file named .env in the root directory and add the following variables:

Ini, TOML

GOOGLE_API_KEY=your_api_key_here
CHROMA_PATH=./chroma_db
DATA_PATH=./Knowledge-base
Usage Instructions
1. Initialize the Database
Before running the agent, you must ingest the data to build the vector store.

Bash

python Load_And_DBCreation.py
Check logs/app.log to confirm successful ingestion.

2. Run the Agent
Start the interactive chat interface.

Bash

python agent_tool.py
Example Commands:

"What is the policy for remote work?"

"Check leave balance for EMP-123"

"My VPN is not working, please create a ticket."

Type exit to quit.

3. Run Evaluation
Execute the automated grading pipeline to generate an accuracy report.

Bash

python Grade.py
This will generate a CSV report named evaluation_results.csv containing scores for various test scenarios.

Logging
This application uses structured file logging. Console output is minimized for clean interaction. To view detailed execution logs, debug information, or errors, open: logs/app.log