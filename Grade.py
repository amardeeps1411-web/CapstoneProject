import time
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from agent_tool import run_agent_with_refine_prompt

# Configuration
GOOGLE_API_KEY = "AIzaSyAI5xtW-81f8A34q15YkQVczmOAIz3F8DU"

# Test Data Definitions
test_dataset = [
    {
        "category": "RAG_Policy",
        "question": "What are the core hours for Ciklum employees?",
        "ground_truth": "Teams typically overlap for 4 Core Hours daily, usually 11:00 AM – 3:00 PM local time."
    },
    {
        "category": "RAG_Policy",
        "question": "Can I work from Bali?",
        "ground_truth": "Yes, under the Nomad Policy, employees can work from a different country for up to 30 days per year."
    },
    {
        "category": "Tool_Database",
        "question": "Check leave balance for EMP-123",
        "ground_truth": "The agent should check the database and report 15 days of Annual Leave and 3 Sick Days."
    },
    {
        "category": "Tool_Database",
        "question": "Check leave for EMP-999",
        "ground_truth": "The agent should report that the employee has 0 days remaining."
    },
    {
        "category": "Hybrid_Context",
        "question": "What is the policy for FCPA?",
        "ground_truth": "Ciklum strictly adheres to the UK Bribery Act and FCPA. No employee may offer or accept gifts to influence decisions."
    },
    {
        "category": "Tool_Action",
        "question": "My laptop screen is broken, please open a ticket.",
        "ground_truth": "The agent must successfully create a support ticket. The response should contain a confirmation message and a Ticket ID starting with 'INC-'."
    }
]

def evaluate_answer(grader_llm, question, predicted_answer, correct_answer):
    """
    Grades the provided answer against the ground truth using an LLM.
    Returns an integer score from 1 (Incorrect) to 5 (Correct).
    """
    prompt = f"""
    You are a technical grader.
    
    QUESTION: {question}
    CORRECT ANSWER: {correct_answer}
    STUDENT ANSWER: {predicted_answer}
    
    Grade the Student Answer from 1 to 5 based on accuracy.
    1 = Incorrect. 5 = Correct.
    Return only the integer number.
    """
    
    try:
        response = grader_llm.invoke(prompt)
        score = response.content.strip()
        if score.isdigit():
            return int(score)
        return 1
    except Exception as e:
        print(f"Grading error: {e}")
        return 0

if __name__ == "__main__":
    print("Starting evaluation pipeline...")
    
    
    agent_executor = run_agent_with_refine_prompt()
    
    
    grader = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", 
        google_api_key=GOOGLE_API_KEY, 
        temperature=0
    )
    
    results = []
    print(f"Running {len(test_dataset)} test cases...\n")

    for i, test in enumerate(test_dataset):
        print(f"Test {i+1}: {test['category']}")
        print(f"Question: {test['question']}")
        
        agent_output = ""
        try:
            
            response = agent_executor.invoke({
                "input": test['question'], 
                "chat_history": [] 
            })
            agent_output = response['output']
            print(f"Answer: {agent_output}")
        except Exception as e:
            agent_output = f"Error: {str(e)}"
            print(agent_output)

        
        score = evaluate_answer(grader, test['question'], agent_output, test['ground_truth'])
        print(f"Score: {score}/5\n")
        
        results.append({
            "Category": test['category'],
            "Question": test['question'],
            "Agent_Answer": agent_output,
            "Ground_Truth": test['ground_truth'],
            "Score": score
        })
        
        time.sleep(10)

    
    df = pd.DataFrame(results)
    
    print("Final Evaluation Report")
    print("-" * 30)
    print(df[['Category', 'Question', 'Score']])
    print("-" * 30)
    
    avg_score = df['Score'].mean()
    print(f"Average Accuracy: {avg_score:.2f} / 5.0")
    
    status = "PASSED" if avg_score > 3.0 else "FAILED"
    print(f"Status: {status}")

    df.to_csv("evaluation_results.csv", index=False)
    print("Detailed results saved to evaluation_results.csv")