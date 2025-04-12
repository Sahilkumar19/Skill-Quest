#!/usr/bin/env python3

from datetime import datetime
from .crew import PathwayTutor
from dotenv import load_dotenv
import os
import litellm
from crewai import Crew, Process
import ast
# Load environment variables
load_dotenv()

# Configure LiteLLM for Groq
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")  # Map Groq key to OpenAI key name
litellm.drop_params = True
MODEL_NAME = os.getenv("MODEL")

class SessionManager:
    """Manages isolated user sessions with temporary memory"""
    def __init__(self):
        self.sessions = {}
        
    def get_session(self, session_id):
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                'history': [],
                'root_category': None,
                'root_question': None
            }
        return self.sessions[session_id]

def is_followup_relevant(new_question, session):
    """Checks if follow-up relates to original context"""
    if not session['root_question'] or not session['root_category']:
        return True  # Allow if root not defined (edge case)

    # Simplified check (You can replace this with an actual model call if needed)
    relevance_prompt = f"""
    Original Question: {session['root_question']}
    Original Category: {session['root_category']}
    Conversation History: {session['history']}
    
    New Follow-up Question: {new_question}
    
    Does the follow-up relate to the original question and category? (Yes/No)
    """
    root_keywords = session['root_question'].lower().split()
    follow_keywords = new_question.lower().split()
    overlap = set(root_keywords) & set(follow_keywords)
    
    return len(overlap) > 0  # crude keyword overlap check

def display_welcome():
    print("\n" + "🌟" * 40)
    print("Welcome to Pathway Tutor - Your AI Learning Companion!")
    print("🌟" * 40)
    print("\nI specialize in Data Science, Machine Learning, and AI concepts.")
    print("Type 'exit' to quit or 'new' to start a fresh session.\n")

def handle_response(category, result, session):
    """Handles response output and user choices"""
    print("\n" + "=" * 60)
    if category == "Irrelevant":
        print("🚫 This question is outside my expertise in Data Science/AI/ML.")
        print("Please ask about Data Science, ML, or AI concepts.")
        print("=" * 60)
        return 'new'
            
    print(f"🧠 CATEGORY: {category}")
    print(f"📘 GUIDANCE:\n{result}")
    print("=" * 60)
    return input("\n🤔 Choose: 1. Follow-up 2. New question 3. Exit\nChoice (1-3): ")

def format_history(history):
    """Formats only relevant history entries"""
    return "\n".join([
        f"Q: {item['question']}\nA: {item['answer']}" 
        for item in history 
        if item['category'] != "Irrelevant"
    ][-3:])  # Last 3 relevant exchanges

def run():
    tutor = PathwayTutor()  
    sessions = SessionManager()
    current_session = sessions.get_session("default")  # Simplified single session

    display_welcome()

    while True:
        try:
            question = input("\n💡 Your Data Science/AI question: ").strip()
            if question.lower() in ['exit', 'quit']:
                print("\n👋 Thank you for using Pathway Tutor!")
                break

            if question.lower() == 'new':
                current_session = sessions.get_session(str(datetime.now()))
                print("\n🆕 New session started!")
                continue

            inputs = {
                'question': question,
                'current_year': str(datetime.now().year),
                'model': MODEL_NAME,
                'history': format_history(current_session['history'])
            }
            
            # 1. Categorize the question
            categorize_task = tutor.categorize_question()
            category_crew = Crew(
                agents=[tutor.classifier()],
                tasks=[categorize_task],
                process=Process.sequential,
                verbose=True
            )
            # print("this is categorized task",categorize_task)
            # print("this is output",categorize_task.output)
            categorization = category_crew.kickoff(inputs=inputs)
            category = str(categorization).strip()
            category_dict = ast.literal_eval(category)
            category = category_dict['category']
            current_session['root_category'] = category
            current_session['root_question'] = question
            
            task_mapping = {
                "Definition-Based": tutor.define_term(),
                "Concept-Explanation": tutor.explain_concept(),
                "Problem-Solving": tutor.solve_problem(),
                "Comparison": tutor.compare_concepts(),
                "Process-Guide": tutor.guide_process(),
                "Doubt-Clearing": tutor.clear_doubt(),
                "Python-Code": tutor.provide_python_code(),
                "Python-Debug": tutor.debug_python_code()
            }
            result = None  # Initialize result variable
            # Handle irrelevant questions immediately
            if category == "Irrelevant":
                result = type('obj', (object,), {
                    'category':'Irrelevant',
                    'output':'This question is outside my expertise in Data Science/AI/ML.'
                })
                handle_response(category, result, current_session)
                continue
            else:
                task = task_mapping.get(category)
                if not task:
                    print(f"⚠️ Unhandled category: {category}")
                    continue

                execution_crew = Crew(
                    agents=[task.agent],
                    tasks=[task],
                    process=Process.sequential,
                    verbose=True
                )
                result = execution_crew.kickoff(inputs=inputs)
                # print("this is result",result)
                # print("this is raw output",result.raw)
                # print("this is summary output",result.summary)
                # print("this is description output",result.description)
                # Store only relevant history
                current_session['history'].append({
                    'question': question,
                    'answer': result['output'],
                    'category': category
                })
            
            while True:
                choice = handle_response(category, result['output'], current_session)
                
                if choice == '1':
                    new_question = input("\n🔍 Follow-up question: ")

                    # Check follow-up relevance
                    if not is_followup_relevant(new_question, current_session):
                        print("\n🚫 This follow-up is off-topic. Please stay within:")
                        print(f"- Original question: {current_session['root_question']}")
                        print(f"- Category: {current_session['root_category']}")
                        continue
                    # valid followup
                    result = execution_crew.kickoff(inputs={
                        'question': new_question,
                        'model': MODEL_NAME,
                        'history': format_history(current_session['history']),
                        'current_year': datetime.now().year
                    })
                    current_session['history'].append({
                        'question': new_question,
                        'answer': result['output'],
                        'category': category
                    })
                elif choice == '2':
                    break
                elif choice == '3':
                    print("\n👋 Session ended.")
                    return
                else:
                    print("⚠️ Invalid choice, please select 1-3")
        except KeyboardInterrupt:
            print("\n\n🛑 Session interrupted. Type 'exit' to quit.")
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    run()