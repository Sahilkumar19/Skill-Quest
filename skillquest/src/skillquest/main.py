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
                'category': None
            }
        return self.sessions[session_id]

def display_welcome():
    print("\n" + "🌟" * 40)
    print("Welcome to Pathway Tutor - Your AI Learning Companion!")
    print("🌟" * 40)
    print("\nI specialize in Data Science, Machine Learning, and AI concepts.")
    print("Type 'exit' to quit or 'new' to start a fresh session.\n")

def handle_response(category, result, session):
    """Handles response output and user choices"""
    print("\n" + "=" * 60)
    
    # try:
    #     # Safe attribute access
    #     category = getattr(category, 'category', 'Unknown')
    #     output = getattr(result, 'output', 'No guidance generated')
        
    if category == "Irrelevant":
        print("🚫 This question is outside my expertise in Data Science/AI/ML.")
        print("Please ask about Data Science, ML, or AI concepts.")
        print("=" * 60)
        return 'new'
            
    print(f"🧠 CATEGORY: {category}")
    print(f"📘 GUIDANCE:\n{result}")
        
    # except AttributeError as e:
    #     print(f"⚠️ Error processing response: {str(e)}")
    
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
    # conversation_history = []
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
            categorization = category_crew.kickoff(inputs=inputs)
            category = str(categorization).strip()
            current_session['category'] = category
            category_dict = ast.literal_eval(category)
            
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
            category = category_dict['category']
            if category == "Irrelevant":
                result = type('obj', (object,), {
                    'category':'Irrelevant',
                    'output':'This question is outside my expertise in Data Science/AI/ML.'
                })
                handle_response(category, result, current_session)
                continue
            # else:
            #     current_session['history'].append({
            #     'question': question,
            #     'answer': result.output,
            #     'category': category
            #     })
            else:
            # Proceed with relevant tasks
                # category_dict = ast.literal_eval(category)
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
                print(result)
                print(type(result))
                # Store only relevant history
                current_session['history'].append({
                    'question': question,
                    'answer': result['output'],
                    'category': category
                })
            # task_mapping = {
            #     "Definition-Based": tutor.define_term(),
            #     "Concept-Explanation": tutor.explain_concept(),
            #     "Problem-Solving": tutor.solve_problem(),
            #     "Comparison": tutor.compare_concepts(),
            #     "Process-Guide": tutor.guide_process(),
            #     "Doubt-Clearing": tutor.clear_doubt(),
            #     "Python-Code": tutor.provide_python_code(),
            #     "Python-Debug": tutor.debug_python_code()
            # }
            # category_dict = ast.literal_eval(category)
            # # print(category)
            # task = task_mapping.get(category_dict.get('category'))
            # # print(task)
            # if not task:
            #     print(f"⚠️ Unhandled category: {category}")
            #     continue

            # execution_crew = Crew(
            #     agents=[task.agent],
            #     tasks=[task],
            #     process=Process.sequential,
            #     verbose=True
            # )
            # result = execution_crew.kickoff(inputs=inputs)

            # # Store only relevant history
            # if category != "Irrelevant":
            #     current_session['history'].append({
            #         'question': question,
            #         'answer': result.output,
            #         'category': category
            #     })
            
            while True:
                choice = handle_response(category, result['output'], current_session)
                
                if choice == '1':
                    new_question = input("\n🔍 Follow-up question: ")
                    result = execution_crew.kickoff(inputs={
                        'question': new_question,
                        'model': MODEL_NAME,
                        'history': format_history(current_session['history']),
                        'current_year': datetime.now().year
                    })
                    if category != "Irrelevant":
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
            # conversation_history.append(f"System Error: {str(e)}")

if __name__ == "__main__":
    run()