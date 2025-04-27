# Patch for pysqlite3 to act as sqlite3
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Import necessary modules
import sys
import os
from datetime import datetime
from crew import PathwayTutor
from dotenv import load_dotenv
import litellm
from crewai import Crew, Process
import ast
import streamlit as st

# ---------- Environment Setup ----------
# # Load environment variables from .env file
load_dotenv()

# Set API key for Groq API
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
# Configure LiteLLM to drop unnecessary parameters
litellm.drop_params = True
# Fetch the model name from environment variables
MODEL_NAME = os.getenv("MODEL")


# ---------- Session Management ----------
class SessionManager:
    """Manages user sessions and their data like history, root question, etc."""
    def __init__(self):
        self.sessions = {}

    def get_session(self, session_id):
        # Create a new session if not exists
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                'history': [],
                'root_category': None,
                'root_question': None
            }
        return self.sessions[session_id]


def format_history(history):
    """Formats last 3 relevant questions and answers for prompt history"""
    return "\n".join([
        f"Q: {item['question']}\nA: {item['answer']}"
        for item in history
        if item['category'] != "Irrelevant"
    ][-3:])


def is_followup_relevant(new_question, session):
    """Checks if the follow-up question is related to the initial topic"""
    if not session['root_question'] or not session['root_category']:
        return True  # Allow follow-up if no root context exists

    # Compare keyword overlaps crudely
    root_keywords = session['root_question'].lower().split()
    follow_keywords = new_question.lower().split()
    overlap = set(root_keywords) & set(follow_keywords)

    return len(overlap) > 0


def initialize_session_state():
    """Initialize Streamlit session state variables if they don't exist"""
    if 'session_manager' not in st.session_state:
        st.session_state.session_manager = SessionManager()
    if 'current_session' not in st.session_state:
        st.session_state.current_session = st.session_state.session_manager.get_session("default")
    if 'tutor' not in st.session_state:
        st.session_state.tutor = PathwayTutor()
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []


# ---------- Processing Logic ----------
def process_question(user_question):
    """Categorizes and processes the user question, returns the answer"""
    # Prepare input parameters
    inputs = {
        'question': user_question,
        'current_year': str(datetime.now().year),
        'model': MODEL_NAME,
        'history': format_history(st.session_state.current_session['history'])
    }

    # Categorize the question
    categorize_task = st.session_state.tutor.categorize_question()
    category_crew = Crew(
        agents=[st.session_state.tutor.classifier()],
        tasks=[categorize_task],
        process=Process.sequential,
        verbose=True
    )
    categorization = category_crew.kickoff(inputs=inputs)
    category_dict = ast.literal_eval(str(categorization).strip())
    category = category_dict['category']

    # Update session's root category and root question
    st.session_state.current_session['root_category'] = category
    st.session_state.current_session['root_question'] = user_question

    # Handle irrelevant queries
    if category == "Irrelevant":
        return "This question is outside my expertise in Data Science/AI/ML. Please ask about Data Science, ML, or AI concepts.", category

    # Map categories to tasks
    task_mapping = {
        "Definition-Based": st.session_state.tutor.define_term(),
        "Concept-Explanation": st.session_state.tutor.explain_concept(),
        "Types-Examples": st.session_state.tutor.give_types_examples(),
        "Problem-Solving": st.session_state.tutor.solve_problem(),
        "Comparison": st.session_state.tutor.compare_concepts(),
        "Process-Guide": st.session_state.tutor.guide_process(),
        "Doubt-Clearing": st.session_state.tutor.clear_doubt(),
        "Python-Code": st.session_state.tutor.provide_python_code(),
        "Python-Debug": st.session_state.tutor.debug_python_code()
    }

     # Fetch the task for the category
    task = task_mapping.get(category)
    if task:
        execution_crew = Crew(
            agents=[task.agent],
            tasks=[task],
            process=Process.sequential,
            verbose=True,
            full_output=True
        )
        result = execution_crew.kickoff(inputs=inputs)
        return result.raw, category

    # Fallback if no matching task found
    return "Unable to process the question.", category


# ---------- Streamlit UI ----------
def main():
    st.set_page_config(
        page_title="Skill Quest - AI Learning Companion",
        page_icon="🎓",
        layout="wide"
    )

    initialize_session_state()

    # Page Title
    st.title("🎓 Skill Quest - Your AI Learning Companion")
    st.markdown("---")

    # Sidebar content
    with st.sidebar:
        st.header("About")
        st.write("I specialize in Data Science, Machine Learning, and AI concepts.")
        if st.button("🆕 Start New Session"):
            # Create a new session
            st.session_state.current_session = st.session_state.session_manager.get_session(
                str(datetime.now())
            )
            st.session_state.chat_history = []
            st.success("New session started!")

    # Display past chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Capture user input
    user_input = st.chat_input("Ask me about Data Science, ML, or AI...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Process the user query
                    answer, category = process_question(user_input)

                    # Render the AI's response
                    st.markdown(answer, unsafe_allow_html=True)


                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer
                    })

                    # Update session history if relevant
                    if category != "Irrelevant":
                        st.session_state.current_session['history'].append({
                            'question': user_input,
                            'answer': answer,
                            'category': category
                        })

                except Exception as e:
                    # Error handling
                    error_msg = f"An error occurred: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": error_msg
                    })

# Entry point
if __name__ == "__main__":
    main()