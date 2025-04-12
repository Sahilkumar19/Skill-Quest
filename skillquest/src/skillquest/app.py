import sys
import os
from datetime import datetime
from .crew import PathwayTutor
from dotenv import load_dotenv
import litellm
from crewai import Crew, Process
import ast
import streamlit as st

# Load environment variables
load_dotenv()

# Configure LiteLLM for Groq
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
litellm.drop_params = True
MODEL_NAME = os.getenv("MODEL")


# ---------- Session Management ----------
class SessionManager:
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


def format_history(history):
    return "\n".join([
        f"Q: {item['question']}\nA: {item['answer']}"
        for item in history
        if item['category'] != "Irrelevant"
    ][-3:])


def is_followup_relevant(new_question, session):
    """Checks if follow-up relates to original context"""
    if not session['root_question'] or not session['root_category']:
        return True  # Allow if root not defined (edge case)

    root_keywords = session['root_question'].lower().split()
    follow_keywords = new_question.lower().split()
    overlap = set(root_keywords) & set(follow_keywords)

    return len(overlap) > 0  # crude keyword overlap check


def initialize_session_state():
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
    """Process a question and return the result"""
    inputs = {
        'question': user_question,
        'current_year': str(datetime.now().year),
        'model': MODEL_NAME,
        'history': format_history(st.session_state.current_session['history'])
    }

    # Categorize question
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

    # Update session context
    st.session_state.current_session['root_category'] = category
    st.session_state.current_session['root_question'] = user_question

    # Handle irrelevant category
    if category == "Irrelevant":
        return "This question is outside my expertise in Data Science/AI/ML. Please ask about Data Science, ML, or AI concepts.", category

    # Map category to appropriate task
    task_mapping = {
        "Definition-Based": st.session_state.tutor.define_term(),
        "Concept-Explanation": st.session_state.tutor.explain_concept(),
        "Problem-Solving": st.session_state.tutor.solve_problem(),
        "Comparison": st.session_state.tutor.compare_concepts(),
        "Process-Guide": st.session_state.tutor.guide_process(),
        "Doubt-Clearing": st.session_state.tutor.clear_doubt(),
        "Python-Code": st.session_state.tutor.provide_python_code(),
        "Python-Debug": st.session_state.tutor.debug_python_code()
    }

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

    return "Unable to process the question.", category


# ---------- Streamlit UI ----------
def main():
    st.set_page_config(
        page_title="Skill Quest - AI Learning Companion",
        page_icon="🎓",
        layout="wide"
    )

    initialize_session_state()

    # Title and Sidebar
    st.title("🎓 Skill Quest - Your AI Learning Companion")
    st.markdown("---")

    with st.sidebar:
        st.header("About")
        st.write("I specialize in Data Science, Machine Learning, and AI concepts.")
        if st.button("🆕 Start New Session"):
            st.session_state.current_session = st.session_state.session_manager.get_session(
                str(datetime.now())
            )
            st.session_state.chat_history = []
            st.success("New session started!")

    # Chat History Display
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User Input
    user_input = st.chat_input("Ask me about Data Science, ML, or AI...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer, category = process_question(user_input)

                    # st.markdown(answer)
                    st.markdown(answer, unsafe_allow_html=True)


                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer
                    })

                    if category != "Irrelevant":
                        st.session_state.current_session['history'].append({
                            'question': user_input,
                            'answer': answer,
                            'category': category
                        })

                except Exception as e:
                    error_msg = f"An error occurred: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": error_msg
                    })


if __name__ == "__main__":
    main()