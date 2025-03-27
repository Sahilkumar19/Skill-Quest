import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(project_root)
from datetime import datetime
from crew import PathwayTutor
from dotenv import load_dotenv
# import os
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

def initialize_session_state():
    if 'session_manager' not in st.session_state:
        st.session_state.session_manager = SessionManager()
    if 'current_session' not in st.session_state:
        st.session_state.current_session = st.session_state.session_manager.get_session("default")
    if 'tutor' not in st.session_state:
        st.session_state.tutor = PathwayTutor()

def main():
    st.set_page_config(
        page_title="Skill Quest - AI Learning Companion",
        page_icon="🎓",
        layout="wide"
    )

    initialize_session_state()

    # Header
    st.title("🎓 Skill Quest - Your AI Learning Companion")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("About")
        st.write("I specialize in Data Science, Machine Learning, and AI concepts.")
        
        if st.button("Start New Session"):
            st.session_state.current_session = st.session_state.session_manager.get_session(
                str(datetime.now())
            )
            st.success("New session started!")

    # Main chat interface
    st.subheader("Ask Your Question")
    user_question = st.text_input("💭 Enter your Data Science/AI question:", key="question_input")

    if st.button("Submit Question"):
        if user_question:
            with st.spinner("Processing your question..."):
                try:
                    # Prepare inputs
                    inputs = {
                        'question': user_question,
                        'current_year': str(datetime.now().year),
                        'model': MODEL_NAME,
                        'history': format_history(st.session_state.current_session['history'])
                    }

                    # Create expander for agent interactions
                    with st.expander("🤖 Agent Interactions", expanded=True):
                        # Categorize question
                        st.write("### Question Categorization")
                        st.write("🔄 Analyzing question category...")
                        
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

                        st.write("📋 Category Analysis:")
                        st.code(str(categorization), language='python')

                        # Update session
                        st.session_state.current_session['root_category'] = category
                        st.session_state.current_session['root_question'] = user_question

                        # Handle response
                        if category == "Irrelevant":
                            st.error("This question is outside my expertise in Data Science/AI/ML. Please ask about Data Science, ML, or AI concepts.")
                        else:
                            # Get appropriate task
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
                                st.write(f"\n### Response Generation ({category})")
                                st.write("🔄 Generating detailed response...")
                                
                                execution_crew = Crew(
                                    agents=[task.agent],
                                    tasks=[task],
                                    process=Process.sequential,
                                    verbose=True
                                )
                                result = execution_crew.kickoff(inputs=inputs)

                                # Display full agent interaction
                                st.write("📝 Agent Reasoning:")
                                st.code(str(result), language='python')

                    # Display final answer in a clean format
                    st.markdown("---")
                    st.markdown("### 📌 Final Answer")
                    if category != "Irrelevant":
                        # Format the output nicely
                        output = result['output']
                        
                        # Split the output into sections if they exist
                        sections = output.split('##')
                        for section in sections:
                            if section.strip():
                                # Remove any remaining '#' characters
                                section = section.replace('#', '').strip()
                                # Add proper markdown formatting
                                st.markdown(f"#### {section}")

                        # Store in history
                        st.session_state.current_session['history'].append({
                            'question': user_question,
                            'answer': result['output'],
                            'category': category
                        })

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.code(str(e), language='python')

    # Display conversation history
    if st.session_state.current_session['history']:
        st.markdown("---")
        st.subheader("Conversation History")
        for item in st.session_state.current_session['history']:
            with st.expander(f"Q: {item['question']}", expanded=False):
                st.write(f"Category: {item['category']}")
                st.write(f"A: {item['answer']}")

if __name__ == "__main__":
    main()