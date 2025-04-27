# Import necessary libraries and modules
from pathlib import Path
from typing import ClassVar, Any
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from langchain_openai import OpenAI
from memory import PathwayMemory
from pydantic import BaseModel, ConfigDict, Field
import yaml
from dotenv import load_dotenv
import os
import litellm

# Define output structure for categorization
class CategoryOutput(BaseModel):
    category: str

# Define output structure for general guidance tasks
class GuidanceOutput(BaseModel):
    output: str

# Configuration class for PathwayTutor project
class PathwayTutorConfig(BaseModel):
    # Pydantic model config
    model_config = ConfigDict(
        extra="ignore",
        arbitrary_types_allowed=True
    )

    # Define base directory where configuration files reside
    base_directory: Path = Field(default_factory=lambda: Path(__file__).parent)
    
    # Define relative paths to agent and task configurations
    agents_config: str = "config/agents.yaml"
    tasks_config: str = "config/tasks.yaml"

# Main class that builds the AI crew using CrewAI
@CrewBase
class PathwayTutor:
    """PathwayTutor AI Crew"""

    def __init__(self):
        # Initialize configuration and memory
        self.config = PathwayTutorConfig()
        self._configure_paths()
        self.memory = PathwayMemory()

    def _configure_paths(self):
        """Set full paths to agent and task configuration YAMLs."""
        self.agents_config_path = self.config.base_directory / self.config.agents_config
        self.tasks_config_path = self.config.base_directory / self.config.tasks_config

    def _create_agent(self, config_name):
        """Create an Agent instance using configuration from YAML file."""
        with open(self.agents_config_path) as f:
            config = yaml.safe_load(f)

        return Agent(
            config=config[config_name],
            verbose=True,
            memory=self.memory,  # Attach memory module
            llm=OpenAI(  # Use OpenAI-compatible model with GROQ API
                model_name=os.getenv("MODEL"),
                openai_api_key=os.getenv("GROQ_API_KEY"),
                base_url="https://api.groq.com/openai/v1",
                temperature=0.3,
                max_tokens=2048,
            ),
            allow_delegation=False,
            max_iter=5
        )

    # === Agent creation methods ===
    @agent
    def classifier(self) -> Agent:
        """Agent to classify incoming questions."""
        return self._create_agent('classifier')

    @agent
    def definition_based(self) -> Agent:
        """Agent to define a term."""
        return self._create_agent('definition_based')

    @agent
    def concept_explanation(self) -> Agent:
        """Agent to explain a concept in depth."""
        return self._create_agent('concept_explanation')

    @agent
    def types_examples(self) -> Agent:
        """Agent to list types and examples."""
        return self._create_agent('types_examples')

    @agent
    def problem_solving(self) -> Agent:
        """Agent to solve problems."""
        return self._create_agent('problem_solving')

    @agent
    def comparison(self) -> Agent:
        """Agent to compare two concepts."""
        return self._create_agent('comparison')

    @agent
    def process_guide(self) -> Agent:
        """Agent to guide a process step-by-step."""
        return self._create_agent('process_guide')

    @agent
    def doubt_clearing(self) -> Agent:
        """Agent to clear user doubts."""
        return self._create_agent('doubt_clearing')

    @agent
    def python_code(self) -> Agent:
        """Agent to generate Python code."""
        return self._create_agent('python_code')

    @agent
    def python_debug(self) -> Agent:
        """Agent to debug Python code."""
        return self._create_agent('python_debug')

    # === Task creation methods ===
    @task
    def categorize_question(self) -> Task:
        """Task to classify the question into a predefined category."""
        return Task(
            config=self.tasks_config['categorization'],
            agent=self.classifier(),
            output_json=CategoryOutput
        )

    @task
    def define_term(self) -> Task:
        """Task to provide the definition of a term."""
        return Task(
            config=self.tasks_config['definition_based_tasks'],
            agent=self.definition_based(),
            output_json=GuidanceOutput
        )

    @task
    def explain_concept(self) -> Task:
        """Task to provide a detailed explanation for a concept."""
        return Task(
            config=self.tasks_config['concept_explanation_tasks'],
            agent=self.concept_explanation(),
            output_json=GuidanceOutput
        )

    @task
    def give_types_examples(self) -> Task:
        """Task to provide different types and examples of a concept."""
        return Task(
            config=self.tasks_config['types_examples_tasks'],
            agent=self.types_examples(),
            output_json=GuidanceOutput
        )

    @task
    def solve_problem(self) -> Task:
        """Task to solve a given problem step-by-step."""
        return Task(
            config=self.tasks_config['problem_solving_tasks'],
            agent=self.problem_solving(),
            output_json=GuidanceOutput
        )

    @task
    def compare_concepts(self) -> Task:
        """Task to compare and contrast two concepts."""
        return Task(
            config=self.tasks_config['comparison_tasks'],
            agent=self.comparison(),
            output_json=GuidanceOutput
        )

    @task
    def guide_process(self) -> Task:
        """Task to guide a user through a complete process."""
        return Task(
            config=self.tasks_config['process_guide_tasks'],
            agent=self.process_guide(),
            output_json=GuidanceOutput
        )

    @task
    def clear_doubt(self) -> Task:
        """Task to clear a user's doubt."""
        return Task(
            config=self.tasks_config['doubt_clearing_tasks'],
            agent=self.doubt_clearing(),
            output_json=GuidanceOutput
        )

    @task
    def provide_python_code(self) -> Task:
        """Task to generate required Python code."""
        return Task(
            config=self.tasks_config['python_code_tasks'],
            agent=self.python_code(),
            output_json=GuidanceOutput
        )

    @task
    def debug_python_code(self) -> Task:
        """Task to debug and fix provided Python code."""
        return Task(
            config=self.tasks_config['python_debug_tasks'],
            agent=self.python_debug(),
            output_json=GuidanceOutput
        )

    # === Crew creation method ===
    @crew
    def crew(self) -> Crew:
        """Creates the complete AI crew by linking agents and tasks."""
        return Crew(
            agents=[
                self.classifier(),
                self.definition_based(),
                self.concept_explanation(),
                self.types_examples(),
                self.problem_solving(),
                self.comparison(),
                self.process_guide(),
                self.doubt_clearing(),
                self.python_code(),
                self.python_debug()
            ],
            tasks=[
                self.categorize_question(),
                self.define_term(),
                self.explain_concept(),
                self.give_types_examples(),
                self.solve_problem(),
                self.compare_concepts(),
                self.guide_process(),
                self.clear_doubt(),
                self.provide_python_code(),
                self.debug_python_code()
            ],
            memory=self.memory,          # Share memory across all agents and tasks
            process=Process.sequential,  # Execute tasks sequentially
            verbose=2,                   # Set verbosity level for better logs
            full_output=True             # Return full outputs after execution
        )
