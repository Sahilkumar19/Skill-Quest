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

class CategoryOutput(BaseModel):
    category: str

class GuidanceOutput(BaseModel):
    output: str


class PathwayTutorConfig(BaseModel):
    model_config = ConfigDict(
        extra="ignore",
        arbitrary_types_allowed=True
    )
    # Define ClassVar fields as class variables but not as model fields
    base_directory: Path = Field(default_factory=lambda: Path(__file__).parent)
    
    # Class variables outside of the Pydantic model field system
    agents_config: str = "config/agents.yaml"
    tasks_config: str = "config/tasks.yaml"

@CrewBase
class PathwayTutor:
    """PathwayTutor AI Crew"""
    def __init__(self):
        # super().__init__()
        self.config = PathwayTutorConfig()  # Instantiate config
        self._configure_paths()
        self.memory = PathwayMemory()

    def _configure_paths(self):
        self.agents_config_path = self.config.base_directory / self.config.agents_config
        self.tasks_config_path = self.config.base_directory / self.config.tasks_config

    def _create_agent(self, config_name):
        with open(self.agents_config_path) as f:
            config = yaml.safe_load(f)
    
        return Agent(
            config=config[config_name],
            verbose=True,
            memory=self.memory,  # Inject memory
            llm=OpenAI(
                model_name=os.getenv("MODEL"),
                openai_api_key=os.getenv("GROQ_API_KEY"),
                base_url="https://api.groq.com/openai/v1",
                temperature=0.3,
                max_tokens=2048,
            ),
            allow_delegation=False,
            max_iter=5
        )
    
    @agent
    def classifier(self) -> Agent:
        return self._create_agent('classifier')

    @agent
    def definition_based(self) -> Agent:
        return self._create_agent('definition_based')

    @agent
    def concept_explanation(self) -> Agent:
        return self._create_agent('concept_explanation')

    @agent
    def problem_solving(self) -> Agent:
        return self._create_agent('problem_solving')

    @agent
    def comparison(self) -> Agent:
        return self._create_agent('comparison')

    @agent
    def process_guide(self) -> Agent:
        return self._create_agent('process_guide')

    @agent
    def doubt_clearing(self) -> Agent:
        return self._create_agent('doubt_clearing')

    @agent
    def python_code(self) -> Agent:
        return self._create_agent('python_code')

    @agent
    def python_debug(self) -> Agent:
        return self._create_agent('python_debug')

    @task
    def categorize_question(self) -> Task:
        """Task to classify the question into a category"""
        return Task(
            config=self.tasks_config['categorization'],
            agent=self.classifier(),
            # output_json=None
            output_json=CategoryOutput
        )

    @task
    def define_term(self) -> Task:
        """Task to provide a definition and basic explanation of a term"""
        return Task(
            config=self.tasks_config['definition_based_tasks'],
            agent=self.definition_based(),
            output_json=GuidanceOutput

        )

    @task
    def explain_concept(self) -> Task:
        """Task to provide a detailed explanation of a concept"""
        return Task(
            config=self.tasks_config['concept_explanation_tasks'],
            agent=self.concept_explanation(),
            # output_file="guidance.md"
            output_json=GuidanceOutput

        )

    @task
    def solve_problem(self) -> Task:
        """Task to solve a problem step-by-step"""
        return Task(
            config=self.tasks_config['problem_solving_tasks'],
            agent=self.problem_solving(),
            output_json=GuidanceOutput

        )

    @task
    def compare_concepts(self) -> Task:
        """Task to compare and contrast concepts"""
        return Task(
            config=self.tasks_config['comparison_tasks'],
            agent=self.comparison(),
            output_json=GuidanceOutput

        )

    @task
    def guide_process(self) -> Task:
        """Task to guide through a process"""
        return Task(
        config=self.tasks_config['process_guide_tasks'],
        agent=self.process_guide(),
        output_json=GuidanceOutput

        )

    @task
    def clear_doubt(self) -> Task:
        """Task to clear a doubt"""
        return Task(
        config=self.tasks_config['doubt_clearing_tasks'],
        agent=self.doubt_clearing(),
        output_json=GuidanceOutput

        )

    @task
    def provide_python_code(self) -> Task:
        """Task to provide Python code"""
        return Task(
        config=self.tasks_config['python_code_tasks'],
        agent=self.python_code(),
        output_json=GuidanceOutput

        )

    @task
    def debug_python_code(self) -> Task:
        """Task to debug Python code"""
        return Task(
        config=self.tasks_config['python_debug_tasks'],
        agent=self.python_debug(),
        output_json=GuidanceOutput

        )

    @crew
    def crew(self) -> Crew:
        """Creates the PathwayTutor crew"""
        return Crew(
            agents=[
                self.classifier(),
                self.definition_based(),
                self.concept_explanation(),
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
                self.solve_problem(),
                self.compare_concepts(),
                self.guide_process(),
                self.clear_doubt(),
                self.provide_python_code(),
                self.debug_python_code()

            ],
            memory=self.memory,
            process=Process.sequential,
            verbose=2,
            full_output=True
        )