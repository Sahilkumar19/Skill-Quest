# Import necessary classes and types
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.memory import BaseMemory
from pydantic import BaseModel, Field
from typing import List, Any, Dict

class PathwayMemory(BaseMemory, BaseModel):
    """Custom memory implementation compatible with modern LangChain patterns."""

    # A list to store conversation history (alternating human and AI messages)
    history: List[Any] = Field(default_factory=list)
    
    @property
    def memory_variables(self) -> List[str]:
        """Return the memory variable names expected by LangChain."""
        return ["chat_history"]

    def __init__(self, **data: Any):
        """Initialize the memory with provided data."""
        super().__init__(**data)
    
    def save_context(self, input_str: str, output_str: str) -> None:
        """
        Save the latest conversation exchange.
        
        Parameters:
        - input_str: User's input message.
        - output_str: AI's response message.
        """
        self.history.extend([
            HumanMessage(content=input_str),
            AIMessage(content=output_str)
        ])
    
    def load_memory_variables(self, inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Load the conversation history into a dictionary format 
        expected by LangChain agents/chains.
        """
        return {"chat_history": self.history}

    def get_history(self, num_exchanges: int = 3) -> str:
        """
        Get the formatted recent conversation history.
        
        Parameters:
        - num_exchanges: Number of past user-AI pairs to retrieve (default: 3).
        
        Returns:
        - A string showing recent user and AI messages.
        """
        recent_history = self.history[-num_exchanges*2:]
        return "\n".join(
            f"{'User' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content}"
            for msg in recent_history
        )

    def clear(self) -> None:
        """Clear the stored conversation history."""
        self.history.clear()

    class Config:
        """Pydantic configuration for the memory class."""
        # Allow arbitrary types in the model
        arbitrary_types_allowed = True
