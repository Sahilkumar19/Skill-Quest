from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.memory import BaseMemory
from pydantic import BaseModel, Field
from typing import List, Any, Dict

class PathwayMemory(BaseMemory, BaseModel):
    """Custom memory implementation with modern LangChain patterns"""
    history: List[Any] = Field(default_factory=list)
    
    @property
    def memory_variables(self) -> List[str]:
        return ["chat_history"]

    def __init__(self, **data: Any):
        super().__init__(**data)
    
    def save_context(self, input_str: str, output_str: str) -> None:
        """Save conversation context with proper message types"""
        self.history.extend([
            HumanMessage(content=input_str),
            AIMessage(content=output_str)
        ])
    
    def load_memory_variables(self, inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        """Load memory in format expected by LangChain"""
        return {"chat_history": self.history}

    def get_history(self, num_exchanges: int = 3) -> str:
        """Get formatted conversation history"""
        recent_history = self.history[-num_exchanges*2:]
        return "\n".join(
            f"{'User' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content}"
            for msg in recent_history
        )

    def clear(self) -> None:
        """Clear memory"""
        self.history.clear()

    class Config:
        arbitrary_types_allowed = True
