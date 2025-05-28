"""Base tool interface for all research tools."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)


class ToolInput(BaseModel):
    """Input data structure for tools."""
    query: str
    context: Optional[Dict[str, Any]] = None


class ToolOutput(BaseModel):
    """Output data structure from tools."""
    result: str
    source: str
    confidence: float
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class BaseTool(ABC):
    """Abstract base class for all research tools."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    @abstractmethod
    async def execute(self, input_data: ToolInput) -> ToolOutput:
        """
        Execute the tool with given input.
        
        Args:
            input_data: The input data for the tool
            
        Returns:
            ToolOutput: The result of the tool execution
        """
        pass
    
    def _create_error_output(self, error_message: str) -> ToolOutput:
        """Create a standardized error output."""
        return ToolOutput(
            result="",
            source=self.name,
            confidence=0.0,
            error=error_message
        )
    
    def is_relevant(self, query: str) -> bool:
        """
        Determine if this tool is relevant for the given query.
        Override in subclasses for more sophisticated matching.
        
        Args:
            query: The user query
            
        Returns:
            bool: True if tool is relevant
        """
        return True
    
    def __str__(self) -> str:
        return f"{self.name}: {self.description}"