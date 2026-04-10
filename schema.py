from typing import TypedDict, Literal
from pydantic import BaseModel

class AgentState(TypedDict):
    query: str
    rewritten_query: str
    use_history: bool
    route: str
    docs: list
    answer: str
    check: bool
    not_grounded_explanation: str
    source: str
    retries: int
    web_retries: int
    chat_history: list
    no_docs: bool
    
    
class RouteDecision(BaseModel):
    route: Literal["rag", "chat", "end"]
    
class ContextDecision(BaseModel):
    use_history: bool
    
class GroundingCheck(BaseModel):
    grounded: bool
    explanation: str