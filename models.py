from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# Define Pydantic models for browser interaction
class CalculatorInput(BaseModel):
    name: str
    type: str
    options: Optional[List[str]] = None
    element_index: Optional[int] = None
    option_indexes: Optional[Dict[str, int]] = None
    units: Optional[List[str]] = None
    description: Optional[str] = None

class CalculatorInputs(BaseModel):
    inputs: List[CalculatorInput]

class CalculatorResult(BaseModel):
    calculator_name: str
    score: str
    interpretation: str

class FormActionPlan(BaseModel):
    form_actions: List[Dict[str, Any]]

# Request/Response models
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    session_id: str

class CalculatorSelectionRequest(BaseModel):
    calculator_name: str
    calculator_url: str
    session_id: str

class ChatHistoryRequest(BaseModel):
    session_id: str

class BrowserCalculatorRequest(BaseModel):
    calculator_name: str
    calculator_url: str
    patient_data: str
    session_id: str 