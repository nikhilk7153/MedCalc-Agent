from fastapi import WebSocket
from typing import Dict, Any
from datetime import datetime
import logging

# Setup logging
logger = logging.getLogger(__name__)

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.chat_sessions: Dict[str, Dict[str, Any]] = {}
        
    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        
        # Initialize session if it doesn't exist
        if session_id not in self.chat_sessions:
            self.chat_sessions[session_id] = {
                "messages": [],
                "conversation_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "model": "gpt-4o",
                "selected_calculator": None,
                "calculator_url": None,
                "conversation_context": {
                    "current_calculation": None,
                    "patient_data": {},
                    "previous_calculations": []
                }
            }
            
            # Add initial welcome message
            self.chat_sessions[session_id]["messages"].append({
                "role": "assistant", 
                "content": "👋 Hi, I'm MedCalc-Agent! I am a browser-augmented LLM agent who can help you with medical calculations and risk assessments! Please select a calculator from the sidebar to get started. 😊"
            })
    
    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
    
    async def send_message(self, session_id: str, message: Dict[str, Any]):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_json(message)
    
    def get_session(self, session_id: str) -> Dict[str, Any]:
        return self.chat_sessions.get(session_id, {})
    
    def update_session(self, session_id: str, data: Dict[str, Any]):
        if session_id in self.chat_sessions:
            self.chat_sessions[session_id].update(data)
            
    def add_message(self, session_id: str, message: Dict[str, str]):
        if session_id in self.chat_sessions:
            self.chat_sessions[session_id]["messages"].append(message)
            return True
        return False
    
    def initialize_new_conversation(self, session_id: str):
        """Reset session to start a new conversation"""
        session = self.get_session(session_id)
        
        if session:
            # Reset the session
            session["messages"] = []
            session["conversation_id"] = datetime.now().strftime("%Y%m%d_%H%M%S")
            session["selected_calculator"] = None
            session["calculator_url"] = None
            session["conversation_context"] = {
                "current_calculation": None,
                "patient_data": {},
                "previous_calculations": []
            }
            
            # Add initial welcome message
            session["messages"].append({
                "role": "assistant", 
                "content": "👋 Hi, I'm MedCalc-Agent! I am a browser-augmented LLM agent who can help you with medical calculations and risk assessments! Please select a calculator from the sidebar to get started. 😊"
            })
            
            # Update the session
            self.update_session(session_id, session)
            return True
        return False 