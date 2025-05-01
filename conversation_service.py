import os
import json
import logging
from typing import Dict, Any
from datetime import datetime

# Setup logging
logger = logging.getLogger(__name__)

# Directory for saving chats
CHATS_DIR = "saved_chats"
os.makedirs(CHATS_DIR, exist_ok=True)

def save_conversation(session: Dict[str, Any]) -> str | None:
    """Save the current conversation to a JSON file"""
    if not session or not session.get("messages"):
        return None
    
    # Create a dictionary with conversation data including settings
    conversation_data = {
        "messages": session["messages"],
        "settings": {
            "model": session.get("model", "gpt-4o"),
            "selected_calculator": session.get("selected_calculator"),
            "calculator_url": session.get("calculator_url")
        },
        "context": session.get("conversation_context", {})
    }
    
    filename = f"{CHATS_DIR}/chat_{session.get('conversation_id', datetime.now().strftime('%Y%m%d_%H%M%S'))}.json"
    try:
        with open(filename, 'w') as f:
            json.dump(conversation_data, f)
        return filename
    except Exception as e:
        logger.error(f"Error saving conversation: {e}")
        return None

def load_conversation(file_path: str) -> Dict[str, Any]:
    """Load a conversation from a JSON file"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Handle both new and old format
        if isinstance(data, dict) and "messages" in data:
            result = {
                "messages": data["messages"]
            }
            
            # Load settings if available
            if "settings" in data:
                settings = data["settings"]
                if "model" in settings:
                    result["model"] = settings["model"]
                if "selected_calculator" in settings:
                    result["selected_calculator"] = settings["selected_calculator"]
                if "calculator_url" in settings:
                    result["calculator_url"] = settings["calculator_url"]
            
            # Load context if available
            if "context" in data:
                result["conversation_context"] = data["context"]
            
            return result
        else:
            # Legacy format (just a list of messages)
            return {
                "messages": data
            }
            
    except Exception as e:
        logger.error(f"Error loading conversation: {e}")
        return {}

def get_saved_conversations():
    """Get list of saved conversations"""
    if os.path.exists(CHATS_DIR):
        saved_files = [f for f in os.listdir(CHATS_DIR) if f.endswith('.json')]
        conversations = []
        
        for file in saved_files:
            file_path = os.path.join(CHATS_DIR, file)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Get conversation info
                conversation_id = file.replace("chat_", "").replace(".json", "")
                calculator = "Unknown"
                timestamp = conversation_id.split("_")[0]
                
                # Try to extract calculator name
                if "settings" in data and "selected_calculator" in data["settings"]:
                    calculator = data["settings"]["selected_calculator"] or "Unknown"
                
                conversations.append({
                    "id": conversation_id,
                    "file": file,
                    "calculator": calculator,
                    "timestamp": timestamp,
                    "message_count": len(data.get("messages", [])),
                })
            except Exception as e:
                logger.error(f"Error loading conversation {file}: {e}")
        
        return conversations
    else:
        return []

def load_calculator_map():
    """Load medical calculators from JSON file"""
    try:
        with open("calculator_map.json", "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading calculators: {e}")
        return {} 