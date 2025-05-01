from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse
import logging
import json
import os
import traceback
import asyncio
from datetime import datetime
from typing import Dict, Any, List

# Import from local modules
from models import ChatRequest, CalculatorSelectionRequest, ChatHistoryRequest, BrowserCalculatorRequest
from connection_manager import ConnectionManager
from llm_service import get_llm_response, update_conversation_context, client
from conversation_service import save_conversation, load_conversation, get_saved_conversations, load_calculator_map
from browser_calculator import run_browser_calculator

# Setup logging
logger = logging.getLogger(__name__)

# Create connection manager
manager = ConnectionManager()

# Create API router
router = APIRouter(prefix="/api")

# Routes
@router.get("/calculators", response_model=Dict[str, str])
async def get_calculators():
    """Get available medical calculators"""
    calculators = load_calculator_map()
    return calculators

@router.post("/chat", response_model=Dict[str, Any])
async def chat(request: ChatRequest):
    """Process a chat message from the user"""
    session_id = request.session_id
    message = request.message
    
    try:
        # Get the current session
        session = manager.get_session(session_id)
        if not session:
            return {"success": False, "error": "Session not found"}
        
        # Add user message to chat history
        manager.add_message(session_id, {"role": "user", "content": message})
        
        # Send acknowledgment to client (user message received)
        await manager.send_message(session_id, {"type": "message_received"})
        
        # Show typing indicator
        await manager.send_message(session_id, {"type": "typing_indicator"})
    
        # TRIGGER DETECTION:
        # Check if a calculator is selected and patient note is provided
        calculator_selected = bool(session.get("selected_calculator") and session.get("calculator_url"))
        
        # Check if this message looks like patient data (sufficient length and contains key medical terms)
        looks_like_patient_data = False
        if len(message.strip()) > 50:  # Minimum length for a patient note
            # Check for common medical terms or patterns in patient notes
            medical_terms = [
                'patient', 'year old', 'yo ', 'history', 'presents', 'diagnosed', 'exam', 
                'medical history', 'medications', 'vitals', 'assessment', 'labs', 'symptoms',
                'diagnosis', 'treatment', 'complained of', 'presented with'
            ]
            if any(term.lower() in message.lower() for term in medical_terms):
                looks_like_patient_data = True
                logger.info("Message detected as potential patient data")
        
        # LOG THE TRIGGERS
        logger.info(f"TRIGGER STATUS: calculator_selected={calculator_selected}, looks_like_patient_data={looks_like_patient_data}")
    
        # Auto-trigger browser calculator if both conditions are met
        if calculator_selected and looks_like_patient_data:
            calculator_name = session.get("selected_calculator", "")
            calculator_url = session.get("calculator_url", "")
            
            logger.info(f"==== TRIGGERING BROWSER CALCULATOR: {calculator_name} ====")
            
            # Use the message directly as patient data
            patient_data = str(message)
            
            # Immediately trigger browser_calculator.py without any intermediate response - no try/except to expose errors
            # Run browser calculator and get results
            calculator_result = await run_browser_calculator(
                calculator_name=calculator_name,
                calculator_url=calculator_url,
                patient_data=patient_data,
                llm_client=client
            )
            logger.info(f"Browser calculator returned: {calculator_result.get('success', False)}")
            
            # Process the results from browser_calculator.py
            if calculator_result.get("success", False):
                # Get screenshot and result from the calculator result
                result = calculator_result.get("result", {})
                score = result.get("score", "No score available")
                screenshot_url = calculator_result.get("screenshot_path")
                
                # Get any input justifications if provided by browser_calculator.py
                input_justifications = ""
                if "extracted_inputs" in calculator_result and "inputs" in calculator_result["extracted_inputs"]:
                    input_justifications = "\n"
                    for input_item in calculator_result["extracted_inputs"]["inputs"]:
                        if "justification" in input_item:
                            # Get the input name and value
                            input_name = next((key for key in input_item.keys() if key != "justification"), "Unknown")
                            input_value = input_item.get(input_name, "")

                            # Format the value properly
                            if isinstance(input_value, list) and len(input_value) > 0:
                                formatted_value = f"{input_value[0]} {input_value[1] if len(input_value) > 1 else ''}"
                            else:
                                formatted_value = str(input_value)
                            
                            # Add to the justifications text
                            input_justifications += f"- **{input_name}** ({formatted_value}): {input_item['justification']}\n"
                
                # Create response content with screenshot URL if available
                if screenshot_url:
                    response_content = f"""
## {calculator_name} Result

**Result:** {score}

**Input Values Used:**{input_justifications}

---
_Calculated using browser-based interaction with MDCalc.com_

![{calculator_name} Result]({screenshot_url})
"""
                else:
                    # No screenshot available
                    response_content = f"""
## {calculator_name} Result

**Result:** {score}

**Input Values Used:**{input_justifications}

---
_Calculated using browser-based interaction with MDCalc.com_
"""
                
                # Add the result message to chat history
                manager.add_message(session_id, {"role": "assistant", "content": response_content})
                
                # Send the response to the client
                await manager.send_message(session_id, {
                    "type": "message",
                    "data": {"role": "assistant", "content": response_content}
                })
                
                return {"success": True}
                
            # Check for missing values
            if "missing_values" in calculator_result:
                # Handle missing values case
                missing_values_str = ", ".join(calculator_result.get("missing_values", []))
                error_message = f"""## {calculator_name} Calculator Error

**Missing Required Values**: {missing_values_str}

I cannot complete this calculation because some required numerical values are missing from the patient data. Please provide the missing values and try again.

If you have this information, please include it in your next message.
"""
                # Add the error message to chat history
                manager.add_message(session_id, {"role": "assistant", "content": error_message})
                
                # Send the error message to the client
                await manager.send_message(session_id, {
                    "type": "message",
                    "data": {"role": "assistant", "content": error_message}
                })
                
                return {"success": True}

        # If triggers are not met or calculator process failed, use normal chatbot flow
        response = await get_llm_response(session, message)
    
        # Add the response to chat history
        manager.add_message(session_id, {"role": "assistant", "content": response})
    
        # Update conversation context
        update_conversation_context(session, message, response)
        
        # Send the message to the client
        await manager.send_message(session_id, {
            "type": "message",
            "data": {"role": "assistant", "content": response}
        })
        
        return {"success": True}
    except Exception as e:
        logger.error(f"Error processing chat message: {str(e)}")
        error_message = "I'm sorry, there was an error processing your message. Please try again."
        
        # Add error message to chat
        manager.add_message(session_id, {"role": "assistant", "content": error_message})
        
        # Send error message to client
        await manager.send_message(session_id, {
            "type": "message",
            "data": {"role": "assistant", "content": error_message}
        })
        
        return {"success": False, "error": str(e)}

@router.post("/select-calculator", response_model=Dict[str, Any])
async def select_calculator(request: CalculatorSelectionRequest):
    """Select a medical calculator"""
    session_id = request.session_id
    calculator_name = request.calculator_name
    calculator_url = request.calculator_url
    
    session = manager.get_session(session_id)
    
    # Update calculator information
    session["selected_calculator"] = calculator_name
    session["calculator_url"] = calculator_url
    
    # Create welcome message for the calculator with instruction about browser-augmented capabilities
    welcome_msg = f"""I'll help you with the **{calculator_name}** calculator.

This calculator uses browser automation to interact with MDCalc.com. This means:
1. I will NOT perform any calculations manually
2. When you provide patient data, I'll extract the necessary inputs
3. The system will automatically fill these inputs into the MDCalc.com browser interface
4. Results will be retrieved directly from MDCalc.com with a screenshot for verification

Simply provide the patient information needed for the calculator, and I'll use the browser integration to get the results. Do not ask me to calculate anything manually - all calculations will be performed through MDCalc.com."""
    
    # Add the welcome message
    if len(session["messages"]) <= 1:
        # If only intro message or no messages, reset and add welcome
        session["messages"] = []
        session["messages"].append({"role": "assistant", "content": welcome_msg})
    else:
        # If there's conversation history, keep it and add the new calculator message
        session["messages"].append({"role": "assistant", "content": welcome_msg})
    
    # Update the session
    manager.update_session(session_id, session)
    
    return {
        "success": True,
        "calculator": calculator_name,
        "message": welcome_msg
    }

@router.post("/save-conversation", response_model=Dict[str, Any])
async def save_chat(request: ChatHistoryRequest):
    """Save the current conversation"""
    session_id = request.session_id
    session = manager.get_session(session_id)
    file_path = save_conversation(session)
    
    if file_path:
        return {
            "success": True,
            "file_path": file_path,
            "message": f"Conversation saved to {file_path}"
        }
    else:
        return {
            "success": False,
            "message": "Failed to save conversation"
        }

@router.post("/new-conversation", response_model=Dict[str, Any])
async def new_conversation(request: ChatHistoryRequest):
    """Start a new conversation"""
    session_id = request.session_id
    success = manager.initialize_new_conversation(session_id)
    
    if success:
        return {
            "success": True,
            "message": "New conversation started"
        }
    else:
        return {
            "success": False,
            "message": "Session not found"
        }

@router.get("/conversations", response_model=List[Dict[str, Any]])
async def get_conversations():
    """Get list of saved conversations"""
    return get_saved_conversations()

@router.post("/load-conversation", response_model=Dict[str, Any])
async def load_chat(request: Request):
    """Load a saved conversation"""
    data = await request.json()
    file_name = data.get("file")
    session_id = data.get("session_id")
    
    if not file_name or not session_id:
        return {
            "success": False,
            "message": "Missing file name or session ID"
        }
    
    file_path = os.path.join("saved_chats", file_name)
    
    if not os.path.exists(file_path):
        return {
            "success": False,
            "message": f"File not found: {file_path}"
        }
    
    # Load the conversation
    conversation_data = load_conversation(file_path)
    
    if conversation_data:
        # Get the session
        session = manager.get_session(session_id)
        
        # Update the session with the loaded conversation
        session.update(conversation_data)
        
        # Set the conversation ID
        conversation_id = file_name.replace("chat_", "").replace(".json", "")
        session["conversation_id"] = conversation_id
        
        # Update the session
        manager.update_session(session_id, session)
        
        return {
            "success": True,
            "message": f"Conversation loaded",
            "messages": conversation_data.get("messages", []),
            "selected_calculator": conversation_data.get("selected_calculator", session.get("selected_calculator")),
            "calculator_url": conversation_data.get("calculator_url", session.get("calculator_url")),
            "model": conversation_data.get("model", session.get("model", "gpt-4o"))
        }
    else:
        return {
            "success": False,
            "message": "Failed to load conversation"
        }

@router.get("/session/{session_id}/history", response_model=Dict[str, Any])
async def get_chat_history(session_id: str):
    """Get the chat history for a session"""
    session = manager.get_session(session_id)
    
    if session:
        return {
            "success": True,
            "messages": session.get("messages", []),
            "selected_calculator": session.get("selected_calculator"),
            "calculator_url": session.get("calculator_url")
        }
    else:
        return {
            "success": False,
            "message": "Session not found"
        }

@router.post("/session/{session_id}/update-model", response_model=Dict[str, Any])
async def update_model(session_id: str, request: Request):
    """Update the model for a session"""
    data = await request.json()
    model = data.get("model")
    
    if not model:
        return {
            "success": False,
            "message": "Missing model"
        }
    
    session = manager.get_session(session_id)
    
    if session:
        # Update the model
        session["model"] = model
        
        # Update the session
        manager.update_session(session_id, session)
        
        return {
            "success": True,
            "message": f"Model updated to {model}"
        }
    else:
        return {
            "success": False,
            "message": "Session not found"
        }

@router.post("/browser-calculator", response_model=Dict[str, Any])
async def run_calculator_with_browser(request: BrowserCalculatorRequest):
    """Run a calculator interaction using browser agent"""
    try:
        # Pass the client to the browser_calculator function
        result = await run_browser_calculator(
            calculator_name=request.calculator_name,
            calculator_url=request.calculator_url,
            patient_data=request.patient_data,
            llm_client=client
        )
        
        # Check if successful
        if not result.get("success", False):
            # Special handling for missing values
            if "missing_values" in result:
                missing_values_str = ", ".join(result.get("missing_values", []))
                error_message = f"""## {request.calculator_name} Calculator Error

**Missing Required Values**: {missing_values_str}

I cannot complete this calculation because some required numerical values are missing from the patient data. Please provide the missing values and try again.

If you have this information, please include it in your next message.
"""
                # Add the error message to chat history
                session_id = request.session_id
                session = manager.get_session(session_id)
                manager.add_message(session_id, {"role": "assistant", "content": error_message})
                
                return {
                    "success": False,
                    "message": error_message,
                    "missing_values": result.get("missing_values", [])
                }
            else:
                # Generic error
                return JSONResponse(
                    status_code=500,
                    content={"success": False, "error": result.get("error", "Unknown error")}
                )
        
        # Get the session
        session_id = request.session_id
        session = manager.get_session(session_id)
        
        # Update calculator information
        session["selected_calculator"] = request.calculator_name
        session["calculator_url"] = request.calculator_url
        
        # Update conversation context
        calculation = {
            "calculator": request.calculator_name,
            "prompt": request.patient_data,
            "response": result.get("result", {}).get("interpretation", ""),
            "score": result.get("result", {}).get("score", ""),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        session["conversation_context"]["current_calculation"] = calculation
        session["conversation_context"]["previous_calculations"].append(calculation)
        
        # Add the result to the chat history
        calculator_result = result.get("result", {})
        score = calculator_result.get("score", "")
        screenshot_url = None
        input_justifications = ""
        
        # Get input justifications if available
        if "extracted_inputs" in result and "inputs" in result["extracted_inputs"]:
            input_justifications = "\n"
            for input_item in result["extracted_inputs"]["inputs"]:
                if "justification" in input_item:
                    # Get the input name and value
                    input_name = next((key for key in input_item.keys() if key != "justification"), "Unknown")
                    input_value = input_item.get(input_name, "")
        
                    # Format the value properly
                    if isinstance(input_value, list) and len(input_value) > 0:
                        formatted_value = f"{input_value[0]} {input_value[1] if len(input_value) > 1 else ''}"
                    else:
                        formatted_value = str(input_value)
                    
                    # Add to the justifications text
                    input_justifications += f"- **{input_name}** ({formatted_value}): {input_item['justification']}\n"
        
        # Add screenshot if available
        if "screenshot_path" in result and result["screenshot_path"]:
            try:
                screenshot_url = result["screenshot_path"]
                
                # Create response content with screenshot URL
                response_content = f"""
## {request.calculator_name} Result

**Result:** {score}

**Input Values Used:**{input_justifications}

---
_Calculated using browser-based interaction with MDCalc.com_

![{request.calculator_name} Result]({screenshot_url})
"""
                logger.info(f"Added screenshot URL to response: {screenshot_url}")
                
                # Add the response to chat history
                manager.add_message(session_id, {"role": "assistant", "content": response_content})
                
                # We've already added the response with screenshot, so return immediately
                return {
                    "success": True,
                    "result": calculator_result,
                    "extracted_inputs": result.get("extracted_inputs", {}),
                    "message": response_content,
                    "screenshot_url": screenshot_url
                }
            except Exception as e:
                logger.error(f"Error processing screenshot: {e}")
                screenshot_url = None
        
        # Fallback response without screenshot
        result_message = f"""
## {request.calculator_name} Result

**Result:** {score}

**Input Values Used:**{input_justifications}

---
_Calculated using browser-based interaction with MDCalc.com_
"""
        
        # Add the result message to chat history
        manager.add_message(session_id, {"role": "assistant", "content": result_message})
        
        # Update the session
        manager.update_session(session_id, session)
        
        # Return the result
        return {
            "success": True,
            "result": calculator_result,
            "extracted_inputs": result.get("extracted_inputs", {}),
            "message": result_message,
            "screenshot_url": screenshot_url
        }
        
    except Exception as e:
        logger.error(f"Error in browser calculator endpoint: {str(e)}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        ) 