from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import logging
import json
from datetime import datetime
from typing import Dict, Any
import traceback
import asyncio

# Import from local modules
from connection_manager import ConnectionManager
from llm_service import get_llm_response, update_conversation_context
from browser_calculator import run_browser_calculator
from llm_service import client

# Setup logging
logger = logging.getLogger(__name__)

# Create connection manager (use the same instance as api.py)
from routes.api import manager

# Create router
router = APIRouter()

@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time chat"""
    await manager.connect(websocket, session_id)
    
    try:
        # Send the initial chat history
        session = manager.get_session(session_id)
        await websocket.send_json({
            "type": "history",
            "data": {
                "messages": session.get("messages", []),
                "selected_calculator": session.get("selected_calculator"),
                "calculator_url": session.get("calculator_url")
            }
        })
        
        while True:
            # Wait for messages from the client
            data = await websocket.receive_json()
            message_type = data.get("type")
            
            if message_type == "chat":
                # Process chat message
                user_message = data.get("message")
                
                # Check if calculator is selected
                if not session.get("selected_calculator"):
                    await websocket.send_json({
                        "type": "error",
                        "data": "Please select a calculator first."
                    })
                    continue
                
                # Add user message to history
                manager.add_message(session_id, {"role": "user", "content": user_message})
                
                # Send immediate acknowledgment
                await websocket.send_json({
                    "type": "message_received",
                    "data": {
                        "message": user_message,
                        "timestamp": datetime.now().isoformat()
                    }
                })
                
                # TRIGGER DETECTION:
                # Check if a calculator is selected and patient note is provided
                calculator_selected = bool(session.get("selected_calculator") and session.get("calculator_url"))
                
                # Check if this message looks like patient data (sufficient length and contains key medical terms)
                looks_like_patient_data = False
                if len(user_message.strip()) > 50:  # Minimum length for a patient note
                    # Check for common medical terms or patterns in patient notes
                    medical_terms = [
                        'patient', 'year old', 'yo ', 'history', 'presents', 'diagnosed', 'exam', 
                        'medical history', 'medications', 'vitals', 'assessment', 'labs', 'symptoms',
                        'diagnosis', 'treatment', 'complained of', 'presented with'
                    ]
                    if any(term.lower() in user_message.lower() for term in medical_terms):
                        looks_like_patient_data = True
                        logger.info("Message detected as potential patient data")
                
                # LOG THE TRIGGERS WITH MORE DETAIL
                logger.info(f"WEBSOCKET TRIGGER STATUS: calculator_selected={calculator_selected}, looks_like_patient_data={looks_like_patient_data}")
                logger.info(f"Calculator info: {session.get('selected_calculator', 'None')}, URL: {session.get('calculator_url', 'None')}")
                logger.info(f"Message length: {len(user_message.strip())}")
            
                # Auto-trigger browser calculator if both conditions are met
                if calculator_selected and looks_like_patient_data:
                    calculator_name = session.get("selected_calculator", "")
                    calculator_url = session.get("calculator_url", "")
                    
                    logger.info(f"==== WEBSOCKET TRIGGERING BROWSER CALCULATOR: {calculator_name} ====")
                    
                    # Use the message directly as patient data
                    message_content = str(user_message)
                    
                    # Show typing indicator
                    await websocket.send_json({"type": "typing_indicator"})
                    
                    # Immediately trigger browser_calculator.py without any intermediate response
                    # No try/except to expose errors
                    try:
                        # Run the browser calculator
                        calculator_result = await run_browser_calculator(
                            calculator_name=calculator_name,
                            calculator_url=calculator_url,
                            patient_data=message_content,
                            llm_client=client
                        )
                        
                        # Check if the result is None
                        if calculator_result is None:
                            logger.error("Browser calculator returned None")
                            calculator_result = {"success": False, "error": "Browser calculator returned None"}
                        
                        # Log the result
                        logger.info(f"Browser calculator returned: {calculator_result.get('success', False)}")
                        
                        # Send result back to client
                        await websocket.send_json(calculator_result)
                        
                    except Exception as e:
                        # Log error and send error response
                        logger.error(f"WebSocket error: {str(e)}")
                        logger.error(f"WebSocket error traceback: {traceback.format_exc()}")
                        error_response = {"success": False, "error": str(e)}
                        await websocket.send_json(error_response)
                        
                    # Process the results from browser_calculator.py
                    if calculator_result.get("success", False):
                        # ------------------
                        # Build rich chat reply with:
                        #  • numerical answer
                        #  • justification list for each extracted input
                        #  • embedded screenshot image
                        # ------------------

                        answer_val = calculator_result.get("answer", "N/A")

                        # Gather input justifications
                        input_justifications = ""
                        if "extracted_inputs" in calculator_result:
                            extracted = calculator_result["extracted_inputs"]

                            # Sometimes wrapped in a dict with "inputs" key
                            if isinstance(extracted, dict) and "inputs" in extracted:
                                extracted = extracted["inputs"]

                            if isinstance(extracted, list):
                                for input_item in extracted:
                                    if isinstance(input_item, dict) and "justification" in input_item:
                                        # Determine the field name (first key that is not 'justification')
                                        input_name = next((k for k in input_item.keys() if k != "justification"), "Unknown")
                                        input_value = input_item.get(input_name, "")

                                        # Format value (handle lists like [126, "in"])
                                        if isinstance(input_value, list):
                                            formatted_value = " ".join(str(v) for v in input_value)
                                        else:
                                            formatted_value = str(input_value)

                                        input_justifications += f"- **{input_name}** ({formatted_value}): {input_item['justification']}\n"

                        if not input_justifications:
                            input_justifications = "No justifications returned."

                        # Prefer the relative path for the screenshot so the browser can serve it
                        screenshot_url = (
                            calculator_result.get("relative_screenshot_path")
                            or calculator_result.get("screenshot_path")
                            or ""
                        )

                        response_content = f"""
## {calculator_name} Result

**Answer:** {answer_val}

**Input Justifications:**\n{input_justifications}

---
_Calculated using browser-based interaction with MDCalc.com_

{'![Calculator Result](' + screenshot_url + ')' if screenshot_url else ''}
"""
                        
                        # Add the result message to chat history
                        manager.add_message(session_id, {"role": "assistant", "content": response_content})
                        
                        # Add a small delay to ensure message is sent before any potential reload
                        await asyncio.sleep(1.0)
                        
                        # Send the response to the client
                        await websocket.send_json({
                            "type": "message",
                            "data": {"role": "assistant", "content": response_content}
                        })
                        
                        # Ensure the client has time to process the message
                        await asyncio.sleep(0.5)
                        
                        # Skip the normal LLM response since we've already handled it
                        continue
                        
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
                        await websocket.send_json({
                            "type": "message",
                            "data": {"role": "assistant", "content": error_message}
                        })
                        
                        # Skip the normal LLM response since we've already handled it
                        continue
                
                # If triggers are not met or calculator process failed, use normal chatbot flow
                response = await get_llm_response(session, user_message)
                
                # Update context
                update_conversation_context(session, user_message, response)
                
                # Add assistant response to history
                manager.add_message(session_id, {"role": "assistant", "content": response})
                
                # Send the response to the client
                await websocket.send_json({
                    "type": "message",
                    "data": {
                        "role": "assistant",
                        "content": response,
                        "timestamp": datetime.now().isoformat()
                    }
                })
                
            elif message_type == "select_calculator":
                # Process calculator selection
                calculator_name = data.get("calculator_name")
                calculator_url = data.get("calculator_url")
                
                # Update calculator information
                session["selected_calculator"] = calculator_name
                session["calculator_url"] = calculator_url
                
                # Create welcome message for the calculator
                welcome_msg = f"I'll help you with the **{calculator_name}** calculator. What patient data would you like to calculate?"
                
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
                
                # Send confirmation to the client
                await websocket.send_json({
                    "type": "calculator_selected",
                    "data": {
                        "calculator": calculator_name,
                        "message": welcome_msg
                    }
                })
                
            elif message_type == "new_conversation":
                # Start a new conversation
                success = manager.initialize_new_conversation(session_id)
                if success:
                    session = manager.get_session(session_id)
                    # Send confirmation to the client
                    await websocket.send_json({
                        "type": "new_conversation",
                        "data": {
                            "messages": session["messages"]
                        }
                    })
                else:
                    await websocket.send_json({
                        "type": "error",
                        "data": "Failed to create new conversation"
                    })
                    
            elif message_type == "save_conversation":
                # Import save_conversation function
                from conversation_service import save_conversation
                
                # Save the conversation
                session = manager.get_session(session_id)
                file_path = save_conversation(session)
                
                if file_path:
                    await websocket.send_json({
                        "type": "conversation_saved",
                        "data": {
                            "file_path": file_path,
                            "message": f"Conversation saved to {file_path}"
                        }
                    })
                else:
                    await websocket.send_json({
                        "type": "error",
                        "data": "Failed to save conversation"
                    })
                    
            elif message_type == "update_model":
                # Update the model
                model = data.get("model")
                
                if model:
                    session["model"] = model
                    manager.update_session(session_id, session)
                    
                    await websocket.send_json({
                        "type": "model_updated",
                        "data": {
                            "model": model
                        }
                    })
                else:
                    await websocket.send_json({
                        "type": "error",
                        "data": "Missing model"
                    })
                    
    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {session_id}")
        manager.disconnect(session_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        logger.error(f"WebSocket error traceback: {traceback.format_exc()}")
        manager.disconnect(session_id) 