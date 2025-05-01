import os
import re
import logging
from typing import Dict, Any, List
from openai import AzureOpenAI
from dotenv import load_dotenv
from datetime import datetime

# Setup logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_version='2024-02-15-preview',
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://azure-openai-miblab-ncu.openai.azure.com/"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY", "9a8467bfe81d443d97b1af452662c33c"),
)

async def get_llm_response(session: Dict[str, Any], prompt: str):
    """Get response from Azure OpenAI using selected model"""
    try:
        # Format the messages for the API call
        messages = []
        
        # Add system message based on selected calculator
        system_message = get_calculator_system_prompt(
            session.get("selected_calculator"),
            session.get("calculator_url")
        )
        messages.append({"role": "system", "content": system_message})
        
        # Add special instruction message about browser calculator
        calculator_name = session.get("selected_calculator")
        if calculator_name:
            instruction_msg = f"""IMPORTANT REMINDER: For the {calculator_name} calculator, I must NEVER perform calculations manually. 
All calculations will be performed through browser integration with MDCalc.com.
When the user provides patient data, I should acknowledge it and wait for the browser calculator to process it.
I should NOT show formulas or step-by-step calculations."""
            messages.append({"role": "system", "content": instruction_msg})
        
        # Add conversation history - include up to last 15 messages to avoid token limits
        recent_messages = session["messages"][-15:] if len(session["messages"]) > 15 else session["messages"]
        for msg in recent_messages:
            messages.append({"role": msg["role"], "content": msg["content"]})
        
        # Add the current user prompt
        messages.append({"role": "user", "content": prompt})
        
        # Call the Azure OpenAI API with the selected model
        response = client.chat.completions.create(
            model=session.get("model", "gpt-4o"),
            messages=messages,
            max_tokens=2000
        )
        
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error calling Azure OpenAI API: {e}")
        return f"I'm sorry, there was an error communicating with the AI service: {str(e)}"

def get_calculator_system_prompt(calculator_name, calculator_url):
    """Generate a system prompt for the selected calculator"""
    base_prompt = """You are MedCalc-Agent, a helpful medical AI assistant specialized in clinical calculations and risk assessments. You have the following capabilities:

    1. Perform medical calculations and risk assessments using MDCalc.com browser integration
    2. Explain medical concepts clearly and concisely
    3. Interpret calculation results in a clinical context
    4. Suggest appropriate follow-up actions based on results
    5. Maintain a professional, helpful tone

    EXTREMELY IMPORTANT: You must NEVER compute medical calculations yourself. Always use the browser-based calculator integration with MDCalc.com. The browser calculator will extract required inputs from patient information, perform the calculation, and provide accurate results.

    Guidelines for your responses:
    1. NEVER perform manual calculations - rely on the browser calculator for all computations
    2. Do not show calculation steps or formulas - the browser calculator handles all calculations
    3. When presented with patient data, extract relevant parameters but do NOT calculate results
    4. Do not provide step-by-step manual calculations even if you know the formula
    5. Interpret results with appropriate clinical context once the browser calculator provides them
    6. Be clear about the limitations of calculators
    7. If patient data is incomplete, ask for missing information rather than making assumptions
    8. Use appropriate units (metric or imperial) based on user input
    9. Structure your responses with clearly labeled sections
    10. If a calculation could be used inappropriately, note any contraindications
    11. Remember that medical calculators are decision aids, not replacements for clinical judgment
    12. Always inform the user that calculations are performed using browser-augmented functionality through MDCalc.com
    """
    
    if calculator_name and calculator_url:
        specific_calculator_prompt = f"\n\nYou are currently set to use the {calculator_name} calculator from MDCalc.com. All calculations will be performed using browser interaction with {calculator_url}. DO NOT perform any calculations manually - the system will use the browser calculator to get results."
        return f"{base_prompt}{specific_calculator_prompt}"
    
    return base_prompt

def extract_patient_data(prompt):
    """Extract potential patient data from user prompts to maintain context"""
    # This is a simple implementation - in production, you would use NER or structured data extraction
    data_points = {}
    
    # Look for common patterns
    
    # Age
    age_match = re.search(r'(\d+)[- ](?:year|yo|y)[s]?[- ]?(?:old)?', prompt.lower())
    if age_match:
        data_points["age"] = int(age_match.group(1))
    
    # Gender
    if re.search(r'\b(?:male|man|boy|gentleman)\b', prompt.lower()):
        data_points["gender"] = "male"
    elif re.search(r'\b(?:female|woman|girl|lady)\b', prompt.lower()):
        data_points["gender"] = "female"
    
    # Height (cm or inches)
    height_cm = re.search(r'(\d+(?:\.\d+)?)\s*(?:cm|centimeters?)', prompt.lower())
    height_in = re.search(r'(\d+(?:\.\d+)?)\s*(?:in(?:ches)?|\'|feet)', prompt.lower())
    if height_cm:
        data_points["height_cm"] = float(height_cm.group(1))
    elif height_in:
        # Approximate conversion
        data_points["height_cm"] = float(height_in.group(1)) * 2.54
    
    # Weight (kg or lbs)
    weight_kg = re.search(r'(\d+(?:\.\d+)?)\s*(?:kg|kilograms?)', prompt.lower())
    weight_lb = re.search(r'(\d+(?:\.\d+)?)\s*(?:lb|pounds?)', prompt.lower())
    if weight_kg:
        data_points["weight_kg"] = float(weight_kg.group(1))
    elif weight_lb:
        # Approximate conversion
        data_points["weight_kg"] = float(weight_lb.group(1)) * 0.453592
    
    return data_points

def update_conversation_context(session: Dict[str, Any], prompt: str, response: str):
    """Update the conversation context with information from the current exchange"""
    if not session:
        return
        
    # Extract patient data from the prompt
    new_data = extract_patient_data(prompt)
    
    # Update patient data
    if new_data:
        session["conversation_context"]["patient_data"].update(new_data)
    
    # Check if this was a calculation (simple heuristic)
    if any(word in response.lower() for word in ["score", "result", "calculation", "formula"]):
        # Store the current calculation
        calculation = {
            "calculator": session.get("selected_calculator"),
            "prompt": prompt,
            "response": response,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        session["conversation_context"]["current_calculation"] = calculation
        
        # Add to previous calculations if not already there
        found = False
        for prev_calc in session["conversation_context"]["previous_calculations"]:
            if prev_calc["prompt"] == prompt and prev_calc["calculator"] == session.get("selected_calculator"):
                found = True
                break
        
        if not found:
            session["conversation_context"]["previous_calculations"].append(calculation)
            
    return session 