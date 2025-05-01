from langchain_openai import ChatOpenAI, AzureChatOpenAI
from browser_use import Agent, Controller
from dotenv import load_dotenv
import os
load_dotenv()
import asyncio
from browser_use.browser.browser import Browser, BrowserConfig, BrowserContextConfig
from openai import AzureOpenAI
import json
from typing import List, Optional, Dict, Any, Union, Tuple
from pydantic import BaseModel, Field
from browser_use.controller.views import (
    InputTextAction,
    ClickElementAction,
    ScrollAction
)
import logging
import aiofiles
import traceback
import sys
import re
import datetime

# Import extraction functionality
try:
    from extractor import extract_numerical_answer, save_results_to_json, process_calculation_result
except ImportError:
    # Define fallback extraction functions if import fails
    def extract_numerical_answer(result_str):
        try:
            # Look for decimal number
            match = re.search(r'(\d+\.\d+)', result_str)
            if match:
                return float(match.group(1))
            # Look for integer
            match = re.search(r'(\d+)', result_str)
            if match:
                return float(match.group(1))
        except Exception:
            pass
        return None
        
    def save_results_to_json(result_dict, calculator_name=None):
        try:
            os.makedirs('results', exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results/calculation_{timestamp}.json"
            with open(filename, 'w') as f:
                json.dump(result_dict, f, indent=2)
            return filename
        except Exception:
            return None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define Pydantic models for structured output
class CalculatorInput(BaseModel):
    name: str
    type: str
    options: Optional[List[str]] = None
    element_index: Optional[int] = None  # Numeric index for the element in the selector_map
    option_indexes: Optional[Dict[str, int]] = None  # Maps option text to its numeric index
    units: Optional[List[str]] = None
    description: Optional[str] = None

class CalculatorInputs(BaseModel):
    inputs: List[CalculatorInput]

# Model for the intermediate step (parsing extracted data)
class ExtractedValue(BaseModel):
    value: Union[str, List[Union[float, str]]]

class ExtractedValues(BaseModel):
    inputs: List[Dict[str, ExtractedValue]]

class CalculatorResult(BaseModel):
    calculator_name: str
    score: str

# Define the form action plan model
class FormActionPlan(BaseModel):
    form_actions: List[Dict[str, Any]]

# Define the result controller
class ResultController:
    async def parse_llm_response(self, llm_response, history):
        return llm_response

# Function to create a structured representation of calculator inputs
def create_input_summary(calculator_inputs: CalculatorInputs) -> Dict[str, Dict[str, Any]]:
    """
    Converts the CalculatorInputs object into a simplified dictionary with input names as keys
    and their options, types, units, and descriptions as values.
    
    Args:
        calculator_inputs: A CalculatorInputs object containing the parsed inputs
        
    Returns:
        A dictionary with the format:
        {
            "Input Name": {
                "type": "option" or "numeric",
                "options": ["Option1", "Option2"] or None,
                "units": ["unit"] or None,
                "description": "Description text" or None
            },
            ...
        }
    """
    result = {}
    
    for input_item in calculator_inputs.inputs:
        input_data: Dict[str, Any] = {
            "type": input_item.type
        }
        
        if input_item.options:
            input_data["options"] = input_item.options
            
        if input_item.units:
            input_data["units"] = input_item.units
            
        if input_item.description:
            input_data["description"] = input_item.description
            
        result[input_item.name] = input_data
        
    return result

# Function to create a mapping of input names to their element IDs
def create_element_id_mapping(calculator_inputs: CalculatorInputs) -> Dict[str, Any]:
    """
    Creates a mapping between input names and their element IDs for both text inputs and 
    radio button/dropdown options.
    
    Args:
        calculator_inputs: A CalculatorInputs object containing the parsed inputs
        
    Returns:
        A dictionary with the format:
        {
            "Input Name": element_id,  # For numeric/text inputs
            "Option Input Name": {
                "Option1": element_id_1,
                "Option2": element_id_2,
                ...
            },  # For radio buttons/dropdowns
            ...
        }
    """
    mapping = {}
    
    for input_item in calculator_inputs.inputs:
        if input_item.type == "numeric" and input_item.element_index is not None:
            # For numeric/text inputs, map directly to the element index
            mapping[input_item.name] = input_item.element_index
        elif input_item.type == "option" and input_item.option_indexes:
            # For radio buttons/dropdowns, map to a dictionary of options and their indices
            mapping[input_item.name] = input_item.option_indexes
    
    return mapping

# Remove hardcoded Azure credentials and load from environment
from dotenv import load_dotenv
load_dotenv()

azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

# Then create your LLM
llm = AzureChatOpenAI(
    model="gpt-4o",
    api_version='2024-02-15-preview',
    azure_endpoint=azure_endpoint,
    api_key=azure_api_key,
)

client = AzureOpenAI(
    api_version='2024-02-15-preview',
    azure_endpoint=azure_endpoint,
    api_key=azure_api_key,
)

# Create controller with output model for structured output
input_controller = Controller()
result_controller = Controller(output_model=CalculatorResult)
action_plan_controller = Controller(output_model=FormActionPlan)

# Function to transform dictionary options to lists if needed
def transform_options(inputs_data):
    if isinstance(inputs_data, dict) and "inputs" in inputs_data:
        transformed_inputs = []
        for input_item in inputs_data["inputs"]:
            if isinstance(input_item, dict):
                transformed_item = CalculatorInput(
                    name=input_item.get("name", ""),
                    type=input_item.get("type", ""),
                    element_index=input_item.get("element_index", None),
                    description=input_item.get("description", None)
                )
                
                # Handle options
                if "options" in input_item:
                    if isinstance(input_item["options"], dict):
                        # Convert dict options to list
                        options_list = list(input_item["options"].keys())
                        option_indexes = {}
                        # Try to preserve indexes if they exist
                        if "option_indexes" in input_item:
                            option_indexes = input_item["option_indexes"]
                        transformed_item.options = options_list
                        transformed_item.option_indexes = option_indexes
                    else:
                        transformed_item.options = input_item["options"]
                        if "option_indexes" in input_item:
                            transformed_item.option_indexes = input_item["option_indexes"]
                
                # Handle units
                if "units" in input_item:
                    transformed_item.units = input_item["units"]
                
                transformed_inputs.append(transformed_item.model_dump(exclude_none=True))
        
        return {"inputs": transformed_inputs}
    return inputs_data

async def run_workflow_calculator(calculator_name: str, calculator_url: str, patient_data: str, llm_client):
    """
    Run a workflow-based calculator interaction with MDCalc.com
    
    Args:
        calculator_name: the name of the calculator to run
        calculator_url: the URL of the calculator to use
        patient_data: patient case details to use for the calculator
        llm_client: an initialized OpenAI client
        
    Returns:
        Dict with calculation result
    """
    browser = None
    max_retries = 2
    retry_count = 0
    
    while retry_count <= max_retries:
        try:
            # Setup logging
            logger.info(f"===== STARTING workflow calculator for {calculator_name} =====")
            logger.info(f"Patient data: {patient_data}")
            logger.info(f"Calculator URL: {calculator_url}")
            
            # 1. Launch browser with appropriate configuration
            try:
                logger.info(f"Launching browser (attempt {retry_count + 1}/{max_retries + 1})...")
    browser = Browser(
        config=BrowserConfig(
                        headless=False,  # Always use headless mode for reliability
            disable_security=True,
        )
    )
                logger.info("Browser launched successfully")
            except Exception as e:
                logger.error(f"Failed to launch browser: {str(e)}")
                if retry_count < max_retries:
                    retry_count += 1
                    logger.info(f"Retrying browser launch (attempt {retry_count + 1}/{max_retries + 1})...")
                    continue
                return {"success": False, "error": f"Browser launch failed: {str(e)}"}
            
            # Create a browser context - using the async context manager pattern
            try:
                logger.info("Creating browser context...")
    async with await browser.new_context() as context:
                    logger.info("Browser context created")
                    
                    try:
                        page = await context.get_current_page()
                        logger.info("Got current page")
                    except Exception as e:
                        logger.error(f"Failed to get page: {str(e)}")
                        if retry_count < max_retries:
                            retry_count += 1
                            continue
                        return {"success": False, "error": f"Page creation failed: {str(e)}"}
                    
                    # Navigate to calculator
                    try:
                        logger.info(f"Navigating to {calculator_url}")
                        await page.goto(calculator_url)
                        logger.info("Page loaded, waiting for network idle")
                        await page.wait_for_load_state("networkidle")
                        logger.info("Network is idle")
                        
                        # Add a longer wait time after page load to ensure stability
                        logger.info("Waiting additional time for page to stabilize...")
                        await asyncio.sleep(3)
                    except Exception as e:
                        logger.error(f"Failed to navigate to {calculator_url}: {str(e)}")
                        if retry_count < max_retries:
                            retry_count += 1
                            continue
                        return {"success": False, "error": f"Navigation failed: {str(e)}"}
                    
                    # Configure controllers for extraction
                    try:
                        input_controller = Controller()
                        logger.info("Input controller configured")
                    except Exception as e:
                        logger.error(f"Failed to configure controller: {str(e)}")
                        return {"success": False, "error": f"Controller configuration failed: {str(e)}"}
                    
                    # Setup LLM for the agent
                    try:
                        try:
                            from langchain_openai import ChatOpenAI
                            llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
                            logger.info("Using langchain ChatOpenAI")
                        except ImportError:
                            logger.info("langchain_openai not available, using provided llm_client")
                            llm = llm_client
                            
                        langchain_llm = llm
                    except Exception as e:
                        logger.error(f"Failed to setup LLM: {str(e)}")
                        return {"success": False, "error": f"LLM setup failed: {str(e)}"}
                    
                    # 2. Extract calculator inputs from the page
                    logger.info(f"Extracting calculator inputs...")
                    
                    try:
                        # Input extraction task for agent
                        task = f"""You are examining a calculator webpage ({calculator_name}) and need to extract information about all input fields available.
                        
                            First, navigate to the calculator at: {calculator_url}
                            
                            Your task is to examine the page and identify ALL input fields (both text inputs and option groups like radio buttons).
                            
                            Each input field has a numeric index in square brackets like [42] that identifies it in the DOM. Use these index numbers.
                            
                            Look carefully at each field to understand its type (numeric input or option selection), available options, and any descriptions.
                     
                            
                            DO NOT CLICK ANY BUTTONS - just observe and report.
                            
                            EXAMPLE CORRECT FORMAT for option group
                {{
                              "name": "Sex",
                  "type": "option",
                              "options": ["Male", "Female"],
                  "option_indexes": {{
                                "Male": 32,
                                "Female": 35
                              }}
                }}
                
                EXAMPLE CORRECT FORMAT for text input:
                {{
                  "name": "Age",
                  "type": "numeric",
                              "element_index": 42,
                  "units": ["years"]
                }}
                
                            Your final output must be a properly formatted JSON with this structure:

               {{
                "inputs": [
                                // All input elements with their correct details
                  ]
                }}
                
                Remember: 
                1. All indexes MUST be integers, not strings.
                2. For numeric inputs, use the element_index field.
                3. For options (radio buttons), list all options in the options array and map each option to its index in option_indexes.
                            4. Always look for and include any explanatory text or description beneath each input field name.
                            5. Do not click or type anything - just examine the DOM structure carefully.
                """

                        # First agent: extract calculator inputs with structured output
            agent = Agent(
                task=task,
                            llm=langchain_llm,
                browser_context=context,
                controller=input_controller,
                            use_vision=True,
            )

                        # Run the first agent and get the history
                        logger.info("Running first agent for input extraction...")
                        try:
            history = await agent.run()
                            logger.info("First agent completed extraction")
                        except Exception as e:
                            if "Target crashed" in str(e):
                                logger.error(f"Browser target crashed during extraction: {str(e)}")
                                if retry_count < max_retries:
                                    retry_count += 1
                                    logger.info(f"Retrying from beginning due to browser crash (attempt {retry_count + 1}/{max_retries + 1})...")
                                    # Close the browser if it's still open
                                    try:
                                        if browser:
                                            await browser.close()
                                            logger.info("Closed crashed browser")
                                    except:
                                        pass
                                    await asyncio.sleep(2)  # Wait before retry
                                    continue
                                else:
                                    return {"success": False, "error": "Browser crashed repeatedly during extraction"}
                            else:
                                raise
            
                        # Get the result from first agent
            raw_result = history.final_result()
                        logger.info("Raw Extracted Content received")
            
                        # Process the first agent result
            try:
                # Default empty values in case processing fails
                parsed_inputs = None
                input_summary = {}
                element_id_mapping = {}
                extracted_inputs = "{}"
                            parsed_json = None  # Initialize parsed_json
                
                # Validate with Pydantic
                if raw_result is None:
                                logger.warning("Error: Agent returned None. Using empty defaults.")
                    parsed_inputs = CalculatorInputs(inputs=[])
                    parsed_json = {"inputs": []}  # Initialize with empty structure
                elif isinstance(raw_result, str):
                    parsed_json = json.loads(raw_result)
                else:
                    parsed_json = raw_result
                
                # Ensure all indexes are integers, not strings
                if parsed_json is not None and "inputs" in parsed_json:
                    for input_item in parsed_json["inputs"]:
                        # Convert Unicode in input names to human-readable form
                        if "name" in input_item:
                            # Replace common Unicode symbols with their readable form
                            input_item["name"] = input_item["name"].replace("\u2265", "≥")
                            input_item["name"] = input_item["name"].replace("\u2264", "≤")
                            input_item["name"] = input_item["name"].replace("\u00b0", "°")
                            input_item["name"] = input_item["name"].replace("\u03bc", "µ")
                        
                        if "element_index" in input_item and input_item["element_index"] is not None:
                            if isinstance(input_item["element_index"], str):
                                input_item["element_index"] = int(input_item["element_index"])
                        
                        if "option_indexes" in input_item and input_item["option_indexes"] is not None:
                            for option, index in input_item["option_indexes"].items():
                                if isinstance(index, str):
                                    input_item["option_indexes"][option] = int(index)
                
                    # Convert back to JSON string with proper format
                    extracted_inputs = json.dumps(parsed_json, indent=2, ensure_ascii=False)
                    
                    # Validate with Pydantic
                    parsed_inputs = CalculatorInputs.model_validate_json(extracted_inputs)
                                logger.info("Transformed Inputs validated")
                    
                                # Create the simplified input summary
                    input_summary = create_input_summary(parsed_inputs)
                                logger.info("Input Summary created")
                    
                                # Create the element ID mapping
                    element_id_mapping = create_element_id_mapping(parsed_inputs)
                                logger.info("Element ID Mapping created")
                else:
                                logger.warning("Error: Missing 'inputs' field in the result. Using empty defaults.")
                    parsed_inputs = CalculatorInputs(inputs=[])
                    # Create empty defaults for summary and mapping
                    input_summary = {}
                    element_id_mapping = {}
                
            except Exception as e:
                            logger.error(f"Error processing inputs: {e}")
                # Ensure we have at least empty defaults
                parsed_inputs = CalculatorInputs(inputs=[])
                input_summary = {}
                element_id_mapping = {}
            
                        # 3. Use an LLM to extract inputs from the patient vignette
                        logger.info("Extracting patient data using LLM...")
            extraction_prompt = f"""You are a helpful assistant that is extracting the inputs needed for the {calculator_name}. 

            Here is a JSON list of the inputs needed for the calculator: 

            {input_summary}

            Here is the patient vignette: 

                        {patient_data}
                        
                        IMPORTANT: If any required numerical values are missing from the patient data, your response MUST include them in a special "missing_values" array in your JSON output. Do not make up or assume values that aren't explicitly provided in the patient data.
                        
                        CRITICAL: You MUST perform unit conversions as needed:
                        - If height is provided in cm, convert to inches (1 cm = 0.3937 inches)
                        - If weight is provided in kg, convert to lbs (1 kg = 2.20462 lbs)
                        - Always show your conversion math in the justification
                        
                        Please output a JSON with following format:
            
            {{
                "inputs": [
                    {{ "justification": "justification for the input (cite any evidence from the patient vignette and along with your reasoning)", "input name": "value", }}, // for option-type inputs, select one option from the provided list
                    {{"justification": "justification for the input (cite any evidence from the patient vignette and along with your reasoning)", "input name": [numerical value, "unit"]}},  // for numerical inputs, specify the number and the unit
                    ...
                            ],
                            "missing_values": [
                                "name of required input that is missing from patient data",
                    ...
                ]
            }}

                        Please note:
                        1. If the units don't match one of the options provided, you MUST convert the units.
                        2. When patients have multiple sets of laboratory results (e.g., from different times), always specify which instance (1st, 2nd, 3rd, etc.) you are using.
                        3. If multiple lab values are provided for the same test, choose the most appropriate one and explain your reasoning.
                        4. For EACH MISSING VALUE, include it in the "missing_values" array with its exact input name.
                        """

                        extraction_response = llm_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": extraction_prompt}],
                response_format={"type": "json_object"}
            )

                        llm_response = extraction_response.choices[0].message.content
                    llm_extraction = json.loads(llm_response)
                        logger.info("Patient data extracted using LLM")
                        
                        # Check for missing values
                        missing_values = llm_extraction.get("missing_values", [])
                        if missing_values:
                            # If there are missing values, return early with a helpful message
                            missing_values_str = ", ".join(missing_values)
                            logger.warning(f"Missing required values: {missing_values_str}")
                            return {
                                "success": False,
                                "error": f"Cannot complete calculation - missing required values: {missing_values_str}",
                                "missing_values": missing_values,
                                "extracted_inputs": llm_extraction.get("inputs", [])
                            }
                        
                        # 4. Generate form actions for input
                        logger.info(f"Generating form actions")
            
            # First collect all actions
            click_actions = []
            text_actions = []
                        processed_fields = set()
            
            for item in llm_extraction.get("inputs", []):
                # Get input name and value
                input_name = next((key for key in item.keys() if key != "justification"), None)
                if not input_name or input_name not in element_id_mapping:
                    continue
                
                            # Skip if already processed
                if input_name in processed_fields:
                                logger.info(f"Skipping duplicate field: {input_name}")
                    continue
                
                processed_fields.add(input_name)
                value = item[input_name]
                 
                # Handle different input types based on the value type
                if isinstance(value, list) and len(value) >= 1:  # Numeric inputs with units
                    # Get element index for this numeric input
                    element_index = element_id_mapping[input_name]
                                
                    text_actions.append({
                        "action": "input_text",
                        "element_index": element_index,
                        "text": str(value[0]),
                        "field_name": input_name
                    })
                elif isinstance(value, str):  # Option inputs
                    # Get element index mapping for this option input
                    option_indexes = element_id_mapping[input_name]
                                
                    if isinstance(option_indexes, dict) and value in option_indexes:
                        click_actions.append({
                            "action": "click",
                            "element_index": option_indexes[value],
                                        "value_text": value,
                            "field_name": input_name
                        })
            
                        # Combine actions (clicks first, then text inputs)
            form_actions = click_actions + text_actions
                        logger.info(f"Generated {len(form_actions)} form actions ({len(click_actions)} clicks, {len(text_actions)} text inputs)")
            
                        # 5. Fill form with form-filling agent
                        logger.info(f"Starting form-filling agent")
                        # Create a dynamic task description for the form filling agent
                fill_task_lines = ["Fill out the calculator with the following patient data:"]
                
                        # Add details of what to fill in
                for item in llm_extraction.get("inputs", []):
                    for key, value in item.items():
                        if key != "justification":  # Skip justification fields
                            if isinstance(value, list) and len(value) > 0:
                                fill_task_lines.append(f"- {key}: {value[0]} {value[1] if len(value) > 1 else ''}")
                            else:
                                fill_task_lines.append(f"- {key}: {value}")
                
                        # Update the instructions to explicitly ask to wait for results without trying to save PDF
                        fill_task_lines.append("\nAfter filling in all fields:")
                        fill_task_lines.append("1. Scroll down to see the calculation results")
                        fill_task_lines.append("2. DO NOT click on any buttons to download or save PDFs")
                        
                fill_task = "\n".join(fill_task_lines)
                
                        # Prepare initial actions for the form-filling agent
                initial_actions = []
                        for action in form_actions:
                    if action["action"] == "input_text":
                        initial_actions.append({
                            "input_text": {
                                "index": action["element_index"],
                                "text": action["text"]
                            }
                        })
                    elif action["action"] == "click":
                        initial_actions.append({
                            "click_element_by_index": {
                                "index": action["element_index"]
                            }
                        })
                
                        # Second agent: create a form-filling agent with initial actions
                fill_agent = Agent(
                    task=fill_task,
                            llm=langchain_llm,
                    browser_context=context,
                            controller=Controller(),
                            initial_actions=initial_actions,
                    use_vision=True
                )
                
                        # Run the second agent for form filling
                        logger.info("Running second agent for form filling...")
                        try:
                fill_history = await fill_agent.run()
                            logger.info("Form filling agent completed")
                
                            # Check if the form-filling agent was successful
                            if not fill_history.is_successful():
                                logger.warning("Form-filling agent did not complete successfully")
                        except Exception as e:
                            if "Target crashed" in str(e):
                                logger.error(f"Browser target crashed during form filling: {str(e)}")
                                if retry_count < max_retries:
                                    retry_count += 1
                                    logger.info(f"Retrying from beginning due to browser crash (attempt {retry_count + 1}/{max_retries + 1})...")
                                    # Close the browser if it's still open
                                    try:
                                        if browser:
                                            await browser.close()
                                            logger.info("Closed crashed browser")
                                    except:
                                        pass
                                    await asyncio.sleep(2)  # Wait before retry
                                    continue
                                else:
                                    return {"success": False, "error": "Browser crashed repeatedly during form filling"}
                else:
                                raise
                        
                        # Wait a short time for any UI updates to complete
                        await asyncio.sleep(1)
                        
                        # 6. Extract results with third agent
                        logger.info(f"Starting result extraction agent")
                        
                        try:
                            # Third agent: define task for result extraction
                result_task = f"""
                            Look at the calculator results that are now displayed on the page for the {calculator_name}.
                            Extract only the numerical score/result shown, no interpretation needed.
                
                            Return ONLY a JSON object with this exact format:
                {{
                                "answer": 123.45
                }}
                            
                            Where the value should be just the numerical result WITHOUT any units or text. For example, if you see "3.87 points", just return {{"answer": 3.87}}.
                            Make sure you return ONLY a valid JSON object with a single "answer" key and a numerical value.
                """
                
                # Create a result extraction agent
                result_agent = Agent(
                    task=result_task,
                                llm=langchain_llm,
                    browser_context=context,
                    controller=result_controller,
                    use_vision=True
                )
                
                            # Run the third agent for result extraction
                            logger.info("Running third agent for result extraction...")
                            try:
                result_history = await result_agent.run()
                                logger.info("Result extraction agent completed")
                            except Exception as e:
                                if "Target crashed" in str(e):
                                    logger.error(f"Browser target crashed during result extraction: {str(e)}")
                                    if retry_count < max_retries:
                                        retry_count += 1
                                        logger.info(f"Retrying from beginning due to browser crash (attempt {retry_count + 1}/{max_retries + 1})...")
                                        # Close the browser if it's still open
                                        try:
                                            if browser:
                                                await browser.close()
                                                logger.info("Closed crashed browser")
                                        except:
                                            pass
                                        await asyncio.sleep(2)  # Wait before retry
                                        continue
                                    else:
                                        return {"success": False, "error": "Browser crashed repeatedly during result extraction"}
                                else:
                                    raise
                
                # Get the result
                calculator_result = result_history.final_result()
                
                # Use the extraction functionality from the extractor module
                answer_value = extract_numerical_answer(str(calculator_result))
                
                # Create a result dictionary
                if answer_value is not None:
                    final_result = {
                        "calculator_name": calculator_name,
                        "answer": answer_value
                    }
                    
                    # Display the result prominently
                    logger.info("")
                    logger.info("===========================================")
                    logger.info(f"FINAL ANSWER: {answer_value}")
                    logger.info("===========================================")
                    logger.info("")
                else:
                    # Fallback if no answer was extracted
                    if calculator_result is None:
                        logger.warning("Result extraction agent returned None")
                        final_result = {
                            "calculator_name": calculator_name,
                            "error": "No result extracted"
                        }
                    else:
                        # Use the original result
                        final_result = calculator_result if isinstance(calculator_result, dict) else {
                            "calculator_name": calculator_name,
                            "original_result": str(calculator_result)
                        }
                
                # Create the result dictionary
                result_dict = {
                    "success": True,
                    "result": final_result,
                    "extracted_inputs": llm_extraction
                }
                
                # Save to JSON
                save_results_to_json(result_dict, calculator_name)
                
                logger.info(f"Calculation complete, returning results")
                
                # Add a small delay to ensure the WebSocket has time to send the result before any potential reload
                await asyncio.sleep(1.5)
                
                return result_dict
                            
                        except Exception as result_error:
                            logger.error(f"Error in result extraction: {str(result_error)}")
                            # Even if result extraction fails, return what we have
                            return {
                                "success": True,
                                "result": {
                                    "calculator_name": calculator_name,
                                    "score": "Calculation complete"
                                },
                                "extracted_inputs": llm_extraction,
                                "error_details": str(result_error)
                            }
                            
                    except Exception as extraction_error:
                        logger.error(f"Error in extraction process: {str(extraction_error)}")
                        if retry_count < max_retries:
                            retry_count += 1
                            logger.info(f"Retrying extraction (attempt {retry_count + 1}/{max_retries + 1})...")
                            continue
                        return {"success": False, "error": f"Extraction error: {str(extraction_error)}"}
                    
            except Exception as context_error:
                logger.error(f"Error in browser context: {str(context_error)}")
                if retry_count < max_retries:
                    retry_count += 1
                    logger.info(f"Retrying due to context error (attempt {retry_count + 1}/{max_retries + 1})...")
                    await asyncio.sleep(1)  # Wait a bit before retrying
                    continue
                return {"success": False, "error": f"Browser context error: {str(context_error)}"}
            
        except Exception as e:
            logger.error(f"Workflow agent error: {str(e)}")
            traceback.print_exc()
            if retry_count < max_retries:
                retry_count += 1
                logger.info(f"Retrying due to general error (attempt {retry_count + 1}/{max_retries + 1})...")
                await asyncio.sleep(1)  # Wait a bit before retrying
                continue
            return {
                "success": False,
                "error": str(e)
            }
        finally:
            # Ensure browser is closed properly
            if browser:
                try:
            await browser.close()
                    logger.info("Browser closed successfully")
                except Exception as close_error:
                    logger.error(f"Error closing browser: {close_error}")
            
            # If we got here without exceptions or retries, break the loop
            break
    
    # If we exhausted all retries and still failed, return a failure
    if retry_count > max_retries:
        return {
            "success": False,
            "error": "Failed after maximum retry attempts"
        }

async def main():
    """
    Main function to run the workflow agent as a standalone script
    """
    try:
        # Create OpenAI client for testing
        try:
            from openai import OpenAI
            client = OpenAI()
            logger.info("Using OpenAI client")
        except ImportError:
            try:
                from openai import AzureOpenAI
                client = AzureOpenAI(
                    api_version='2024-02-15-preview',
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
                    api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
                )
                logger.info("Using AzureOpenAI client")
            except ImportError:
                logger.error("No OpenAI client available")
                return

        # Example usage
        calculator_name = "FIB-4 Index for Liver Fibrosis"
        calculator_url = "https://www.mdcalc.com/calc/2200/fibrosis-4-fib-4-index-liver-fibrosis"
        patient_data = """
        A 58-year-old male with chronic hepatitis C presents for evaluation. 
        Laboratory results show:
        - AST: 68 U/L
        - ALT: 72 U/L
        - Platelets: 120 × 10^9/L
        - Age: 58 years
        """

        # Run the workflow agent
        result = await run_workflow_calculator(
            calculator_name=calculator_name,
            calculator_url=calculator_url,
            patient_data=patient_data,
            llm_client=client
        )
        
        # Save results to JSON file
        try:
            # Create a results directory if it doesn't exist
            os.makedirs('results', exist_ok=True)
            
            # Generate a filename based on calculator name and timestamp
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results/{calculator_name.replace(' ', '_')}_{timestamp}.json"
            
            # Write the results to the file
            with open(filename, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Results saved to {filename}")
        except Exception as save_error:
            logger.error(f"Failed to save results to JSON: {str(save_error)}")

        # Print the result
        logger.info("Workflow Agent Result:")
        logger.info(json.dumps(result, indent=2, ensure_ascii=False))
        
        return result
    
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        traceback.print_exc()
        return {"error": str(e)}

if __name__ == "__main__":
    asyncio.run(main())


