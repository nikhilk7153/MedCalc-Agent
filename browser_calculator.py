from browser_use import Agent, Controller
from browser_use.browser.browser import Browser, BrowserConfig, BrowserContextConfig
from typing import Dict, List, Any, Optional, Union
import json
import logging
import asyncio
import traceback
import os
import sys
from datetime import datetime

try:
    from pydantic import BaseModel, Field
except ImportError:
    # Fallback for environments without pydantic
    print("Warning: Pydantic not available, using fallback class")
    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    Field = lambda *args, **kwargs: None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("browser_calculator")

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

# Create controller for structured output
input_controller = Controller(output_model=CalculatorInputs)

# Function to create a structured representation of calculator inputs
def create_input_summary(calculator_inputs: CalculatorInputs) -> Dict[str, Dict[str, Any]]:
    """
    Converts the CalculatorInputs object into a simplified dictionary with input names as keys
    and their options, types, units, and descriptions as values.
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

async def run_browser_calculator(calculator_name, calculator_url, patient_data, llm_client, *, keep_open: bool = False, existing_context=None):
    """
    Run a browser-based calculator to extract form field attributes using the agent approach
    
    Args:
        calculator_name: Name of the calculator
        calculator_url: URL of the calculator
        patient_data: Patient case data
        llm_client: OpenAI client for agent
        keep_open: Whether to keep the browser open after the calculation
        existing_context: Existing browser context to reuse
        
    Returns:
        Dictionary with the form field attributes
    """
    browser = None
    try:
        # Log that we're starting
        logger.info(f"===== STARTING browser calculator for {calculator_name} =====")
        logger.info(f"Patient data: {patient_data}")
        logger.info(f"Calculator URL: {calculator_url}")
        
        # Launch the browser with the correct configuration
        logger.info("Launching browser...")
        browser = Browser(
            config=BrowserConfig(
                headless=False,  # Use headed mode for reliability with MDCalc
                disable_security=True,
                extra_browser_args=[
                    "--disable-window-activation",  # Prevent browser from taking focus
                    "--disable-focus-on-load"       # Prevent browser from stealing focus on load
                ]
            )
        )
        
        # Create a browser context
        logger.info("Creating browser context...")
        async with await browser.new_context() as context:
            
            # Get the current page
            page = await context.get_current_page()
            
            # Navigate to the calculator URL
            logger.info(f"Navigating to {calculator_url}")
            await page.goto(calculator_url)
            
            # Wait for page to load with retry mechanism
            logger.info("Waiting for page to load...")
            retry_count = 0
            max_retries = 3
            load_success = False
            
            while retry_count < max_retries and not load_success:
                try:
                    # Use a shorter timeout for networkidle to avoid long hangs
                    await page.wait_for_load_state("domcontentloaded", timeout=30000)
                    
                    # Instead of waiting for networkidle (which can timeout), wait for specific elements to be visible
                    logger.info("Page DOM loaded, waiting for page to stabilize...")
                    await asyncio.sleep(5)  # Give additional time for JavaScript to execute
                    
                    # Check if the page has loaded by verifying a common element is present
                    content_visible = await page.evaluate("() => document.body !== null && document.body.children.length > 0")
                    
                    if content_visible:
                        logger.info("Page content is visible, proceeding...")
                        load_success = True
                    else:
                        logger.warning(f"Page content not visible yet, retry {retry_count + 1}/{max_retries}")
                        retry_count += 1
                        await asyncio.sleep(3)
                except Exception as load_error:
                    logger.warning(f"Error waiting for page to load (attempt {retry_count + 1}/{max_retries}): {str(load_error)}")
                    retry_count += 1
                    await asyncio.sleep(3)
            
            if not load_success:
                logger.warning("Page load state wait unsuccessful after retries, proceeding anyway...")
            
            # Additional waiting to ensure page is stable
            logger.info("Waiting for page to stabilize...")
            await asyncio.sleep(5)
            
            # Extract calculator inputs using an agent
            logger.info("Extracting form fields and attributes...")
            
            # Define a task for the agent to extract input fields
            task = f"""You are examining a calculator webpage ({calculator_name}) and need to extract information about all input fields available.
            
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
            
            # Create the agent to extract input fields
            controller = input_controller
            
            # Convert the OpenAI client to a LangChain compatible format
            try:
                # First try to import LangChain's ChatOpenAI
                from langchain_openai import ChatOpenAI
                # Create a LangChain wrapper around the client
                langchain_llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
                logger.info("Created LangChain ChatOpenAI wrapper")
            except ImportError:
                # If LangChain is not available, create a simple adapter class
                logger.info("LangChain not available, using client directly")
                # This is a simple adapter to mimic LangChain's interface
                class LLMAdapter:
                    def __init__(self, client):
                        self.client = client
                    
                    async def ainvoke(self, prompt, **kwargs):
                        response = self.client.chat.completions.create(
                            model="gpt-4o",
                            messages=[{"role": "user", "content": prompt}]
                        )
                        return response.choices[0].message.content
                
                langchain_llm = LLMAdapter(llm_client)
            
            agent = Agent(
                task=task,
                llm=langchain_llm,
                browser_context=context,
                controller=controller,
                use_vision=True,
            )
            
            # Run the agent
            logger.info("Running agent to extract form fields...")
            try:
                history = await agent.run()
                logger.info("Agent completed extraction")
                
                # Get the raw result
                raw_result = history.final_result()
                logger.info("Raw Extracted Content received")
                
                # Process the result
                try:
                    # Validate and parse the result
                    if raw_result is None:
                        logger.warning("Error: Agent returned None")
                        parsed_inputs = CalculatorInputs(inputs=[])
                    elif isinstance(raw_result, str):
                        parsed_json = json.loads(raw_result)
                        parsed_inputs = CalculatorInputs.model_validate(parsed_json)
                    else:
                        parsed_inputs = raw_result
                    
                    # Process the inputs
                    for input_item in parsed_inputs.inputs:
                        # Convert Unicode in input names to human-readable form
                        input_item.name = input_item.name.replace("\u2265", "≥")
                        input_item.name = input_item.name.replace("\u2264", "≤")
                        input_item.name = input_item.name.replace("\u00b0", "°")
                        input_item.name = input_item.name.replace("\u03bc", "µ")
                        
                        # Ensure numeric indexes are integers
                        if input_item.element_index is not None and isinstance(input_item.element_index, str):
                            input_item.element_index = int(input_item.element_index)
                        
                        # Ensure option indexes are integers
                        if input_item.option_indexes:
                            for option, index in input_item.option_indexes.items():
                                if isinstance(index, str):
                                    input_item.option_indexes[option] = int(index)
                    
                    # Create the summary dictionary
                    input_summary = create_input_summary(parsed_inputs)
                    logger.info("Created input summary")
                    
                    # Create the element mapping
                    element_id_mapping = create_element_id_mapping(parsed_inputs)
                    logger.info("Created element ID mapping")
                
                except Exception as e:
                    logger.error(f"Error processing inputs: {e}")
                    traceback.print_exc()
                    parsed_inputs = CalculatorInputs(inputs=[])
                    input_summary = {}
                    element_id_mapping = {}
                
                # Print out the extracted elements with proper formatting
                print("\n======= FORM FIELD ATTRIBUTES =======\n")
                print(json.dumps(parsed_inputs.dict() if hasattr(parsed_inputs, 'dict') else vars(parsed_inputs), indent=2))
                print("\n====================================\n")
                
                # Print the simplified summary
                print("\n======= SIMPLIFIED FIELD SUMMARY =======\n")
                for i, field in enumerate(parsed_inputs.inputs):
                    print(f"{i+1}. {field.name} (Type: {field.type}, Element ID: {field.element_index})")
                    if field.options:
                        print(f"   Options: {', '.join(field.options)}")
                    if field.units:
                        print(f"   Units: {', '.join(field.units)}")
                    print()
                print("\n======================================\n")
                
                # Save the extracted form information to a JSON file
                os.makedirs('results', exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"results/{calculator_name.replace(' ', '_')}_{timestamp}.json"
                
                # Create the result dictionary
                result = {
                    "success": True,
                    "calculator_name": calculator_name,
                    "form_fields": parsed_inputs.dict() if hasattr(parsed_inputs, 'dict') else vars(parsed_inputs),
                    "input_summary": input_summary,
                    "element_id_mapping": element_id_mapping
                }
                
                # Save to file
                with open(filename, 'w') as f:
                    json.dump(result, f, indent=2, default=lambda o: o.dict() if hasattr(o, 'dict') else vars(o))
                logger.info(f"Results saved to {filename}")
                
                # Now extract inputs from the patient data using the input summary as a guide
                logger.info("Extracting inputs from patient data...")
                
                extraction_prompt = f"""You are a helpful assistant that is extracting the inputs needed for the {calculator_name}. 

Here is a JSON list of the inputs needed for the calculator: 

{input_summary}

USER'S QUESTION:
{patient_data.split("PATIENT DATA:", 1)[0].strip() if "PATIENT DATA:" in patient_data else ""}

Here is the patient vignette: 

{patient_data.split("PATIENT DATA:", 1)[1].strip() if "PATIENT DATA:" in patient_data else patient_data}
                
IMPORTANT INSTRUCTIONS:
1. Review the calculator's required inputs carefully.
2. Focus directly on the USER'S QUESTION above - it specifies exactly what needs to be calculated.
3. Match each calculator input field with the corresponding data from the patient's clinical information.
4. Be precise in your data extraction, as medical calculations require exact values.
5. Please be sure to use the values and patient's health status at the time of admission prior to treatment. If a patient vignette does not mention anything about a patient's health status, you can assume it is absent.
                
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

Additional instructions:
1. If the units don't match one of the options provided, you MUST convert the units.
2. When patients have multiple sets of laboratory results (e.g., from different times), always specify which instance (1st, 2nd, 3rd, etc.) you are using.
3. Use the patient's health status at the time of admission prior to treatment. If a patient vignette does not mention anything about a patient's health status, you can assume it is absent/false, but you must select one of the options available.
4. If calculated values (like BMI) are requested and not directly provided in the notes, perform the calculation yourself using available data.
5. For any ambiguous or unclear data points, choose the most clinically appropriate option based on the overall patient presentation.
"""

                try:
                    # Call the LLM to extract patient data
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
                            "extracted_inputs": llm_extraction.get("inputs", []),
                            "form_fields": parsed_inputs.dict() if hasattr(parsed_inputs, 'dict') else vars(parsed_inputs),
                        }
                    
                    # Generate form actions
                    logger.info(f"Generating form actions")
                    
                    # First collect all actions
                    click_actions = []
                    text_actions = []
                    processed_fields = set()
                    
                    for item in llm_extraction.get("inputs", []):
                        # Get input name and value
                        input_name = next((key for key in item.keys() if key != "justification"), None)
                        if not input_name or input_name not in element_id_mapping:
                            logger.warning(f"Skipping field not found in element mapping: {input_name}")
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
                    
                    # Prepare initial actions for form filling
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
                    
                    # Update result with patient data extraction
                    result.update({
                        "extracted_inputs": llm_extraction.get("inputs", []),
                        "form_actions": form_actions
                    })
                    
                    # Save the updated result
                    with open(filename, 'w') as f:
                        json.dump(result, f, indent=2, default=lambda o: o.dict() if hasattr(o, 'dict') else vars(o))
                    logger.info(f"Updated results saved with extracted inputs to {filename}")
                    
                    # Now create and run form-filling agent to complete the calculation
                    logger.info("Starting form-filling agent to complete calculation")
                    
                    # Create the fill form task
                    fill_task_lines = ["Fill out the calculator with the following patient data:"]
                    
                    # Add details of what to fill in
                    for item in llm_extraction.get("inputs", []):
                        for key, value in item.items():
                            if key != "justification":  # Skip justification fields
                                if isinstance(value, list) and len(value) > 0:
                                    fill_task_lines.append(f"- {key}: {value[0]} {value[1] if len(value) > 1 else ''}")
                                else:
                                    fill_task_lines.append(f"- {key}: {value}")
                    
                    # Update the instructions to explicitly ask to wait for results
                    fill_task_lines.append("\nAfter filling in all fields:")
                    fill_task_lines.append("1. WAIT at least 3 seconds – most MDCalc tools auto-update once inputs are entered (no 'Calculate' button needed).")
                    fill_task_lines.append("2. Do NOT click anything else. Simply allow the page to refresh the result display.")
                    fill_task_lines.append("3. As soon as the numerical result appears, the task is complete – no scrolling or extra steps required.")
                    fill_task_lines.append("4. DO NOT click on any buttons to download or save PDFs.")
                    
                    fill_task = "\n".join(fill_task_lines)
                    
                    # Create a form-filling agent with initial actions
                    fill_agent = Agent(
                        task=fill_task,
                        llm=langchain_llm,
                        browser_context=context,
                        controller=Controller(),
                        initial_actions=initial_actions,
                        use_vision=True
                    )
                    
                    # Run the form filling agent with error handling
                    logger.info("Running form filling agent with error resilience...")
                    try:
                        try:
                            fill_history = await fill_agent.run(max_steps=5)  # Limit to max 5 steps
                            logger.info("Form filling agent completed successfully")
                        except Exception as inner_error:
                            logger.error(f"Error during form action: {str(inner_error)}")
                            logger.info("Will attempt to continue despite form filling errors")
                        
                        # Wait longer for UI to stabilize
                        logger.info("Waiting for page to stabilize...")
                        await asyncio.sleep(5)
                        
                        # Scroll to the very top to ensure the screenshot starts from the beginning of the page
                        try:
                            logger.info("Scrolling to the top of the page before screenshot...")
                            await page.evaluate("window.scrollTo(0, 0);")
                            # Give the browser a moment to settle after scrolling
                            await asyncio.sleep(1)
                        except Exception as scroll_err:
                            logger.warning(f"Could not scroll to top before screenshot: {scroll_err}")
                        
                        # Take a screenshot of the result
                        # Create static directory for screenshots if it doesn't exist
                        static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "screenshots")
                        os.makedirs(static_dir, exist_ok=True)
                        
                        # Create screenshot filename
                        screenshot_filename = f"{calculator_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                        screenshot_path = os.path.join(static_dir, screenshot_filename)
                        
                        # Take the screenshot of the full page
                        logger.info("Taking screenshot of calculation result...")
                        await page.screenshot(path=screenshot_path, full_page=True)
                        logger.info(f"Screenshot saved to {screenshot_path}")
                        
                        # Create a relative path that can be used in the frontend
                        relative_screenshot_path = f"static/screenshots/{screenshot_filename}"
                        
                        # Display the screenshot directly in the chat interface
                        print(f"\n\n# {calculator_name} Result\n")
                        print(f"![{calculator_name} Result]({relative_screenshot_path})\n")
                        
                        # Extract the result with a third agent
                        logger.info("Extracting calculator result...")
                        
                        # Create a result extraction agent
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
                            controller=Controller(),
                            use_vision=True
                        )
                        
                        # Run the result extraction agent with error handling
                        logger.info("Running result extraction agent...")
                        try:
                            result_history = await result_agent.run(max_steps=5)  # Limit to max 5 steps
                            logger.info("Result extraction agent completed")
                        except Exception as extract_error:
                            logger.error(f"Error during result extraction: {str(extract_error)}")
                            logger.info("Will attempt to parse results despite extraction error")
                            result_history = None
                        
                        # Extract the numerical result
                        calculator_result = result_history.final_result() if result_history else None
                        logger.info(f"Raw calculator result: {calculator_result}")
                        
                        # Process the calculator result to extract the numerical answer
                        numerical_answer = None
                        try:
                            if isinstance(calculator_result, dict) and "answer" in calculator_result:
                                numerical_answer = float(calculator_result["answer"])
                            elif isinstance(calculator_result, str):
                                # Try to parse as JSON
                                try:
                                    result_json = json.loads(calculator_result)
                                    if "answer" in result_json:
                                        numerical_answer = float(result_json["answer"])
                                except:
                                    # Try to extract number with regex
                                    import re
                                    number_match = re.search(r'(\d+\.\d+)', calculator_result)
                                    if number_match:
                                        numerical_answer = float(number_match.group(1))
                        except Exception as e:
                            logger.error(f"Error extracting numerical answer: {str(e)}")
                        
                        # Display the numerical answer prominently
                        if numerical_answer is not None:
                            logger.info("")
                            logger.info("===========================================")
                            logger.info(f"FINAL ANSWER: {numerical_answer}")
                            logger.info("===========================================")
                            logger.info("")
                        
                        if numerical_answer is not None:
                            print(f"**Final Score: {numerical_answer}**\n")
                        
                        # Final result with screenshot path and numerical answer
                        final_result = {
                            "success": True,
                            "calculator_name": calculator_name,
                            "form_fields": parsed_inputs.dict() if hasattr(parsed_inputs, 'dict') else vars(parsed_inputs),
                            "extracted_inputs": llm_extraction.get("inputs", []),
                            "screenshot_path": screenshot_path,
                            "relative_screenshot_path": relative_screenshot_path,
                            "answer": numerical_answer,
                            "message": "Calculation complete",
                            "image_markdown": f"![Calculator Result]({relative_screenshot_path})"
                        }
                        
                        # Save the final result
                        with open(filename, 'w') as f:
                            json.dump(final_result, f, indent=2, default=lambda o: o.dict() if hasattr(o, 'dict') else vars(o))
                        logger.info(f"Final result saved to {filename}")
                        
                        return final_result
                        
                    except Exception as extraction_error:
                        logger.error(f"Error extracting patient data: {str(extraction_error)}")
                        traceback.print_exc()
                        return {
                            "success": False,
                            "error": f"Error extracting patient data: {str(extraction_error)}",
                            "form_fields": parsed_inputs.dict() if hasattr(parsed_inputs, 'dict') else vars(parsed_inputs)
                        }
                
                except Exception as extraction_error:
                    logger.error(f"Error extracting patient data: {str(extraction_error)}")
                    traceback.print_exc()
                    return {
                        "success": False,
                        "error": f"Error extracting patient data: {str(extraction_error)}",
                        "form_fields": parsed_inputs.dict() if hasattr(parsed_inputs, 'dict') else vars(parsed_inputs)
                    }
                
                return result
                
            except Exception as agent_error:
                logger.error(f"Error with extraction agent: {str(agent_error)}")
                traceback.print_exc()
                return {
                    "success": False,
                    "error": f"Agent extraction error: {str(agent_error)}"
                }
            
    except Exception as e:
        logger.error(f"Error running browser calculator: {str(e)}")
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }
    finally:
        # Make sure we close the browser unless caller wants to keep it open
        if browser and not keep_open:
            try:
                await browser.close()
                logger.info("Browser closed")
            except Exception as e:
                logger.error(f"Error closing browser: {str(e)}") 