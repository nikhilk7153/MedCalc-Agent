#!/usr/bin/env python3
"""
Medical Calculator Workflow Functions
------------------------------------
This module provides direct access to workflow_agent functionality with clean interfaces.
Unlike workflow_adapter.py, this file doesn't create wrappers or mock data - it directly
imports and exposes the functionality from workflow_agent.py in modular form.
"""

import json
import asyncio
import os
import traceback
import logging
import sys
from typing import Dict, Any, Tuple, List, Optional, Union

# Configure logging first so it's available throughout the module
logger = logging.getLogger("workflow_functions")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

# Add debug logging at the start
logger.info("Loading workflow_functions module")

try:
    # Import necessary components from workflow_agent.py with error handling
    logger.info("Importing from workflow_agent.py")
    from workflow_agent import (
        AzureChatOpenAI, 
        AzureOpenAI,
        Controller, 
        Browser, 
        BrowserConfig,
        Agent,
        CalculatorInputs,
        create_input_summary,
        create_element_id_mapping
    )
    logger.info("Successfully imported from workflow_agent")
except ImportError as e:
    logger.error(f"Error importing from workflow_agent: {str(e)}")
    raise

try:
    # Import needed models or define new ones if necessary
    logger.info("Importing from pydantic")
    from pydantic import BaseModel
    logger.info("Successfully imported from pydantic")
except ImportError as e:
    logger.error(f"Error importing pydantic: {str(e)}")
    raise

try:
    logger.info("Importing from browser_use.controller.views")
    from browser_use.controller.views import (
        InputTextAction,
        ClickElementAction,
        ScrollAction
    )
    logger.info("Successfully imported from browser_use.controller.views")
except ImportError as e:
    logger.error(f"Error importing from browser_use.controller.views: {str(e)}")
    raise

# Store last error for debugging
last_error = None

class ExtractedValueReasoning(BaseModel):
    """Class for representing an extracted value with reasoning."""
    input_name: str
    value: Union[str, float, List[Union[str, float]]]
    unit: Optional[str] = None
    reasoning: str  # Detailed reasoning for the extraction
    evidence: str   # Evidence from the patient note
    confidence: str = "medium"  # Confidence level (low, medium, high)

class ExtractedValuesWithReasoning(BaseModel):
    """Class for representing all extracted values with reasoning."""
    input_values: List[ExtractedValueReasoning]

class CalculatorResult(BaseModel):
    """Class for representing calculator results."""
    score: str
    interpretation: Optional[str] = None
    additional_info: Optional[Dict[str, Any]] = None

def get_llm():
    """Returns an instance of the Azure Chat OpenAI model."""
    try:
        logger.info("Creating Azure Chat OpenAI instance")
        # Setting these environment variables explicitly before creating the LLM
        os.environ["AZURE_OPENAI_API_KEY"] = "9a8467bfe81d443d97b1af452662c33c"
        os.environ["AZURE_OPENAI_ENDPOINT"] = "https://azure-openai-miblab-ncu.openai.azure.com/"

        # Create the LLM
        return AzureChatOpenAI(
            model="gpt-4o",
            api_version='2024-02-15-preview',
            azure_endpoint="https://azure-openai-miblab-ncu.openai.azure.com/",
            api_key="9a8467bfe81d443d97b1af452662c33c",
        )
    except Exception as e:
        global last_error
        last_error = e
        logger.error(f"Error creating LLM: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def get_client():
    """Returns an instance of the Azure OpenAI client."""
    try:
        logger.info("Creating Azure OpenAI client")
        return AzureOpenAI(
            api_version='2024-02-15-preview',
            azure_endpoint="https://azure-openai-miblab-ncu.openai.azure.com/",
            api_key="9a8467bfe81d443d97b1af452662c33c",
        )
    except Exception as e:
        global last_error
        last_error = e
        logger.error(f"Error creating client: {str(e)}")
        logger.error(traceback.format_exc())
        raise

async def extract_calculator_inputs(calculator_name: str, url: str) -> Tuple[CalculatorInputs, Dict[str, Dict[str, Any]], Dict[str, Union[int, Dict[str, int]]]]:
    """
    Extracts calculator inputs from a specified URL.
    
    Args:
        calculator_name: Name of the calculator
        url: URL of the calculator
        
    Returns:
        A tuple containing:
        - parsed_inputs: A CalculatorInputs object with the parsed inputs
        - input_summary: A dictionary mapping input names to their properties
        - element_id_mapping: A dictionary mapping input names to their element IDs
    """
    global last_error
    browser = None
    
    try:
        # Get the LLM
        logger.info(f"Starting extract_calculator_inputs for {calculator_name} at {url}")
        llm = get_llm()
        
        # Create controller with output model for structured output
        logger.info("Creating controller with output model")
        input_controller = Controller(output_model=CalculatorInputs)
        
        # Create browser with timeout
        logger.info("Creating browser instance")
        browser = Browser(
            config=BrowserConfig(
                headless=False,
                disable_security=True,
            )
        )
        
        logger.info("Creating new browser context")
        context = await asyncio.wait_for(browser.new_context(), timeout=30.0)  # 30 second timeout
        
        try:
            async with context:
                # Define the task for the agent
                logger.info("Defining task for browser agent")
                task = f"""You are an agent whose job is to extract the relevant calculator inputs for the {calculator_name} from MDCalc.com.

                    Here is the url of the medical calculator:
                    {url}
                    
                    CRITICAL INSTRUCTIONS:
                    1. Visit the URL and analyze all input fields on the calculator.
                    2. Look at the page to see the numeric index values associated with each element.
                    
                    When browser-use highlights elements on the page, each interactive element gets a numeric 
                    index like [1], [2], [3], etc. These index values are crucial - they are how we'll interact 
                    with the elements later.
                    
                    For each input field, extract:
                    - The exact input name as shown on the website
                    - The type of input (numeric for text inputs, option for radio buttons/dropdowns)
                    - For text inputs: The numeric element index shown on the page (e.g., [42])
                    - For radio buttons or options: 
                      * List all the option values as a simple array (e.g., ["No", "Yes"])
                      * Map each option to its numeric element index (e.g., {{"No": 15, "Yes": 16}})
                    - IMPORTANT: For any input name that has explanatory text or a description underneath it, also extract this 
                      description text in full. This is critical information about how to interpret the input. This information exists directly beneath the input name, if it exists. If there is no description, you should not include a description field in the JSON for that input. 
                    
                    DO NOT include any scoring values associated with options. The options should be ONLY the visible text choices. You should not be clicking or typing anything for this task. You should simply be examining the page and extracting the information. For items which are optional, they should be included in the JSON.
                    
                    EXAMPLE CORRECT FORMAT for option group (like radio buttons) WITH A DESCRIPTION:
                    {{
                      "name": "Mobility",
                      "type": "option",
                      "options": ["Normal, out of bed", "Bed rest <72 hours", "Bed rest >72 hours"],
                      "option_indexes": {{
                        "Normal, out of bed": 15,
                        "Bed rest <72 hours": 19,
                        "Bed rest >72 hours": 21
                      }},
                      "description": "Bed rest is defined as not being able to walk 30 feet (10 meters) at one time. Bathroom privileges or walking in the room are not considered ambulation. Walking this distance reduces the VTE risk by 50%. Click here for VIDEO. PE mortality increased for those immobile for >4 days."
                    }}
                    
                    EXAMPLE CORRECT FORMAT for text input:
                    {{
                      "name": "Age",
                      "type": "numeric",
                      "element_index": 42,  // The numeric index shown on the page [42]
                      "units": ["years"]
                    }}
                    
                    Your final output must be a properly formatted JSON following this exact structure:

                   {{
                    "inputs": [
                        // all input elements with their correct numeric indexes and descriptions
                      ]
                    }}
                    
                    Remember: 
                    1. All indexes MUST be integers, not strings.
                    2. For numeric inputs, use the element_index field.
                    3. For options (radio buttons), list all options in the options array and map each option to its index in option_indexes.
                    4. Always look for and include any explanatory text or description beneath each input field name (not the text inputs or radio buttons).
                    5. You should not be clicking or typing anything for this task. You should simply be examining the page and extracting the information.
                    """

                # Create agent with structured output using Pydantic model
                logger.info("Creating agent instance")
                agent = Agent(
                    task=task,
                    llm=llm,
                    browser_context=context,
                    controller=input_controller,
                    use_vision=True
                )

                # Run the agent with timeout
                logger.info("Running the browser agent (with 5 minute timeout)")
                history = await asyncio.wait_for(agent.run(), timeout=300.0)  # 5 minute timeout
                
                # Get the result
                logger.info("Getting agent result")
                raw_result = history.final_result()
                
                # Process the result
                try:
                    # Default empty values in case processing fails
                    parsed_inputs = None
                    input_summary = {}
                    element_id_mapping = {}
                    extracted_inputs = "{}"
                    parsed_json = None
                    
                    # Validate with Pydantic
                    if raw_result is None:
                        logger.warning("Agent returned None result")
                        parsed_inputs = CalculatorInputs(inputs=[])
                        parsed_json = {"inputs": []}
                    elif isinstance(raw_result, str):
                        logger.info("Parsing string result as JSON")
                        parsed_json = json.loads(raw_result)
                    else:
                        logger.info("Using raw result object")
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
                        logger.info("Validating with Pydantic")
                        parsed_inputs = CalculatorInputs.model_validate_json(extracted_inputs)
                        
                        # Create the simplified input summary
                        logger.info("Creating input summary")
                        input_summary = create_input_summary(parsed_inputs)
                        
                        # Create the element ID mapping
                        logger.info("Creating element ID mapping")
                        element_id_mapping = create_element_id_mapping(parsed_inputs)
                    else:
                        logger.warning("Invalid or empty JSON result, using defaults")
                        parsed_inputs = CalculatorInputs(inputs=[])
                        input_summary = {}
                        element_id_mapping = {}
                    
                    logger.info(f"Successfully extracted {len(parsed_inputs.inputs) if parsed_inputs else 0} inputs from calculator")
                    return parsed_inputs, input_summary, element_id_mapping
                    
                except Exception as e:
                    last_error = e
                    logger.error(f"Error processing inputs: {e}")
                    logger.error(traceback.format_exc())
                    # Ensure we have at least empty defaults
                    parsed_inputs = CalculatorInputs(inputs=[])
                    input_summary = {}
                    element_id_mapping = {}
                    return parsed_inputs, input_summary, element_id_mapping
                
        except asyncio.TimeoutError as e:
            last_error = e
            logger.error("Browser task timed out")
            # Ensure we have at least empty defaults
            parsed_inputs = CalculatorInputs(inputs=[])
            input_summary = {}
            element_id_mapping = {}
            return parsed_inputs, input_summary, element_id_mapping
    
    except Exception as e:
        last_error = e
        logger.error(f"Error in extract_calculator_inputs: {str(e)}")
        logger.error(traceback.format_exc())
        # Ensure we have at least empty defaults
        parsed_inputs = CalculatorInputs(inputs=[])
        input_summary = {}
        element_id_mapping = {}
        return parsed_inputs, input_summary, element_id_mapping
    
    finally:
        # Always ensure browser is closed
        if browser:
            try:
                logger.info("Closing browser")
                await asyncio.wait_for(browser.close(), timeout=10.0)
            except Exception as e:
                logger.error(f"Error closing browser: {str(e)}")

async def extract_values_with_reasoning(calculator_name: str, input_summary: Dict[str, Dict[str, Any]], patient_note: str) -> Tuple[ExtractedValuesWithReasoning, Dict[str, Any]]:
    """
    Extracts values from a patient note with reasoning.
    
    Args:
        calculator_name: Name of the calculator
        input_summary: A dictionary mapping input names to their properties
        patient_note: The patient note text
        
    Returns:
        A tuple containing:
        - reasoning_result: An ExtractedValuesWithReasoning object with extracted values and reasoning
        - final_values: A dictionary with calculator-friendly values for each input
    """
    # Get the OpenAI client
    client = get_client()
    
    # Construct the extraction prompt
    extraction_prompt = f"""You are a helpful assistant that is extracting the inputs needed for the {calculator_name}. 

    Here is a JSON list of the inputs needed for the calculator: 

    {input_summary}

    Here is the patient note: 

    {patient_note}
    
    Please output a JSON with following format, you should not output any other text: 
    
    {{
        "inputs": [
            {{ 
                "input_name": "Input Name",
                "value": "value",  // String value for options
                "confidence": "high/medium/low",
                "evidence": "direct quote from the patient note that supports this value",
                "reasoning": "detailed explanation of why this value was selected based on the evidence"
            }},
            {{ 
                "input_name": "Another Input Name",
                "value": 123,  // Numeric value
                "unit": "unit",  // Include unit if applicable
                "confidence": "high/medium/low",
                "evidence": "direct quote from the patient note that supports this value",
                "reasoning": "detailed explanation of why this value was selected based on the evidence"
            }},
            ...
        ]
    }}

    For each input:
    1. Find the most appropriate value from the patient note
    2. Include the specific evidence (quote) from the note that supports your answer
    3. Provide reasoning that explains your decision process
    4. Indicate your confidence level (high/medium/low)
    5. For numerical values, include the appropriate unit
    6. For option-type inputs, select one option from the provided list
    7. If a value cannot be determined from the note, make your best guess and mark confidence as "low"

    Please note that if the units in the patient note don't match the expected units, convert them appropriately.
    """

    # Get the response from the model
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": extraction_prompt}],
        response_format={"type": "json_object"}
    )

    llm_response = response.choices[0].message.content
    
    # Parse the response
    try:
        # Process the result
        extraction_data = json.loads(llm_response)
        
        # Convert to our model format
        input_values = []
        final_values = {
            "calculator_name": calculator_name,
            "inputs": {}
        }
        
        for item in extraction_data.get("inputs", []):
            input_name = item.get("input_name", "")
            value = item.get("value", "")
            unit = item.get("unit", None)
            confidence = item.get("confidence", "medium")
            evidence = item.get("evidence", "")
            reasoning = item.get("reasoning", "")
            
            # Add to our reasoning model
            input_values.append(
                ExtractedValueReasoning(
                    input_name=input_name,
                    value=value,
                    unit=unit,
                    reasoning=reasoning,
                    evidence=evidence,
                    confidence=confidence
                )
            )
            
            # Add to final values in the format expected by calculator
            if unit:
                final_values["inputs"][input_name] = [value, unit]
            else:
                final_values["inputs"][input_name] = value
        
        # Calculate a score (this would be delegated to the calculator logic)
        final_values["score"] = "TBD"  # Placeholder - actual calculation would happen elsewhere
        final_values["interpretation"] = "TBD"  # Placeholder
        
        # Create the reasoning result
        reasoning_result = ExtractedValuesWithReasoning(input_values=input_values)
        
        return reasoning_result, final_values
        
    except Exception as e:
        print(f"Error processing extracted values: {e}")
        return ExtractedValuesWithReasoning(input_values=[]), {
            "calculator_name": calculator_name,
            "inputs": {},
            "score": "Error",
            "interpretation": f"Error: {str(e)}"
        }

async def submit_calculator_values(calculator_name: str, url: str, extracted_values: Dict[str, Any], element_id_mapping: Dict[str, Union[int, Dict[str, int]]]) -> CalculatorResult:
    """
    Submits extracted values to the calculator webpage and retrieves the result.
    
    Args:
        calculator_name: Name of the calculator
        url: URL of the calculator
        extracted_values: Dictionary of extracted values (input name -> value)
        element_id_mapping: Mapping between input names and their element IDs
        
    Returns:
        CalculatorResult object containing the score and interpretation
    """
    # Get the LLM
    logger.info(f"Starting submit_calculator_values for {calculator_name} at {url}")
    llm = get_llm()
    
    # Create browser and context
    logger.info("Creating browser instance")
    browser = Browser(
        config=BrowserConfig(
            headless=False,
            disable_security=True,
        )
    )
    
    try:
        async with await browser.new_context() as context:
            # Create controller for interacting with the page
            controller = Controller()
            
            # Create agent
            agent = Agent(
                task=f"""You are an agent whose job is to fill out the {calculator_name} calculator form and get the result.

                Here is the URL of the calculator: {url}
                
                I've extracted values from a patient note that need to be inputted into this calculator:
                {json.dumps(extracted_values["inputs"], indent=2)}
                
                Here's a mapping of input names to their element IDs on the page:
                {json.dumps(element_id_mapping, indent=2)}
                
                First visit the URL. Then, follow these steps:
                1. For each input in the extracted values, locate its corresponding element on the page
                2. If it's a text/numeric input, enter the value
                3. If it's a radio button or dropdown, select the appropriate option
                4. After filling all inputs, find and click the "Calculate" button
                5. After calculation, read and extract the result score and any interpretation text
                
                The result is typically displayed prominently after calculation, often with a heading like "Results" or "Score".
                Extract both the numeric score and any text interpretation of the score.
                
                IMPORTANT: Make sure to handle both numeric inputs and selection options correctly:
                - For numeric inputs, the element_id_mapping contains the element index to input text
                - For option selections, the element_id_mapping contains a dictionary mapping option text to element index
                
                Report back the final score and interpretation in this format:
                {{
                  "score": "the score value",
                  "interpretation": "text explaining what the score means",
                  "additional_info": {{ any additional relevant information }}
                }}
                """,
                llm=llm,
                browser_context=context,
                controller=controller,
                use_vision=True
            )

            # Run the agent and get the history
            history = await agent.run()
            
            # Get the result
            raw_result = history.final_result()
            
            # Process the result
            try:
                if raw_result is None:
                    return CalculatorResult(score="Error", interpretation="Failed to get result from calculator")
                
                if isinstance(raw_result, str):
                    # Try to parse JSON result
                    try:
                        result_data = json.loads(raw_result)
                    except json.JSONDecodeError:
                        # If it's not JSON, try to extract score with a simple pattern
                        import re
                        score_match = re.search(r'score[:\s]+([0-9.]+)', raw_result.lower())
                        score = score_match.group(1) if score_match else "Unknown"
                        return CalculatorResult(score=score, interpretation=raw_result)
                else:
                    result_data = raw_result
                
                # Extract score and interpretation
                score = result_data.get("score", "Unknown")
                interpretation = result_data.get("interpretation", "")
                additional_info = result_data.get("additional_info", {})
                
                return CalculatorResult(
                    score=score,
                    interpretation=interpretation,
                    additional_info=additional_info
                )
                
            except Exception as e:
                print(f"Error processing calculator result: {e}")
                return CalculatorResult(score="Error", interpretation=f"Error processing result: {str(e)}")
    finally:
        await browser.close()

# Example of how to use these functions
async def example_usage():
    """Example of how to use the functions in this module."""
    calculator_name = "Glasgow-Blatchford Bleeding Score (GBS)"
    url = "https://www.mdcalc.com/calc/518/glasgow-blatchford-bleeding-score-gbs"
    
    # Sample patient note
    patient_note = """
    A 65-year-old male presented to the emergency department with chest pain that began 2 hours ago. 
    The pain is described as pressure-like, radiating to the left arm, and rated 8/10 in severity. 
    He has a history of hypertension and hyperlipidemia, for which he takes lisinopril and atorvastatin. 
    He has a 30-pack-year smoking history but quit 5 years ago. His father had an MI at age 60.
    
    Vital signs:
    BP: 140/90 mmHg
    HR: 82 bpm
    RR: 18/min
    Temp: 37.0°C
    O2 Sat: 98% on room air
    
    Physical examination reveals an anxious-appearing man in mild distress. 
    Cardiovascular exam shows regular rhythm without murmurs, rubs, or gallops. 
    Lungs are clear to auscultation bilaterally. 
    ECG shows 1mm ST depression in leads V3-V5. 
    Initial troponin I is 0.04 ng/mL (normal <0.03 ng/mL).
    """
    
    # Step 1: Extract calculator inputs from the website
    parsed_inputs, input_summary, element_id_mapping = await extract_calculator_inputs(calculator_name, url)
    
    # Step 2: Extract values with reasoning from the patient note
    reasoning_result, final_values = await extract_values_with_reasoning(calculator_name, input_summary, patient_note)
    
    # Step 3: Submit the values to the calculator and get the result
    calculator_result = await submit_calculator_values(calculator_name, url, final_values, element_id_mapping)
    
    print(f"Extracted {len(parsed_inputs.inputs)} inputs from calculator: {calculator_name}")
    print(f"Extracted {len(reasoning_result.input_values)} values from patient note")
    print(f"Calculator Result: Score = {calculator_result.score}")
    print(f"Interpretation: {calculator_result.interpretation}")

# Add a debug test function
async def test_browser(calculator_url=None):
    """
    Test function to check if the browser can be initialized properly.
    
    Args:
        calculator_url: Optional URL to test navigation to. If None, will just test browser initialization.
    
    Returns:
        True if the test was successful, False otherwise.
    """
    logger.info("Starting browser test function")
    browser = None
    try:
        if calculator_url:
            logger.info(f"Will test navigation to: {calculator_url}")
        else:
            logger.info("No URL provided, will only test browser initialization")
        
        # First ensure the LLM can be initialized
        try:
            logger.info("Testing LLM initialization")
            llm = get_llm()
            logger.info("Successfully initialized LLM")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            return False
            
        # Create browser with robust error handling
        logger.info("Creating browser instance")
        try:
            browser = Browser(
                config=BrowserConfig(
                    headless=False,
                    disable_security=True,
                )
            )
            logger.info("Successfully created browser instance")
        except Exception as e:
            logger.error(f"Failed to create browser: {e}")
            logger.error(traceback.format_exc())
            return False
        
        # Create context with timeout and error handling
        logger.info("Creating new browser context")
        try:
            context = await asyncio.wait_for(browser.new_context(), timeout=30.0)
            logger.info("Successfully created browser context")
        except asyncio.TimeoutError:
            logger.error("Timed out creating browser context")
            return False
        except Exception as e:
            logger.error(f"Error creating browser context: {e}")
            logger.error(traceback.format_exc())
            return False
        
        # If a URL was provided, try to navigate to it
        if calculator_url:
            try:
                async with context:
                    logger.info(f"Navigating to calculator: {calculator_url}")
                    # Create a controller and agent for navigation
                    controller = Controller()
                    
                    # Use an agent to visit the URL, similar to how it's done in the main functions
                    agent = Agent(
                        task=f"Visit this medical calculator: {calculator_url}",
                        llm=llm,
                        browser_context=context,
                        controller=controller,
                        use_vision=True
                    )
                    
                    # Run the agent with a timeout
                    logger.info("Running test navigation (with 60 second timeout)")
                    history = await asyncio.wait_for(agent.run(), timeout=60.0)
                    
                    logger.info("Successfully navigated to calculator webpage")
                    await asyncio.sleep(2)  # Wait a moment to let page load completely
            except asyncio.TimeoutError:
                logger.error("Timed out during calculator navigation")
                return False
            except Exception as e:
                logger.error(f"Error during calculator navigation: {e}")
                logger.error(traceback.format_exc())
                return False
        
        logger.info("Browser test completed successfully")
        return True
    except Exception as e:
        logger.error(f"Browser test failed: {str(e)}")
        logger.error(traceback.format_exc())
        return False
    finally:
        if browser:
            try:
                logger.info("Closing test browser")
                await browser.close()
            except Exception as e:
                logger.error(f"Error closing test browser: {str(e)}")

if __name__ == "__main__":
    # Example with default URL for testing module directly
    calculator_name = "Glasgow-Blatchford Bleeding Score (GBS)"
    calculator_url = "https://www.mdcalc.com/calc/518/glasgow-blatchford-bleeding-score-gbs"
    
    logger.info("=== STARTING WORKFLOW FUNCTIONS MODULE TEST ===")
    
    # Run the browser test with the example URL
    logger.info("Running browser test")
    test_result = asyncio.run(test_browser(calculator_url))
    
    if test_result:
        logger.info("Browser test passed, running example usage")
        asyncio.run(example_usage())
    else:
        logger.error("Browser test failed, skipping example usage") 