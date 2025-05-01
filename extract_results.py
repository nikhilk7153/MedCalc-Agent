import re
import json
import os
import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_numerical_answer(result_str):
    """
    Extract numerical answer from a result string using multiple regex patterns
    to ensure all decimal places are preserved.
    
    Args:
        result_str: String containing the result
        
    Returns:
        Float value of the extracted answer or None if no value found
    """
    answer_value = None
    
    try:
        # First try: Look for JSON pattern {"answer": X.XX}
        json_match = re.search(r'"answer"\s*:\s*(\d+\.\d+)', result_str)
        if json_match:
            answer_value = float(json_match.group(1))
            logger.info(f"Extracted answer value using JSON regex: {answer_value}")
            return answer_value
        
        # Second try: Look for decimal value with multiple digits after decimal point
        number_match = re.search(r'(\d+\.\d+)', result_str)
        if number_match:
            answer_value = float(number_match.group(1))
            logger.info(f"Extracted answer value using decimal regex: {answer_value}")
            return answer_value
        
        # Third try: Look for integer value as fallback
        number_match = re.search(r'(\d+)', result_str)
        if number_match:
            answer_value = float(number_match.group(1))
            logger.info(f"Extracted answer value using integer regex: {answer_value}")
            return answer_value
        
    except Exception as e:
        logger.error(f"Error extracting numerical value: {str(e)}")
    
    return answer_value

def save_results_to_json(result_dict, calculator_name=None):
    """
    Save calculation results to a JSON file in the results directory
    
    Args:
        result_dict: Dictionary containing the result data
        calculator_name: Name of the calculator (used for filename)
        
    Returns:
        Path to the saved file or None if saving failed
    """
    try:
        # Create a results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
        
        # Generate a filename based on calculator name and timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        calc_name = calculator_name or "calculation"
        filename = f"results/{calc_name.replace(' ', '_')}_{timestamp}.json"
        
        # Write the results to the file
        with open(filename, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        logger.info(f"Results saved to {filename}")
        return filename
    
    except Exception as save_error:
        logger.error(f"Failed to save results to JSON: {str(save_error)}")
        return None

def process_calculation_result(calculator_result, calculator_name="Medical Calculator"):
    """
    Process a calculator result to extract the numerical value and format it
    
    Args:
        calculator_result: The result string or dictionary from the calculator
        calculator_name: Name of the calculator
        
    Returns:
        Dictionary with the formatted result
    """
    # Convert to string if not already
    result_str = str(calculator_result)
    
    # Extract numerical answer
    answer_value = extract_numerical_answer(result_str)
    
    # Create result dictionary
    if answer_value is not None:
        result = {
            "calculator_name": calculator_name,
            "answer": answer_value,
            "raw_result": result_str
        }
        
        # Display the result prominently
        logger.info("")
        logger.info("===========================================")
        logger.info(f"FINAL ANSWER: {answer_value}")
        logger.info("===========================================")
        logger.info("")
    else:
        result = {
            "calculator_name": calculator_name,
            "error": "No numerical result extracted",
            "raw_result": result_str
        }
    
    # Save the result to JSON
    save_results_to_json(result, calculator_name)
    
    return result

# Example usage
if __name__ == "__main__":
    # Test with a mock result
    test_result = '{"calculator_name": "FIB-4 Index for Liver Fibrosis", "score": "3.87"}'
    processed = process_calculation_result(test_result, "FIB-4 Index")
    print(f"Processed result: {processed}") 