import asyncio
import json
import os
import csv
import time
from openai import AzureOpenAI
from browser_calculator import run_browser_calculator
from dotenv import load_dotenv

load_dotenv()

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_version='2024-02-15-preview',
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

# Load calculator map from JSON file
def load_calculator_map():
    try:
        with open("calculator_map.json", "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading calculator_map.json: {e}")
        return {}

# Global calculator map
CALCULATOR_MAP = load_calculator_map()

# CSV file containing calculator instances
CSV_FILE = "first_calculator_instances.csv"

# Create a summary file for all results
SUMMARY_FILE = f"results/summary_{time.strftime('%Y%m%d_%H%M%S')}.csv"

# Set up summary file header
def setup_summary_file():
    os.makedirs('results', exist_ok=True)
    with open(SUMMARY_FILE, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'Row Number', 
            'Calculator ID', 
            'Calculator Name', 
            'Calculated Answer', 
            'Ground Truth', 
            'Lower Bound', 
            'Upper Bound', 
            'Within Bounds?', 
            'Result Status',
            'Screenshot Path'
        ])
    print(f"Created summary file at {SUMMARY_FILE}")

# Validate if an answer is within bounds
def is_within_bounds(answer, lower_bound, upper_bound):
    try:
        answer_float = float(answer) if answer is not None else None
        lower_float = float(lower_bound) if lower_bound is not None else None
        upper_float = float(upper_bound) if upper_bound is not None else None
        
        if answer_float is None or lower_float is None or upper_float is None:
            return "Unknown"
        
        if lower_float <= answer_float <= upper_float:
            return "Yes"
        else:
            return "No"
    except (ValueError, TypeError):
        return "Error"

# Add result to summary file
def add_to_summary(row_number, row, result):
    calculated_answer = result.get('answer', None)
    ground_truth = row.get('Ground Truth Answer', None)
    lower_bound = row.get('Lower Limit', None)
    upper_bound = row.get('Upper Limit', None)
    
    within_bounds = is_within_bounds(calculated_answer, lower_bound, upper_bound)
    
    screenshot_path = result.get('screenshot_path', "None")
    if screenshot_path and isinstance(screenshot_path, str):
        screenshot_path = os.path.basename(screenshot_path)
    
    with open(SUMMARY_FILE, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            row_number,
            row.get('Calculator ID', ''),
            row.get('Calculator Name', ''),
            calculated_answer,
            ground_truth,
            lower_bound,
            upper_bound,
            within_bounds,
            "Success" if result.get('success', False) else "Failure",
            screenshot_path
        ])

async def process_calculator_instance(row, row_number):
    try:
        calculator_id = row['Calculator ID']
        calculator_name = row['Calculator Name']
        patient_note = row['Patient Note']
        question = row['Question']
        
        # Get calculator URL from the calculator map
        calculator_url = CALCULATOR_MAP.get(calculator_name)
        if not calculator_url:
            raise ValueError(f"URL not found for calculator: {calculator_name}")
        
        print(f"\n\n=== Processing {calculator_name} (Row {row_number}) ===")
        print(f"Question: {question}")
        
        # Combine the patient note with the question for better context
        enhanced_patient_data = f"QUESTION: {question}\n\nPATIENT DATA: {patient_note}"
        
        # Run the browser calculator
        result = await run_browser_calculator(
            calculator_name=calculator_name,
            calculator_url=calculator_url,
            patient_data=enhanced_patient_data,
            llm_client=client
        )
        
        # Save the result with the row information
        result_with_question = {
            "row_number": row_number,
            "calculator_id": calculator_id,
            "calculator_name": calculator_name,
            "question": question,
            "ground_truth": row.get('Ground Truth Answer', None),
            "lower_bound": row.get('Lower Limit', None),
            "upper_bound": row.get('Upper Limit', None),
            "result": result
        }
        
        # Save to a JSON file
        os.makedirs('results', exist_ok=True)
        results_file = f"results/calc_{calculator_id}_row{row_number}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(result_with_question, f, indent=2)
        
        print(f"Result saved to {results_file}")
        
        # Print success or failure message
        if result.get("success", False):
            answer = result.get('answer', 'Unknown')
            ground_truth = row.get('Ground Truth Answer', None)
            lower_bound = row.get('Lower Limit', None)
            upper_bound = row.get('Upper Limit', None)
            
            print(f"Success! Calculated answer: {answer}")
            print(f"Ground truth: {ground_truth} (Bounds: {lower_bound} - {upper_bound})")
            
            # Check if answer is within bounds
            within_bounds = is_within_bounds(answer, lower_bound, upper_bound)
            if within_bounds == "Yes":
                print(f"✅ Answer is within expected bounds!")
            elif within_bounds == "No":
                print(f"❌ Answer is outside expected bounds!")
            else:
                print(f"⚠️ Could not validate bounds")
                
            if "screenshot_path" in result:
                print(f"Screenshot saved at: {result['screenshot_path']}")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
            if "missing_values" in result:
                print(f"Missing values: {', '.join(result.get('missing_values', []))}")
        
        # Add to summary file
        add_to_summary(row_number, row, result)
                
        # Wait a bit between calculations to avoid overwhelming the browser
        await asyncio.sleep(5)
        
        return result
        
    except Exception as e:
        print(f"Exception processing {row.get('Calculator Name', 'unknown calculator')}: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Add error to summary
        error_result = {"success": False, "error": str(e)}
        add_to_summary(row_number, row, error_result)
        
        return {"success": False, "error": str(e)}

async def process_csv():
    # Set up summary file
    setup_summary_file()
    
    # Read the CSV file
    with open(CSV_FILE, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)
    
    print(f"Found {len(rows)} calculator instances in {CSV_FILE}")
    
    # Ask which instances to process
    print("\nOptions:")
    print("1. Process all instances")
    print("2. Process a specific instance (by row number)")
    print("3. Process a range of instances")
    
    choice = input("Enter your choice (1-3): ")
    
    if choice == '1':
        # Process all instances
        for i, row in enumerate(rows):
            row_number = i + 1
            await process_calculator_instance(row, row_number)
    elif choice == '2':
        # Process a specific instance
        row_num = int(input(f"Enter row number (1-{len(rows)}): "))
        if 1 <= row_num <= len(rows):
            await process_calculator_instance(rows[row_num-1], row_num)
        else:
            print(f"Invalid row number. Must be between 1 and {len(rows)}")
    elif choice == '3':
        # Process a range of instances
        start = int(input(f"Enter start row (1-{len(rows)}): "))
        end = int(input(f"Enter end row (1-{len(rows)}): "))
        if 1 <= start <= end <= len(rows):
            for i in range(start-1, end):
                row_number = i + 1
                await process_calculator_instance(rows[i], row_number)
        else:
            print(f"Invalid range. Must be between 1 and {len(rows)}")
    else:
        print("Invalid choice")
    
    print(f"\nAll processing complete! Summary of results saved to {SUMMARY_FILE}")

if __name__ == "__main__":
    asyncio.run(process_csv()) 