import asyncio
import json
import os
import csv
import time
import argparse
import concurrent.futures
from openai import AzureOpenAI
from browser_calculator import run_browser_calculator
import subprocess
import sys

# CSV file containing calculator instances
CSV_FILE = "first_calculator_instances.csv"

# Define model configurations
MODEL_CONFIGS = {
    "gpt4o_vision": {
        "model": "gpt-4o",
        "use_vision": True,
        "dir": "results_gpt4o_vision"
    },
    "gpt4o_no_vision": {
        "model": "gpt-4o",
        "use_vision": False,
        "dir": "results_gpt4o_no_vision"
    },
    "gpt4o_mini_vision": {
        "model": "gpt-4o-mini",
        "use_vision": True,
        "dir": "results_gpt4o_mini_vision"
    },
    "gpt4o_mini_no_vision": {
        "model": "gpt-4o-mini",
        "use_vision": False,
        "dir": "results_gpt4o_mini_no_vision"
    }
}

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

def initialize_client():
    """Initialize Azure OpenAI client"""
    return AzureOpenAI(
        api_version='2024-02-15-preview',
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://azure-openai-miblab-ncu.openai.azure.com/"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY", "9a8467bfe81d443d97b1af452662c33c"),
    )

def setup_summary_file(summary_file):
    """Set up a summary file with headers"""
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    with open(summary_file, 'w', newline='') as csvfile:
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
            'Model',
            'Vision',
            'Time (s)',
            'Result Status',
            'Screenshot Path'
        ])
    print(f"Created summary file at {summary_file}")
    return summary_file

def is_within_bounds(answer, lower_bound, upper_bound):
    """Check if an answer is within the specified bounds"""
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

def add_to_summary(summary_file, row_number, row, result, model_name, use_vision, execution_time):
    """Add a result to the summary file"""
    calculated_answer = result.get('answer', None)
    ground_truth = row.get('Ground Truth Answer', None)
    lower_bound = row.get('Lower Limit', None)
    upper_bound = row.get('Upper Limit', None)
    
    within_bounds = is_within_bounds(calculated_answer, lower_bound, upper_bound)
    
    screenshot_path = result.get('screenshot_path', "None")
    if screenshot_path and isinstance(screenshot_path, str):
        screenshot_path = os.path.basename(screenshot_path)
    
    with open(summary_file, 'a', newline='') as csvfile:
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
            model_name,
            "Yes" if use_vision else "No",
            f"{execution_time:.2f}",
            "Success" if result.get('success', False) else "Failure",
            screenshot_path
        ])

async def run_single_model(config_key, row, row_number, global_summary_file):
    """Run a single calculator with the specified model configuration"""
    client = initialize_client()
    config = MODEL_CONFIGS[config_key]
    model_name = config["model"]
    use_vision = config["use_vision"]
    results_dir = config["dir"]
    
    # Create model-specific directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Model-specific summary file
    model_summary_file = f"{results_dir}/summary_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    setup_summary_file(model_summary_file)
    
    try:
        calculator_id = row['Calculator ID']
        calculator_name = row['Calculator Name']
        patient_note = row['Patient Note']
        question = row['Question']
        
        # Get calculator URL from the calculator map
        calculator_url = CALCULATOR_MAP.get(calculator_name)
        if not calculator_url:
            raise ValueError(f"URL not found for calculator: {calculator_name}")
        
        print(f"\n=== Processing {calculator_name} (Row {row_number}) with {model_name} (Vision: {use_vision}) ===")
        
        # Combine the patient note with the question for better context
        enhanced_patient_data = f"QUESTION: {question}\n\nPATIENT DATA: {patient_note}"
        
        # Record start time
        start_time = time.time()
        
        # Set environment variables for the model configuration
        os.environ["BROWSER_USE_MODEL"] = model_name
        os.environ["BROWSER_USE_VISION"] = "true" if use_vision else "false"
        
        # Run the browser calculator
        result = await run_browser_calculator(
            calculator_name=calculator_name,
            calculator_url=calculator_url,
            patient_data=enhanced_patient_data,
            llm_client=client
        )
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Save the result with the row information
        result_with_question = {
            "row_number": row_number,
            "calculator_id": calculator_id,
            "calculator_name": calculator_name,
            "question": question,
            "ground_truth": row.get('Ground Truth Answer', None),
            "lower_bound": row.get('Lower Limit', None),
            "upper_bound": row.get('Upper Limit', None),
            "model": model_name,
            "vision": use_vision,
            "execution_time": execution_time,
            "result": result
        }
        
        # Save to a JSON file in the model-specific directory
        results_file = f"{results_dir}/calc_{calculator_id}_row{row_number}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(result_with_question, f, indent=2)
        
        # Print result info
        if result.get("success", False):
            answer = result.get('answer', 'Unknown')
            ground_truth = row.get('Ground Truth Answer', None)
            lower_bound = row.get('Lower Limit', None)
            upper_bound = row.get('Upper Limit', None)
            
            within_bounds = is_within_bounds(answer, lower_bound, upper_bound)
            status = "✅" if within_bounds == "Yes" else "❌" if within_bounds == "No" else "⚠️"
            
            print(f"{status} {model_name} (Vision: {use_vision}): Answer={answer}, Time={execution_time:.2f}s")
            print(f"   Ground truth: {ground_truth} (Bounds: {lower_bound} - {upper_bound})")
        else:
            print(f"❌ {model_name} (Vision: {use_vision}): Failed, Time={execution_time:.2f}s")
            print(f"   Error: {result.get('error', 'Unknown error')}")
        
        # Add to model-specific summary file
        add_to_summary(model_summary_file, row_number, row, result, model_name, use_vision, execution_time)
        
        # Add to global summary file
        add_to_summary(global_summary_file, row_number, row, result, model_name, use_vision, execution_time)
        
        return result
        
    except Exception as e:
        print(f"Exception with {model_name} (Vision: {use_vision}): {str(e)}")
        import traceback
        traceback.print_exc()
        
        error_result = {"success": False, "error": str(e)}
        execution_time = 0
        
        # Add error to summary files
        add_to_summary(model_summary_file, row_number, row, error_result, model_name, use_vision, execution_time)
        add_to_summary(global_summary_file, row_number, row, error_result, model_name, use_vision, execution_time)
        
        return error_result

async def run_parallel_models(row, row_number, configs_to_run, global_summary_file):
    """Run multiple model configurations in parallel"""
    tasks = []
    for config_key in configs_to_run:
        tasks.append(run_single_model(config_key, row, row_number, global_summary_file))
    
    # Run tasks in parallel
    return await asyncio.gather(*tasks)

async def process_csv(args):
    """Process the CSV file with the specified model configurations"""
    # Set up global summary file
    global_summary_file = f"results/comparison_summary_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    setup_summary_file(global_summary_file)
    
    # Read the CSV file
    with open(CSV_FILE, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)
    
    print(f"Found {len(rows)} calculator instances in {CSV_FILE}")
    
    # Determine which configurations to run
    configs_to_run = []
    if args.all or args.gpt4o_vision:
        configs_to_run.append("gpt4o_vision")
    if args.all or args.gpt4o_no_vision:
        configs_to_run.append("gpt4o_no_vision")
    if args.all or args.gpt4o_mini_vision:
        configs_to_run.append("gpt4o_mini_vision")
    if args.all or args.gpt4o_mini_no_vision:
        configs_to_run.append("gpt4o_mini_no_vision")
    
    if not configs_to_run:
        print("No configurations selected. Use --all or specify individual configurations.")
        return
    
    print(f"Running with configurations: {', '.join(configs_to_run)}")
    
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
            await run_parallel_models(row, row_number, configs_to_run, global_summary_file)
    elif choice == '2':
        # Process a specific instance
        row_num = int(input(f"Enter row number (1-{len(rows)}): "))
        if 1 <= row_num <= len(rows):
            await run_parallel_models(rows[row_num-1], row_num, configs_to_run, global_summary_file)
        else:
            print(f"Invalid row number. Must be between 1 and {len(rows)}")
    elif choice == '3':
        # Process a range of instances
        start = int(input(f"Enter start row (1-{len(rows)}): "))
        end = int(input(f"Enter end row (1-{len(rows)}): "))
        if 1 <= start <= end <= len(rows):
            for i in range(start-1, end):
                row_number = i + 1
                await run_parallel_models(rows[i], row_number, configs_to_run, global_summary_file)
        else:
            print(f"Invalid range. Must be between 1 and {len(rows)}")
    else:
        print("Invalid choice")
    
    print(f"\nAll processing complete! Summary of results saved to {global_summary_file}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run browser calculator with different model configurations')
    parser.add_argument('--all', action='store_true', help='Run all model configurations')
    parser.add_argument('--gpt4o-vision', action='store_true', help='Run GPT-4o with vision')
    parser.add_argument('--gpt4o-no-vision', action='store_true', help='Run GPT-4o without vision')
    parser.add_argument('--gpt4o-mini-vision', action='store_true', help='Run GPT-4o-mini with vision')
    parser.add_argument('--gpt4o-mini-no-vision', action='store_true', help='Run GPT-4o-mini without vision')
    return parser.parse_args()

def run_in_separate_process(config_key, row, row_number):
    """Run a model configuration in a separate process"""
    command = [
        sys.executable, 
        "run_single_config.py",
        "--config", config_key,
        "--row", str(row_number)
    ]
    process = subprocess.Popen(command)
    return process

if __name__ == "__main__":
    args = parse_arguments()
    asyncio.run(process_csv(args)) 