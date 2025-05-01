import os
import json
import csv
import glob
from datetime import datetime

# CSV file containing calculator instances with ground truth
CSV_FILE = "first_calculator_instances.csv"

# Results directory
RESULTS_DIR = "results"

# New summary file
VALIDATION_SUMMARY = f"results/validation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

def load_csv_data():
    """Load the ground truth data from the CSV file"""
    csv_data = {}
    with open(CSV_FILE, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(rows := list(reader)):
            row_num = i + 1
            csv_data[row_num] = {
                'calculator_id': row.get('Calculator ID', ''),
                'calculator_name': row.get('Calculator Name', ''),
                'ground_truth': row.get('Ground Truth Answer', None),
                'lower_limit': row.get('Lower Limit', None),
                'upper_limit': row.get('Upper Limit', None)
            }
    return csv_data, len(rows)

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

def find_result_files():
    """Find all result JSON files in the results directory"""
    return glob.glob(os.path.join(RESULTS_DIR, "calc_*.json"))

def validate_results():
    """Validate all calculator results against ground truth"""
    # Load ground truth data
    csv_data, total_rows = load_csv_data()
    print(f"Loaded {total_rows} rows from {CSV_FILE}")
    
    # Find all result files
    result_files = find_result_files()
    print(f"Found {len(result_files)} result files in {RESULTS_DIR}")
    
    # Create validation summary file
    os.makedirs(os.path.dirname(VALIDATION_SUMMARY), exist_ok=True)
    with open(VALIDATION_SUMMARY, 'w', newline='') as csvfile:
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
            'Result File'
        ])
    
    # Process each result file
    total_validated = 0
    within_bounds_count = 0
    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                result_data = json.load(f)
            
            # Extract data
            row_number = result_data.get('row_number')
            calculator_id = result_data.get('calculator_id')
            calculator_name = result_data.get('calculator_name')
            
            if not row_number:
                # Try to extract row number from filename
                filename = os.path.basename(result_file)
                if 'row' in filename:
                    try:
                        row_number = int(filename.split('row')[1].split('_')[0])
                    except (ValueError, IndexError):
                        row_number = None
            
            # Skip if row number not found
            if not row_number:
                print(f"Warning: Could not determine row number for {result_file}, skipping")
                continue
            
            # Get ground truth data
            ground_truth_data = csv_data.get(row_number)
            if not ground_truth_data:
                print(f"Warning: No ground truth data for row {row_number} in {result_file}, skipping")
                continue
            
            # Get calculation result
            result = result_data.get('result', {})
            calculated_answer = result.get('answer')
            
            # Get bounds
            ground_truth = ground_truth_data.get('ground_truth')
            lower_limit = ground_truth_data.get('lower_limit')
            upper_limit = ground_truth_data.get('upper_limit')
            
            # Check if within bounds
            within_bounds = is_within_bounds(calculated_answer, lower_limit, upper_limit)
            
            # Write to summary
            with open(VALIDATION_SUMMARY, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    row_number,
                    calculator_id,
                    calculator_name,
                    calculated_answer,
                    ground_truth,
                    lower_limit,
                    upper_limit,
                    within_bounds,
                    os.path.basename(result_file)
                ])
            
            # Count successful validations
            total_validated += 1
            if within_bounds == "Yes":
                within_bounds_count += 1
                
            # Print validation result
            status = "✅" if within_bounds == "Yes" else "❌" if within_bounds == "No" else "⚠️"
            print(f"{status} Row {row_number}: {calculator_name}")
            print(f"   Answer: {calculated_answer}, Ground Truth: {ground_truth}, Bounds: {lower_limit} - {upper_limit}")
            
        except Exception as e:
            print(f"Error processing {result_file}: {str(e)}")
    
    # Print summary
    print("\n=== VALIDATION SUMMARY ===")
    print(f"Total results validated: {total_validated}")
    if total_validated > 0:
        within_bounds_percent = (within_bounds_count / total_validated) * 100
        print(f"Results within bounds: {within_bounds_count} ({within_bounds_percent:.2f}%)")
    print(f"Detailed summary saved to: {VALIDATION_SUMMARY}")

if __name__ == "__main__":
    validate_results() 