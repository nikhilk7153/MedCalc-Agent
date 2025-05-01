import csv
import sys
import os
from collections import Counter

def analyze_results(csv_path):
    """
    Analyze calculator results from the summary CSV file
    """
    if not os.path.exists(csv_path):
        print(f"Error: File {csv_path} not found.")
        return
    
    # Statistics
    stats = {
        "total": 0,
        "success": 0,
        "failed": 0,
        "within_bounds": 0,
        "outside_bounds": 0,
        "unknown_bounds": 0,
        "by_calculator": {}
    }
    
    # Read CSV file
    with open(csv_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        
        # Check if the expected columns exist
        first_row = next(reader, None)
        if not first_row:
            print("No data found in CSV file.")
            return
        
        # Check column names
        required_columns = ["Calculator Name", "Within Bounds?", "Result Status"]
        missing_columns = [col for col in required_columns if col not in first_row]
        
        if missing_columns:
            print(f"Error: Missing columns: {', '.join(missing_columns)}")
            print(f"Available columns: {', '.join(first_row.keys())}")
            return
        
        # Process first row
        process_row(first_row, stats)
        
        # Process remaining rows
        for row in reader:
            process_row(row, stats)
    
    # Print results
    print(f"\n=== RESULTS ANALYSIS: {csv_path} ===")
    print(f"Total calculator runs: {stats['total']}")
    print(f"Successful runs: {stats['success']} ({stats['success']/stats['total']*100:.1f}%)")
    print(f"Failed runs: {stats['failed']} ({stats['failed']/stats['total']*100:.1f}%)")
    print("\nBOUNDS VALIDATION:")
    print(f"Within bounds: {stats['within_bounds']} ({stats['within_bounds']/stats['total']*100:.1f}%)")
    print(f"Outside bounds: {stats['outside_bounds']} ({stats['outside_bounds']/stats['total']*100:.1f}%)")
    print(f"Unknown bounds: {stats['unknown_bounds']} ({stats['unknown_bounds']/stats['total']*100:.1f}%)")
    
    print("\nRESULTS BY CALCULATOR:")
    for calc_name, calc_stats in stats["by_calculator"].items():
        success_rate = calc_stats["success"] / calc_stats["total"] * 100
        bounds_rate = calc_stats["within_bounds"] / calc_stats["total"] * 100 if calc_stats["total"] > 0 else 0
        print(f"{calc_name}: {calc_stats['total']} runs, {success_rate:.1f}% success, {bounds_rate:.1f}% within bounds")

def process_row(row, stats):
    """Process a single row from the CSV"""
    stats["total"] += 1
    
    # Get calculator name
    calc_name = row.get("Calculator Name", "Unknown")
    if calc_name not in stats["by_calculator"]:
        stats["by_calculator"][calc_name] = {
            "total": 0,
            "success": 0,
            "failed": 0,
            "within_bounds": 0,
            "outside_bounds": 0,
            "unknown_bounds": 0
        }
    
    # Update calculator-specific stats
    calc_stats = stats["by_calculator"][calc_name]
    calc_stats["total"] += 1
    
    # Check success status
    result_status = row.get("Result Status", "").lower()
    if result_status == "success":
        stats["success"] += 1
        calc_stats["success"] += 1
    else:
        stats["failed"] += 1
        calc_stats["failed"] += 1
    
    # Check bounds status
    bounds_status = row.get("Within Bounds?", "").lower()
    if bounds_status == "yes":
        stats["within_bounds"] += 1
        calc_stats["within_bounds"] += 1
    elif bounds_status == "no":
        stats["outside_bounds"] += 1
        calc_stats["outside_bounds"] += 1
    else:
        stats["unknown_bounds"] += 1
        calc_stats["unknown_bounds"] += 1

if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        # Default to the CSV file mentioned in the query
        csv_path = "results/summary_20250430_213851.csv"
        print(f"No CSV file specified, using default: {csv_path}")
    
    analyze_results(csv_path) 