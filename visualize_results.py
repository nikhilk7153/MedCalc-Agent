import csv
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def load_data(csv_path):
    """Load and process the CSV data into a DataFrame"""
    if not os.path.exists(csv_path):
        print(f"Error: File {csv_path} not found.")
        return None
    
    try:
        # Read the CSV into a pandas DataFrame
        df = pd.read_csv(csv_path)
        
        # Check for required columns
        required_columns = ["Calculator Name", "Within Bounds?", "Result Status"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Error: Missing columns: {', '.join(missing_columns)}")
            print(f"Available columns: {', '.join(df.columns)}")
            return None
        
        # Clean the data
        # Convert 'Within Bounds?' to a categorical variable
        if 'Within Bounds?' in df.columns:
            df['Within Bounds?'] = df['Within Bounds?'].str.lower()
            df['Bounds Status'] = df['Within Bounds?'].map({
                'yes': 'Within Bounds', 
                'no': 'Outside Bounds',
                'unknown': 'Unknown'
            })
            df['Bounds Status'] = df['Bounds Status'].fillna('Unknown')
            
        # Clean up 'Result Status' 
        if 'Result Status' in df.columns:
            df['Result Status'] = df['Result Status'].str.lower()
            df['Result Status'] = df['Result Status'].map({
                'success': 'Success', 
                'failure': 'Failure'
            })
            df['Result Status'] = df['Result Status'].fillna('Unknown')
            
        # Try to convert Calculated Answer to numeric
        if 'Calculated Answer' in df.columns:
            df['Calculated Answer'] = pd.to_numeric(df['Calculated Answer'], errors='coerce')
            
        # Try to convert Ground Truth to numeric
        if 'Ground Truth' in df.columns:
            df['Ground Truth'] = pd.to_numeric(df['Ground Truth'], errors='coerce')
            
        # Try to convert Trajectory Length to numeric if it exists
        if 'Trajectory Length' in df.columns:
            df['Trajectory Length'] = pd.to_numeric(df['Trajectory Length'], errors='coerce')
            
        # Try to process model information if available
        if 'Model' in df.columns:
            df['Model'] = df['Model'].fillna('Unknown')
            
        if 'Vision' in df.columns:
            df['Vision'] = df['Vision'].fillna('Unknown')
            df['Vision'] = df['Vision'].map({'Yes': 'With Vision', 'No': 'Without Vision'})

        return df
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def create_output_dir():
    """Create output directory for visualizations"""
    output_dir = "visualizations"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def visualize_results(df, output_dir):
    """Generate multiple visualizations from the results data"""
    if df is None or df.empty:
        print("No data to visualize.")
        return
        
    # Set the style
    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # List to track all generated files
    generated_files = []
    
    try:
        # 1. Overall Success Rate Pie Chart
        fig, ax = plt.subplots(figsize=(10, 6))
        success_counts = df['Result Status'].value_counts()
        colors = ['#4CAF50', '#F44336', '#9E9E9E']  # Green for success, red for failure
        ax.pie(success_counts, labels=success_counts.index, autopct='%1.1f%%', 
               startangle=90, colors=colors, explode=[0.05] * len(success_counts))
        ax.set_title('Overall Success Rate')
        plt.tight_layout()
        
        file_path = os.path.join(output_dir, '1_success_rate_pie.png')
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        generated_files.append(file_path)
        plt.close()

        # 2. Bounds Validation Pie Chart
        fig, ax = plt.subplots(figsize=(10, 6))
        bounds_counts = df['Bounds Status'].value_counts()
        colors = ['#4CAF50', '#F44336', '#FFC107']  # Green, Red, Yellow
        ax.pie(bounds_counts, labels=bounds_counts.index, autopct='%1.1f%%', 
               startangle=90, colors=colors, explode=[0.05] * len(bounds_counts))
        ax.set_title('Results Bounds Validation')
        plt.tight_layout()
        
        file_path = os.path.join(output_dir, '2_bounds_validation_pie.png')
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        generated_files.append(file_path)
        plt.close()

        # 3. Success Rate by Calculator Bar Chart
        plt.figure(figsize=(14, 8))
        calc_success = df.groupby('Calculator Name')['Result Status'].apply(
            lambda x: (x == 'Success').mean() * 100
        ).sort_values(ascending=False)
        
        # Fix barplot with explicit hue assignment
        df_bar = pd.DataFrame({'Calculator': calc_success.index, 'Success Rate': calc_success.values})
        ax = sns.barplot(x='Calculator', y='Success Rate', data=df_bar, hue='Calculator', palette='viridis', legend=False)
        plt.title('Success Rate by Calculator')
        plt.xlabel('Calculator')
        plt.ylabel('Success Rate (%)')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 100)
        plt.axhline(y=calc_success.mean(), color='r', linestyle='--', label=f'Average ({calc_success.mean():.1f}%)')
        plt.legend()
        plt.tight_layout()
        
        file_path = os.path.join(output_dir, '3_success_by_calculator.png')
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        generated_files.append(file_path)
        plt.close()

        # 4. Bounds Validation by Calculator
        plt.figure(figsize=(14, 8))
        calc_bounds = df.groupby('Calculator Name')['Bounds Status'].apply(
            lambda x: (x == 'Within Bounds').mean() * 100
        ).sort_values(ascending=False)
        
        # Fix barplot with explicit hue assignment
        df_bar = pd.DataFrame({'Calculator': calc_bounds.index, 'Within Bounds Rate': calc_bounds.values})
        ax = sns.barplot(x='Calculator', y='Within Bounds Rate', data=df_bar, hue='Calculator', palette='viridis', legend=False)
        plt.title('Within Bounds Rate by Calculator')
        plt.xlabel('Calculator')
        plt.ylabel('Within Bounds Rate (%)')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 100)
        plt.axhline(y=calc_bounds.mean(), color='r', linestyle='--', label=f'Average ({calc_bounds.mean():.1f}%)')
        plt.legend()
        plt.tight_layout()
        
        file_path = os.path.join(output_dir, '4_bounds_by_calculator.png')
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        generated_files.append(file_path)
        plt.close()

        # 5. Heatmap of Success vs Bounds
        plt.figure(figsize=(10, 6))
        heatmap_data = pd.crosstab(df['Result Status'], df['Bounds Status'], normalize='all') * 100
        sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', fmt='.1f')
        plt.title('Success vs Bounds Validation (%)')
        plt.tight_layout()
        
        file_path = os.path.join(output_dir, '5_success_vs_bounds_heatmap.png')
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        generated_files.append(file_path)
        plt.close()
        
        # NEW: Trajectory Length vs Bounds Status
        if 'Trajectory Length' in df.columns:
            plt.figure(figsize=(12, 8))
            
            # Create a box plot
            ax = sns.boxplot(x='Bounds Status', y='Trajectory Length', data=df, palette={
                'Within Bounds': '#4CAF50',
                'Outside Bounds': '#F44336',
                'Unknown': '#9E9E9E'
            })
            
            # Add swarm points for individual data points
            sns.swarmplot(x='Bounds Status', y='Trajectory Length', data=df, 
                          alpha=0.6, size=4, color='.3')
            
            plt.title('Trajectory Length vs Bounds Status')
            plt.xlabel('Bounds Status')
            plt.ylabel('Trajectory Length (number of steps)')
            
            # Add statistics
            for bounds_status in df['Bounds Status'].unique():
                subset = df[df['Bounds Status'] == bounds_status]
                if not subset.empty:
                    avg = subset['Trajectory Length'].mean()
                    plt.text(
                        list(df['Bounds Status'].unique()).index(bounds_status),
                        avg,
                        f'Mean: {avg:.1f}',
                        ha='center', va='bottom', color='black', fontweight='bold'
                    )
            
            plt.tight_layout()
            
            file_path = os.path.join(output_dir, '11_trajectory_length_vs_bounds.png')
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            generated_files.append(file_path)
            plt.close()
        
        # Check if we have model data for further analysis
        if 'Model' in df.columns and 'Vision' in df.columns:
            # 6. Success Rate by Model and Vision
            plt.figure(figsize=(12, 7))
            model_success = df.groupby(['Model', 'Vision'])['Result Status'].apply(
                lambda x: (x == 'Success').mean() * 100
            ).reset_index()
            
            ax = sns.barplot(x='Model', y='Result Status', hue='Vision', data=model_success, palette='Set2')
            plt.title('Success Rate by Model and Vision')
            plt.xlabel('Model')
            plt.ylabel('Success Rate (%)')
            plt.ylim(0, 100)
            plt.tight_layout()
            
            file_path = os.path.join(output_dir, '6_success_by_model_vision.png')
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            generated_files.append(file_path)
            plt.close()
            
            # 7. Bounds Rate by Model and Vision
            plt.figure(figsize=(12, 7))
            model_bounds = df.groupby(['Model', 'Vision'])['Bounds Status'].apply(
                lambda x: (x == 'Within Bounds').mean() * 100
            ).reset_index()
            
            ax = sns.barplot(x='Model', y='Bounds Status', hue='Vision', data=model_bounds, palette='Set2')
            plt.title('Within Bounds Rate by Model and Vision')
            plt.xlabel('Model')
            plt.ylabel('Within Bounds Rate (%)')
            plt.ylim(0, 100)
            plt.tight_layout()
            
            file_path = os.path.join(output_dir, '7_bounds_by_model_vision.png')
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            generated_files.append(file_path)
            plt.close()
            
            # NEW: Trajectory Length by Model and Vision (if available)
            if 'Trajectory Length' in df.columns:
                plt.figure(figsize=(12, 7))
                
                ax = sns.boxplot(x='Model', y='Trajectory Length', hue='Vision', data=df, palette='Set2')
                plt.title('Trajectory Length by Model and Vision')
                plt.xlabel('Model')
                plt.ylabel('Trajectory Length (steps)')
                plt.tight_layout()
                
                file_path = os.path.join(output_dir, '12_trajectory_length_by_model.png')
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
                generated_files.append(file_path)
                plt.close()
        
        # 8. Error Analysis (if we have data on error types)
        if 'error' in df.columns or 'Error' in df.columns:
            error_col = 'error' if 'error' in df.columns else 'Error'
            # Only analyze rows that have errors
            error_df = df[df['Result Status'] == 'Failure']
            
            if not error_df.empty and error_col in error_df.columns:
                plt.figure(figsize=(14, 8))
                error_counts = error_df[error_col].value_counts().head(10)  # Top 10 errors
                
                # Fix barplot with explicit hue assignment
                error_df_plot = pd.DataFrame({'Error': error_counts.index, 'Count': error_counts.values})
                ax = sns.barplot(x='Error', y='Count', data=error_df_plot, hue='Error', palette='Reds_r', legend=False)
                plt.title('Top 10 Error Types')
                plt.xlabel('Error Type')
                plt.ylabel('Count')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                file_path = os.path.join(output_dir, '8_error_analysis.png')
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
                generated_files.append(file_path)
                plt.close()
                
        # 9. Calculated vs Ground Truth Scatter Plot
        if 'Calculated Answer' in df.columns and 'Ground Truth' in df.columns:
            # Filter out missing values
            plot_df = df.dropna(subset=['Calculated Answer', 'Ground Truth']).copy()
            
            if not plot_df.empty:
                plt.figure(figsize=(10, 10))
                
                # Add color based on bounds
                colors = plot_df['Bounds Status'].map({
                    'Within Bounds': 'green',
                    'Outside Bounds': 'red',
                    'Unknown': 'gray'
                })
                
                plt.scatter(plot_df['Ground Truth'], plot_df['Calculated Answer'], 
                           alpha=0.7, c=colors, s=50)
                
                # Add a perfect prediction line
                min_val = min(plot_df['Ground Truth'].min(), plot_df['Calculated Answer'].min())
                max_val = max(plot_df['Ground Truth'].max(), plot_df['Calculated Answer'].max())
                plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect Prediction')
                
                plt.xlabel('Ground Truth')
                plt.ylabel('Calculated Answer')
                plt.title('Calculated vs Ground Truth Values')
                
                # Add legend for colors
                from matplotlib.lines import Line2D
                legend_elements = [
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Within Bounds'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Outside Bounds'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Unknown'),
                    Line2D([0], [0], color='k', linestyle='--', label='Perfect Prediction')
                ]
                plt.legend(handles=legend_elements, loc='best')
                
                plt.grid(True)
                plt.tight_layout()
                
                file_path = os.path.join(output_dir, '9_calculated_vs_truth.png')
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
                generated_files.append(file_path)
                plt.close()
                
                # 10. Error Distribution
                plt.figure(figsize=(12, 6))
                
                # Calculate percent error - fix the SettingWithCopyWarning
                plot_df.loc[:, 'Percent Error'] = abs(plot_df['Calculated Answer'] - plot_df['Ground Truth']) / plot_df['Ground Truth'] * 100
                
                # Plot histogram
                sns.histplot(data=plot_df, x='Percent Error', bins=20, kde=True)
                plt.title('Distribution of Calculation Error (%)')
                plt.xlabel('Absolute Percent Error')
                plt.ylabel('Frequency')
                plt.xlim(left=0)
                plt.axvline(x=plot_df['Percent Error'].median(), color='r', linestyle='--', 
                           label=f'Median ({plot_df["Percent Error"].median():.1f}%)')
                plt.legend()
                plt.tight_layout()
                
                file_path = os.path.join(output_dir, '10_error_distribution.png')
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
                generated_files.append(file_path)
                plt.close()
            
        # Create an HTML report that combines all visualizations
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Calculator Results Analysis</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
                h1 {{ color: #2c3e50; text-align: center; }}
                h2 {{ color: #3498db; margin-top: 30px; }}
                .dashboard {{ max-width: 1200px; margin: 0 auto; }}
                .visualization {{ margin-bottom: 40px; }}
                .visualization img {{ max-width: 100%; height: auto; border: 1px solid #ddd; box-shadow: 0px 0px 10px rgba(0,0,0,0.1); }}
                .summary {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 30px; }}
                .footer {{ text-align: center; margin-top: 50px; font-size: 0.8em; color: #7f8c8d; }}
            </style>
        </head>
        <body>
            <div class="dashboard">
                <h1>Calculator Results Analysis</h1>
                
                <div class="summary">
                    <h2>Summary Statistics</h2>
                    <p><strong>Total Calculator Runs:</strong> {len(df)}</p>
                    <p><strong>Success Rate:</strong> {(df['Result Status'] == 'Success').mean() * 100:.1f}%</p>
                    <p><strong>Within Bounds Rate:</strong> {(df['Bounds Status'] == 'Within Bounds').mean() * 100:.1f}%</p>
                    <p><strong>Number of Calculators:</strong> {df['Calculator Name'].nunique()}</p>
                </div>
        """
        
        # Add each visualization to the HTML
        for i, file_path in enumerate(generated_files):
            file_name = os.path.basename(file_path)
            html_content += f"""
                <div class="visualization">
                    <h2>Visualization {i+1}: {file_name.split('.')[0].replace('_', ' ').title().replace('Vs', 'vs')}</h2>
                    <img src="{file_name}" alt="{file_name.split('.')[0].replace('_', ' ').title()}">
                </div>
            """
        
        html_content += """
                <div class="footer">
                    <p>Generated by visualize_results.py</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Write the HTML file
        html_path = os.path.join(output_dir, 'visualization_report.html')
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        print(f"\nSuccessfully generated {len(generated_files)} visualizations!")
        print(f"HTML report saved to: {html_path}")
        print("\nGenerated visualizations:")
        for file in generated_files:
            print(f"- {file}")
            
    except Exception as e:
        print(f"Error creating visualizations: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    csv_path = "results/summary_20250430_213851.csv"
    
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    
    print(f"Loading data from: {csv_path}")
    df = load_data(csv_path)
    
    if df is not None:
        output_dir = create_output_dir()
        visualize_results(df, output_dir)
    else:
        print("Cannot proceed without valid data.") 