import csv
import os
import pandas as pd

def csv_to_latex_pandas(df, output_file, is_int = False):
    
    # Start LaTeX table
    latex_output = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Model Performance Comparison}",
        f"\\begin{{tabular}}{{l{'r' * (len(df.columns)-1)}}}",
        "\\hline"
    ]
    
    # Add header
    latex_header = " & ".join(df.columns) + " \\\\"
    latex_output.append(latex_header)
    latex_output.append("\\hline")
    
    # Process each row
    for _, row in df.iterrows():
        formatted_row = [str(row.iloc[0])]  # Model name (first column) as is
        
        # Format numerical values
        for value in row.iloc[1:]:
            if pd.isna(value):
                formatted_row.append('')
            # elif type(value) == str:
            #     formatted_row.append(value)
            elif is_int and type(value) != str:
                formatted_row.append(f"{value:.0f}")
            else:
                try:
                    value = float(value)
                    if abs(value) > 1000:
                        formatted_row.append(f"{value:.2e}")
                    else:
                        formatted_row.append(f"{value:.4f}")
                except:
                    formatted_row.append(value)
        
        # Join the row with & and add LaTeX line ending
        latex_row = " & ".join(formatted_row) + " \\\\"
        latex_output.append(latex_row)
    
    # End LaTeX table
    latex_output.extend([
        "\\hline",
        "\\end{tabular}",
        "\\label{tab:model-comparison}",
        "\\end{table}"
    ])
    
    # Write to output file
    with open(output_file, 'w') as file:
        file.write('\n'.join(latex_output))

# Example usage
sample_data = """Model,RMSE,MedAE,R2
Random Forest,0.0210,0.0072,0.8919
MLP,0.0217,0.0101,0.8848
XGBoost,0.0218,0.0105,0.8839
Bayesian Regression,155.1432,0.1377,-7403000"""

# Write sample data to a temporary CSV file
with open('sample.csv', 'w') as f:
    f.write(sample_data)

def convert_csv_to_latex(input_file, output_file):
    try:
        # Read the CSV file
        with open(input_file, 'r', newline='') as csvfile:
            csv_reader = csv.reader(csvfile)
            headers = next(csv_reader)  # Get the headers
            data = list(csv_reader)  # Get the rest of the data

        if not data:
            print("The CSV file is empty.")
            return

        # Start building the LaTeX table
        latex_code = "\\begin{table}[h]\n\\centering\n\\begin{tabular}{"
        latex_code += "|" + "c|" * len(headers) + "}\n\\hline\n"

        # Add headers
        latex_code += " & ".join(headers) + " \\\\ \\hline\n"

        # Add data rows
        for row in data:
            latex_code += " & ".join(row) + " \\\\ \\hline\n"

        # Close the LaTeX table
        latex_code += "\\end{tabular}\n\\caption{CSV Data}\n\\label{tab:csvdata}\n\\end{table}"

        # Write the LaTeX code to the output file
        with open(output_file, 'w') as texfile:
            texfile.write(latex_code)

        print(f"LaTeX code has been saved to {output_file}")

        # Display the first 500 characters of the LaTeX code
        print("Preview of the LaTeX code:")
        print(latex_code[:500] + "...")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Example usage
# input_csv_file = 'regressor_cmp.csv'
# output_latex_file = 'regressor_cmp.tex'

# # Create a sample CSV file for demonstration
# sample_csv_content = """Name,Age,City
# John Doe,30,New York
# Jane Smith,25,Los Angeles
# Bob Johnson,35,Chicago"""

# with open(input_csv_file, 'w', newline='') as csvfile:
#     csvfile.write(sample_csv_content)
# print(f"Sample CSV file '{input_csv_file}' has been created.")

# Convert CSV to LaTeX
# convert_csv_to_latex(input_csv_file, output_latex_file)

# # Clean up the sample CSV file
# os.remove(input_csv_file)
# print(f"Sample CSV file '{input_csv_file}' has been removed.")