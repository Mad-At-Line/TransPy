import os
import glob
import json
import random

# Set your paths here
dataset_path = r"C:\Users\mythi\OneDrive\Desktop\Important BTYSE Work\Project_CodeNet\data"
input_output_path = r"C:\Users\mythi\OneDrive\Desktop\Important BTYSE Work\Project_CodeNet\derived\input_output\data"
output_path = r"C:\Users\mythi\OneDrive\Desktop\Important BTYSE Work\preprocessed_data"

# Create directories for processed data
os.makedirs(output_path, exist_ok=True)

# Pair Python and C++ files based on their index in each folder, assuming they solve the same problem
for folder in glob.glob(os.path.join(dataset_path, 'p*')):
    pairs = []
    py_files = sorted(glob.glob(os.path.join(folder, 'python', '*.py')))
    cpp_files = sorted(glob.glob(os.path.join(folder, 'C++', '*.cpp')))
    input_file = os.path.join(input_output_path, os.path.basename(folder), 'input.txt')
    output_file = os.path.join(input_output_path, os.path.basename(folder), 'output.txt')
    
    if os.path.exists(input_file) and os.path.exists(output_file):
        with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'r', encoding='utf-8') as outfile:
            input_data = infile.read()
            output_data = outfile.read()
            for i in range(min(len(py_files), len(cpp_files))):
                py_file = py_files[i]
                cpp_file = cpp_files[i]
                with open(py_file, 'r', encoding='utf-8') as py_fp:
                    py_code = py_fp.read()
                with open(cpp_file, 'r', encoding='utf-8') as cpp_fp:
                    cpp_code = cpp_fp.read()
                pairs.append({
                    "input": f"translate Python to C++: {py_code}\nGiven input: {input_data}\nExpected output: {output_data}",
                    "output": cpp_code
                })
                # Debug print to confirm files are being processed more sparingly
                if i % 2000 == 0:
                    print(f"Added {len(pairs)} pairs so far...")
    
    # Write pairs to a separate JSONL file for each problem folder
    if len(pairs) > 0:
        output_file_path = os.path.join(output_path, f"{os.path.basename(folder)}_pairs.jsonl")
        with open(output_file_path, "w", encoding='utf-8') as f:
            for pair in pairs:
                f.write(json.dumps(pair) + "\n")

# If not enough pairs are found, generate synthetic examples
if len(pairs) < 500:
    synthetic_pairs = [
        {"input": "translate Python to C++: def add(a, b):\n    return a + b", "output": "int add(int a, int b) {\n    return a + b;\n}"},
        {"input": "translate Python to C++: print('Hello, World!')", "output": "#include <iostream>\nint main() {\n    std::cout << \"Hello, World!\" << std::endl;\n    return 0;\n}"},
        # Add more synthetic pairs as needed
    ]
    synthetic_output_file_path = os.path.join(output_path, "synthetic_pairs.jsonl")
    with open(synthetic_output_file_path, "w", encoding='utf-8') as f:
        for pair in synthetic_pairs:
            f.write(json.dumps(pair) + "\n")

# Final confirmation
print(f"Data preprocessing completed. JSONL files have been created in: {output_path}")