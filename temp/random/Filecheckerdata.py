import os
import json

output_path = r"C:\Users\mythi\OneDrive\Desktop\Important BTYSE Work\preprocessed_data"

# Check a few files from the output directory
for i, file_name in enumerate(os.listdir(output_path)):
    if file_name.endswith('.jsonl'):
        file_path = os.path.join(output_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            # Print the first few lines of each file
            for j, line in enumerate(f):
                print(json.loads(line))
                if j >= 2:  # Print only the first 3 lines per file for validation
                    break
        if i >= 2:  # Check only the first 3 files
            break