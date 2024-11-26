import os
import json

dataset_path = "/workspace/ai_training_database/"
jsonl_files = [f for f in os.listdir(dataset_path) if f.endswith(".jsonl")]

for jsonl_file in jsonl_files:
    file_path = os.path.join(dataset_path, jsonl_file)
    with open(file_path, 'r') as f:
        lines = f.readlines()

    valid_lines = []
    for idx, line in enumerate(lines):
        try:
            json_obj = json.loads(line)
            valid_lines.append(json_obj)
        except json.JSONDecodeError as e:
            print(f"Error in file '{jsonl_file}', line {idx + 1}: {e}")

    # Save the valid lines back to a new file (optional)
    with open(os.path.join(dataset_path, f"fixed_{jsonl_file}"), 'w') as f:
        for obj in valid_lines:
            f.write(json.dumps(obj) + '\n')
