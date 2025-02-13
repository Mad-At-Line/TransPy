import os
import json
from datasets import load_dataset
from tqdm import tqdm
from collections import Counter

# Define constants
OUTPUT_FOLDER = r"C:\Users\mythi\Desktop\TransPy\Cpp_dataset_bigstack"
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, "dataset.jsonl")

# Ensure output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load the dataset in streaming mode
print("Loading dataset...")
dataset = load_dataset("bigcode/the-stack", split="train", streaming=True)

# Initialize dictionaries for Python and C++ samples
all_python = {}
all_cpp = {}
language_counts = Counter()

print("Collecting examples...")

# Debug: Inspect a few dataset entries
print("Inspecting first few dataset entries for structure...")
for i, example in enumerate(dataset):
    if i >= 5:  # Limit to 5 samples
        break
    print(json.dumps(example, indent=4))  # Pretty print an example

print("Continuing with processing...")

for example in tqdm(dataset, desc="Processing examples"):
    if 'language' in example and 'content' in example:
        metadata = example.get('metadata', {})
        filename = metadata.get('filename')
        
        # Count languages for debugging
        language_counts[example['language']] += 1
        
        if not filename:
            continue

        if example['language'] == "Python":
            all_python[filename] = example['content']
        elif example['language'] == "C++":
            all_cpp[filename] = example['content']

# Debugging: Print counts
print("Language distribution:", language_counts)
print(f"Total Python files collected: {len(all_python)}")
print(f"Total C++ files collected: {len(all_cpp)}")

# Now match files
pairs = []
matched_files = set()
for f in all_python:
    if f in all_cpp:
        pairs.append({"input": all_python[f], "output": all_cpp[f]})
        matched_files.add(f)

# Debugging: Print some matched filenames
print("Matched Files Sample:", list(matched_files)[:10])

# Save pairs to a JSONL file
def save_to_jsonl(pairs):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for pair in pairs:
            json.dump(pair, f)
            f.write("\n")

print(f"Saving {len(pairs)} pairs to {OUTPUT_FILE}...")
save_to_jsonl(pairs)
print("Dataset successfully saved!")
