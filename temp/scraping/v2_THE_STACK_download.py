import os
import json
from datasets import load_dataset
from tqdm import tqdm

# Define constants
OUTPUT_FOLDER = r"C:\Users\mythi\Desktop\TransPy\Cpp_dataset_bigstack"
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, "dataset.jsonl")

# Ensure output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load the dataset in streaming mode
print("Loading dataset...")
dataset = load_dataset("bigcode/the-stack", split="train", streaming=True)

# Initialize empty list for Python-C++ pairs
pairs = []

# Number of pairs to extract
MAX_PAIRS = 100

# Function to filter and pair Python and C++ examples
def extract_pairs(dataset, max_pairs):
    python_samples = {}
    cpp_samples = {}

    for example in tqdm(dataset, desc="Processing examples"):
        # Ensure the example has necessary keys
        if 'language' in example and 'content' in example:
            code = example['content']
            lang = example['language']

            # Collect Python samples
            if lang == "Python" and example['metadata']['filename'] not in python_samples:
                python_samples[example['metadata']['filename']] = code

            # Collect C++ samples
            elif lang == "C++" and example['metadata']['filename'] not in cpp_samples:
                cpp_samples[example['metadata']['filename']] = code

            # Check for matching filenames to create pairs
            if (example['metadata']['filename'] in python_samples and
                example['metadata']['filename'] in cpp_samples):

                pairs.append({
                    "input": python_samples[example['metadata']['filename']],
                    "output": cpp_samples[example['metadata']['filename']]
                })

                # Stop if we have enough pairs
                if len(pairs) >= max_pairs:
                    break

    return pairs

print("Filtering and pairing Python-C++ examples...")
# Extract Python-C++ pairs
pairs = extract_pairs(dataset, MAX_PAIRS)

# Save pairs to a JSONL file
def save_to_jsonl(pairs):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for pair in pairs:
            json.dump(pair, f)
            f.write("\n")

print(f"Saving {len(pairs)} pairs to {OUTPUT_FILE}...")
save_to_jsonl(pairs)
print("Dataset successfully saved!")
