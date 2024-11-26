import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Paths to the trained model and tokenizer
model_dir = r"C:\Users\mythi\OneDrive\Desktop\Please_work_models"

# Load the trained model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)

# Example Python code to translate to C++
python_code = """
def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1

arr = [10, 23, 45, 70, 11, 15]
target = 70
index = linear_search(arr, target)
if index != -1:
    print(f"Element {target} found at index {index}")
else:
    print(f"Element {target} not found")
"""

# Prepare input for the model
prompt = f"translate Python to C++: {python_code}"
inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)

# Generate the output (translated C++ code)
print("Generating C++ translation...")
outputs = model.generate(**inputs, max_length=2048, num_beams=10, early_stopping=True)
translated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Print the translated C++ code
print("\nTranslated C++ Code:\n")
print(translated_code)
