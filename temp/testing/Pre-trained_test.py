# This script is used to test the initial, untrained CodeT5 model to serve as a control.
# It will evaluate the model's ability to translate Python to C++ before training.

import torch
from transformers import RobertaTokenizer, T5ForConditionalGeneration

# Set model checkpoint and device
model_checkpoint = "Salesforce/codet5-base"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained(model_checkpoint)
model = T5ForConditionalGeneration.from_pretrained(model_checkpoint).to(device)

# Test examples for Python to C++ translation
test_examples = [
    "Please translate the following Python code to C++: def add(a, b):\n    return a + b",
    "Please translate the following Python code to C++: print('Hello, World!')",
    "Please translate the following Python code to C++: for i in range(5):\n    print(i)",
]

# Function to generate C++ code from Python
def translate_python_to_cpp(python_code):
    inputs = tokenizer(python_code, return_tensors="pt").to(device)
    output_sequences = model.generate(**inputs, max_length=512, num_beams=5, early_stopping=True)
    translated_code = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    return translated_code

# Run test examples
for example in test_examples:
    translated_cpp = translate_python_to_cpp(example)
    print(f"\nPython Code:\n{example}\n\nGenerated C++ Code:\n{translated_cpp}\n")