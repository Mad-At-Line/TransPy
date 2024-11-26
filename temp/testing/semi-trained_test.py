# This script will load the trained CodeT5 model and test its performance on new input examples.

import os
import torch
from transformers import RobertaTokenizer, T5ForConditionalGeneration

# Set paths
model_dir = r"C:\Users\mythi\OneDrive\Desktop\Important BTYSE Work\trained_model"
model_checkpoint = "Salesforce/codet5-base"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and trained model
tokenizer = RobertaTokenizer.from_pretrained(model_checkpoint)
model = T5ForConditionalGeneration.from_pretrained(model_dir).to(device)

# Function to translate Python code to C++ using the trained model
def translate_python_to_cpp(python_code):
    inputs = tokenizer(python_code, return_tensors="pt", max_length=512, truncation=True, padding="max_length").to(device)
    outputs = model.generate(**inputs, max_length=512, num_beams=5, early_stopping=True)
    cpp_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return cpp_code

# Test examples
python_examples = [
    "def add(a, b):\n    return a + b",
    "for i in range(10):\n    print(i)",
    "class Dog:\n    def __init__(self, name):\n        self.name = name\n    def bark(self):\n        print(f'{self.name} says woof!')"
]

# Translate each example and print the C++ output
for example in python_examples:
    print("Python Code:")
    print(example)
    cpp_translation = translate_python_to_cpp(example)
    print("\nTranslated C++ Code:")
    print(cpp_translation)
    print("\n" + "="*50 + "\n")