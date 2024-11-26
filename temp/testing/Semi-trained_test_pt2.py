import torch
from transformers import RobertaTokenizer, T5ForConditionalGeneration

# Load the fine-tuned model and tokenizer
model_dir = r"./fine_tuned_model_continued"  # Path to your fine-tuned model directory
tokenizer = RobertaTokenizer.from_pretrained(model_dir)
model = T5ForConditionalGeneration.from_pretrained(model_dir)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Function to translate Python to C++
def translate_python_to_cpp(python_code):
    # Tokenize the input Python code
    input_ids = tokenizer(python_code, return_tensors="pt", max_length=512, truncation=True).input_ids.to(device)

    # Generate translated C++ code
    outputs = model.generate(input_ids, max_length=512, num_beams=4, early_stopping=True)

    # Decode the generated tokens to get the C++ code
    translated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_code

# Example Python code to translate
python_code = """
def add(a, b):
    return a + b
"""

# Translate the Python code to C++
cpp_code = translate_python_to_cpp(python_code)

# Print the translated C++ code
print("Python Code:")
print(python_code)
print("\nTranslated C++ Code:")
print(cpp_code)
