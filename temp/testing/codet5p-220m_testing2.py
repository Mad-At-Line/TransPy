import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Paths to the trained model and tokenizer
model_dir = r"C:\Users\mythi\OneDrive\Desktop\Please_work_models\Model_2"

# Load the trained model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)

# Example Python code to translate to C++
python_code = """
def matrix_multiplication(A, B):
    # Ensure that A and B can be multiplied
    if len(A[0]) != len(B):
        raise ValueError("Matrices cannot be multiplied due to incompatible dimensions.")

    # Initialize result matrix with zeros
    result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]

    # Perform matrix multiplication
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]

    return result

# Example usage:
A = [[1, 2, 3], [4, 5, 6]]
B = [[7, 8], [9, 10], [11, 12]]
result = matrix_multiplication(A, B)
for row in result:
    print(row)
"""

# Prepare input for the model
prompt = f"translate Python to C++: {python_code}"
inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(device)

# Generate the output (translated C++ code)
print("Generating C++ translation...")
outputs = model.generate(**inputs, max_length=1024, num_beams=10, early_stopping=False, do_sample=True, temperature=0.7)
translated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Print the translated C++ code
print("\nTranslated C++ Code:\n")
print(translated_code)
