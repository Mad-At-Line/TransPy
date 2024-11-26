import torch
from transformers import RobertaTokenizer, T5ForConditionalGeneration

# Load tokenizer and trained model
model_dir = r"C:\Users\mythi\OneDrive\Desktop\BTYSTE 2025 USE ME\fine_tuned_model_continued"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = RobertaTokenizer.from_pretrained(model_dir)
model = T5ForConditionalGeneration.from_pretrained(model_dir).to(device)

# Test input Python code
test_code = """
def add(a, b):
    return a + b
"""

# Tokenize input
tokens = tokenizer(test_code, return_tensors="pt").to(device)

# Generate translation with model using beam search for better results
output_tokens = model.generate(
    **tokens,
    max_length=150,
    num_beams=5,  # Beam search for more coherent output
    temperature=0.7,  # Controls randomness
    repetition_penalty=1.2  # Discourages repetition
)

# Decode the generated tokens to get C++ code
translated_code = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

# Display the translation
print("Translated C++ Code:\n", translated_code)