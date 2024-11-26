import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load CodeT5+ model and tokenizer
checkpoint = "Salesforce/codet5p-220m"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the tokenizer and model
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, trust_remote_code=True).to(device)

# Modify the test code to provide a very explicit translation instruction
test_code = """
def add(a, b):
    return a + b
"""

# Tokenize the input code
tokens = tokenizer(test_code, return_tensors="pt").to(device)
tokens['decoder_input_ids'] = tokens['input_ids'].clone()

# Generate the C++ code using beam search for better results
print("Generating C++ translation...")
output_tokens = model.generate(
    **tokens,
    max_length=150,
    num_beams=5,  # Beam search for more coherent output
    temperature=0.7,  # Controls randomness
    repetition_penalty=1.2  # Discourages repetition
)

# Decode the generated tokens to get the C++ code
translated_code = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

# Display the translated C++ code
print("\nTranslated C++ Code:\n", translated_code)
