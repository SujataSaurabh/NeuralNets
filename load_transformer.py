from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 1. Define the model stub (we'll use a lightweight, production-friendly DistilBERT)
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

# 2. Load the Tokenizer and the Model
# Tokenizer: Converts raw string text into numerical token IDs the model understands
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# 3. Raw client data
raw_texts = [
    "The new software update is incredibly fast, but the UI is a bit confusing.",
    "This platform deployment was an absolute disaster."
]

# 4. Preprocessing (Tokenization & Padding)
# padding=True ensures all sequences in the batch have the same length by adding 0s
# truncation=True cuts off text that exceeds the model's maximum context length (usually 512 tokens)
inputs = tokenizer(raw_texts, padding=True, truncation=True, return_tensors="pt")

print("--- Tokenizer Outputs ---")
print("Token IDs (input_ids):\n", inputs['input_ids'])
print("Attention Mask:\n", inputs['attention_mask']) # Tells the model which tokens to ignore (the padding)

# 5. Inference (Forward Pass)
# torch.no_grad() disables gradient calculation, reducing memory usage and speeding up inference
with torch.no_grad():
    outputs = model(**inputs)

# 6. Post-processing
# The model outputs raw scores called "logits". We apply Softmax to get probabilities.
logits = outputs.logits
probabilities = torch.softmax(logits, dim=-1)

print("\n--- Predictions ---")
for i, text in enumerate(raw_texts):
    prob_negative = probabilities[i][0].item()
    prob_positive = probabilities[i][1].item()
    print(f"Text: '{text}'")
    print(f"-> Positive: {prob_positive:.2%}, Negative: {prob_negative:.2%}\n")