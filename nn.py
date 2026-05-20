import torch
import torch.nn as nn
import torch.optim as optim

# 1. Define the Network Architecture
class CustomerChurnModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CustomerChurnModel, self).__init__()
        
        # Fully Connected Layer 1: Inputs -> Hidden
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Non-linear activation
        self.relu = nn.ReLU()
        # Fully Connected Layer 2: Hidden -> Output (2 classes: Churn / No Churn)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # Pass input through first layer, then activate, then pass to output
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 2. Hyperparameters & Initialization
INPUT_FEATURES = 10  # e.g., account age, usage metrics, support tickets
HIDDEN_UNITS = 32
OUTPUT_CLASSES = 2   # Binary classification
LEARNING_RATE = 0.001

model = CustomerChurnModel(input_dim=INPUT_FEATURES, hidden_dim=HIDDEN_UNITS, output_dim=OUTPUT_CLASSES)

# 3. Define Loss Function and Optimizer
# CrossEntropyLoss expects raw logits (scores) because it applies Softmax internally
criterion = nn.CrossEntropyLoss() 
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 4. Mock Training Loop (What happens under the hood)
# Simulating a batch of 5 customers
mock_inputs = torch.randn(5, INPUT_FEATURES) 
mock_targets = torch.tensor([1, 0, 1, 1, 0])  # Actual ground truth labels

# Forward Pass
outputs = model(mock_inputs)
loss = criterion(outputs, mock_targets)

# Backward Pass (The calculus part)
optimizer.zero_grad() # Clear previous gradients to avoid accumulation
loss.backward()       # Compute gradients via backpropagation
optimizer.step()      # Update weights

print(f"Initial Mock Loss: {loss.item():.4f}")

# --- 5. Inference (Production Use) ---
# Imagine these are 3 new customers from the website today
new_customers_data = torch.tensor([
    [0.1, 0.5, 0.2, 0.8, 0.1, 0.3, 0.7, 0.2, 0.5, 0.1], # Low usage, high complaints (Likely Churn)
    [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9], # High usage, long tenure (Unlikely Churn)
    [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]  # Average
])

# Turn off gradient tracking (we are just predicting, not learning here)
model.eval()
with torch.no_grad():
    # Get raw scores
    logits = model(new_customers_data)
    # Convert to probabilities (e.g. 90% chance positive)
    probabilities = torch.softmax(logits, dim=1)

print("\n--- Live Prediction Results ---")
for i, probs in enumerate(probabilities):
    # probs[0] is class 0 (e.g. No Churn), probs[1] is class 1 (Churn)
    churn_probability = probs[1].item()
    
    if churn_probability > 0.6: # Business Rule: Flag if >60% confident
        risk_level = "⚠️ HIGH RISK"
    else:
        risk_level = "Low Risk"

    print(f"Customer {i+1}: Churn Probability = {churn_probability:.2%}" + 
          f" ({'Red Card' if churn_probability > 0.5 else 'Green Card'}) {risk_level}")