import torch

# Use your GTX 1060
device = torch.device("cuda")
print(f"Training on: {torch.cuda.get_device_name(0)}")

# Create simple data: AI learns that 2*x = y
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]], device=device)
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]], device=device)

# Create tiny neural network
model = torch.nn.Linear(1, 1).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss_fn = torch.nn.MSELoss()

print("\nTraining...")
for i in range(100):
    pred = model(x)
    loss = loss_fn(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 20 == 0:
        print(f"Step {i}: loss = {loss.item():.4f}")

print(f"\nDONE! AI learned: {x[2].item()} * {model.weight.item():.2f} = {model(x[2]).item():.2f}")
print("It learned 2*x = y!")