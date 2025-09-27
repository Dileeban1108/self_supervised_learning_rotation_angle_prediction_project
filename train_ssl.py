import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from models.rotation_net import RotationNet
from utils.data_utils import get_mnist_dataset, rotation_collate_fn

# 1. Dataset & Loader
train_data = get_mnist_dataset(train=True)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True, collate_fn=rotation_collate_fn)

# 2. Model, Loss, Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RotationNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. Training Loop
for epoch in range(5):
    total_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        logits, _ = model(imgs)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "rotation_ssl.pth")
print("✅ SSL training complete, model saved.")
