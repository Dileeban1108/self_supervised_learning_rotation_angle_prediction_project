import torch
from torch.utils.data import DataLoader, Subset
from torch import nn, optim
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from models.rotation_net import RotationNet
from utils.data_utils import get_mnist_dataset

# 1. Load pre-trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone = RotationNet()
backbone.load_state_dict(torch.load("rotation_ssl.pth", map_location=device))
backbone = backbone.to(device)
backbone.eval()

# 2. Prepare labeled subset (1000 samples only)
full_train = get_mnist_dataset(train=True)
subset = Subset(full_train, range(1000))
train_loader = DataLoader(subset, batch_size=64, shuffle=True)
test_loader = DataLoader(get_mnist_dataset(train=False), batch_size=64)

# 3. Extract features
def extract_features(loader):
    feats, labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            _, f = backbone(imgs)
            feats.append(f.cpu())
            labels.append(lbls)
    return torch.cat(feats).numpy(), torch.cat(labels).numpy()

X_train, y_train = extract_features(train_loader)
X_test, y_test = extract_features(test_loader)

# 4. Train Logistic Regression Classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"📊 Downstream Classification Accuracy: {acc*100:.2f}%")
