import random
import torch
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF

def get_mnist_dataset(train=True, download=True):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root="./data", train=train, download=download, transform=transform)
    return dataset

def rotate_image(img):
    angles = [0, 90, 180, 270]
    angle = random.choice(angles)
    return TF.rotate(img, angle), angles.index(angle)

def rotation_collate_fn(batch):
    rotated, labels = [], []
    for img, _ in batch:  # ignore original labels
        r, l = rotate_image(img)
        rotated.append(r)
        labels.append(l)
    rotated = torch.stack(rotated)
    labels = torch.tensor(labels)
    return rotated, labels
