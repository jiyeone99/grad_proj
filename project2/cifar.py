import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from flwr_datasets import FederatedDataset

def load_data(partition_id: int):
    """Load partitioned CIFAR10 data."""
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": 10})
    partition = fds.load_partition(partition_id)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    return trainloader, testloader

class SimpleCNN(nn.Module):
    """Simplified CNN for CIFAR-10."""
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(
    net: SimpleCNN,
    trainloader: DataLoader,
    epochs: int,
    device: torch.device,
):
    """Train the SimpleCNN model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    net.to(device)
    net.train()
    for epoch in range(epochs):
        for data in trainloader:
            images, labels = data["img"].to(device), data["label"].to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

def test(
    net: SimpleCNN,
    testloader: DataLoader,
    device: torch.device,
):
    """Test the SimpleCNN model."""
    criterion = nn.CrossEntropyLoss()
    correct, total_loss = 0, 0.0

    net.to(device)
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data["img"].to(device), data["label"].to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

    accuracy = correct / len(testloader.dataset)
    return total_loss, accuracy
