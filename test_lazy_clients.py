import os
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim


# 간단한 CNN 모델 정의
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 데이터셋 나누는 함수
def split_dataset(dataset, num_clients, shuffle=True):
    num_samples = len(dataset)
    indices = list(range(num_samples))

    if shuffle:
        torch.manual_seed(75)  # Reproducibility
        indices = torch.randperm(num_samples).tolist()

    split_size = num_samples // num_clients
    split_indices = [indices[i * split_size:(i + 1) * split_size] for i in range(num_clients)]
    return split_indices


# 클라이언트별 데이터 로더 생성
def create_client_dataloaders(dataset, split_indices, batch_size):
    client_loaders = []
    for indices in split_indices:
        subset = Subset(dataset, indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        client_loaders.append(loader)
    return client_loaders


# 클라이언트 학습 및 파라미터 저장
def train_client(client_id, data_loader, model, criterion, optimizer, save_dir, num_epochs=1):
    model.train()
    for epoch in range(num_epochs):
        for images, labels in data_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Client {client_id}, Epoch {epoch+1}, Loss: {loss.item()}")

    # 파라미터 저장
    save_path = os.path.join(save_dir, f"client_{client_id}_params.npy")
    os.makedirs(save_dir, exist_ok=True)  # 디렉토리 생성
    parameters = {name: param.detach().numpy() for name, param in model.state_dict().items()}
    np.save(save_path, parameters)
    print(f"Client {client_id} parameters saved to {save_path}")


# CIFAR-10 데이터셋 로드
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)

# 데이터셋 나누기
num_clients = 8
split_indices = split_dataset(train_dataset, num_clients)

# 클라이언트별 데이터 로더 생성
batch_size = 32
client_loaders = create_client_dataloaders(train_dataset, split_indices, batch_size)

# 모델, 손실 함수, 옵티마이저 정의
save_dir = "./client_parameters15"
criterion = nn.CrossEntropyLoss()

# 클라이언트 학습
for client_id, data_loader in enumerate(client_loaders):
    print(f"Training Client {client_id}")
    model = SimpleCNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_client(client_id, data_loader, model, criterion, optimizer, save_dir, num_epochs=2)
