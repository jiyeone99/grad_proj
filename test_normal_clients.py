import os
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim


# Same SimpleCNN as FL environment
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

# # 데이터셋 나누는 함수
# def split_dataset(dataset, num_clients, shuffle=True):
#     num_samples = len(dataset)
#     indices = list(range(num_samples))

#     if shuffle:
#         torch.manual_seed(65)  # Reproducibility
#         indices = torch.randperm(num_samples).tolist()

#     split_size = num_samples // num_clients
#     split_indices = [indices[i * split_size:(i + 1) * split_size] for i in range(num_clients)]
#     return split_indices

def split_dataset(dataset, num_clients, small_client_id=None, small_fraction=0.5, shuffle=True):
    """
    Split dataset among clients, optionally reducing the size for one client.

    Args:
        dataset: PyTorch dataset to split.
        num_clients: Total number of clients.
        small_client_id: ID of the client with a smaller dataset.
        small_fraction: Fraction of the dataset for the small client.
        shuffle: Whether to shuffle the dataset before splitting.

    Returns:
        List of indices for each client's dataset.
    """
    num_samples = len(dataset)
    indices = list(range(num_samples))

    if shuffle:
        torch.manual_seed(69)  # Reproducibility
        indices = torch.randperm(num_samples).tolist()

    split_size = num_samples // num_clients
    split_indices = [indices[i * split_size:(i + 1) * split_size] for i in range(num_clients)]

    # Adjust the size for the small client
    if small_client_id is not None:
        small_split_size = int(split_size * small_fraction)
        split_indices[small_client_id] = split_indices[small_client_id][:small_split_size]

    return split_indices


# 클라이언트별 데이터 로더 생성
def create_client_dataloaders(dataset, split_indices, batch_size):
    client_loaders = []
    for indices in split_indices:
        subset = Subset(dataset, indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        client_loaders.append(loader)
    return client_loaders

# Update parameters saving method
def save_parameters(model, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Parameters saved to {save_path}")


# Simulate FL training
# def federated_training(client_loaders, model, criterion, optimizer, num_rounds, save_dir, epochs=1):
#     """
#     Federated training simulation with multiple epochs per round.

#     Args:
#         client_loaders (list): List of DataLoader for each client.
#         model (nn.Module): Global model shared across clients.
#         criterion: Loss function.
#         optimizer: Optimizer for training.
#         num_rounds (int): Number of federated rounds.
#         save_dir (str): Directory to save models.
#         epochs (int): Number of epochs per round for each client.
#     """
#     global_model = model
#     os.makedirs(save_dir, exist_ok=True)  # Ensure save directory exists

#     for round_num in range(num_rounds):
#         print(f"[Round {round_num+1}]")
        
#         # Save global model at the beginning of each round
#         # global_save_path = os.path.join(save_dir, f"global_model_round_{round_num}.pth")
#         # save_parameters(global_model, global_save_path)

#         for client_id, data_loader in enumerate(client_loaders):
#             print(f"Training Client {client_id}")

#             # Initialize local model with global model parameters
#             local_model = SimpleCNN()
#             local_model.load_state_dict(global_model.state_dict())  # Start with global model
#             local_model.train()

#             # Train locally for `epochs`
#             for epoch in range(epochs):
#                 print(f"Epoch {epoch+1}/{epochs} for Client {client_id}")
#                 for images, labels in data_loader:
#                     optimizer.zero_grad()
#                     outputs = local_model(images)
#                     loss = criterion(outputs, labels)
#                     loss.backward()
#                     optimizer.step()

#             # Save client model
#             client_save_path = os.path.join(save_dir, f"lazy_client_{client_id}_round_{round_num+110}.pth")
#             save_parameters(local_model, client_save_path)

#         # Aggregate local models (simple average for demonstration purposes)
#         global_state_dict = global_model.state_dict()
#         num_clients = len(client_loaders)
#         for key in global_state_dict.keys():
#             global_state_dict[key] = torch.stack(
#                 [torch.load(os.path.join(save_dir, f"lazy_client_{client_id}_round_{round_num+110}.pth"))[key] for client_id in range(num_clients)]
#             ).mean(dim=0)
        
#         global_model.load_state_dict(global_state_dict)
#         print(f"[Round {round_num+1} Completed]")

import os
import torch
import numpy as np

def federated_training(
    client_loaders,
    model,
    criterion,
    optimizer,
    num_rounds,
    save_dir,
    epochs=10,
    lazy_client_id=None,
    lazy_epochs=3
):
    """
    Federated training simulation with a lazy client having fewer epochs.

    Args:
        client_loaders (list): List of DataLoader for each client.
        model (nn.Module): Global model shared across clients.
        criterion: Loss function.
        optimizer: Optimizer for training.
        num_rounds (int): Number of federated rounds.
        save_dir (str): Directory to save lazy client parameters.
        epochs (int): Number of epochs per round for normal clients.
        lazy_client_id (int): ID of the lazy client with fewer epochs.
        lazy_epochs (int): Number of epochs for the lazy client.
    """
    global_model = model
    os.makedirs(save_dir, exist_ok=True)  # Ensure save directory exists

    for round_num in range(num_rounds):
        print(f"[Round {round_num+1}]")

        local_updates = []
        for client_id, data_loader in enumerate(client_loaders):
            print(f"Training Client {client_id}")

            # Initialize local model with global model parameters
            local_model = SimpleCNN()
            local_model.load_state_dict(global_model.state_dict())  # Start with global model
            local_model.train()

            # Determine number of epochs for the client
            client_epochs = lazy_epochs if client_id == lazy_client_id else epochs

            # Train locally for `client_epochs`
            for epoch in range(client_epochs):
                print(f"Epoch {epoch+1}/{client_epochs} for Client {client_id}")
                for images, labels in data_loader:
                    optimizer.zero_grad()
                    outputs = local_model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

            # Save lazy client's parameters
            if client_id == lazy_client_id:
                lazy_save_path = os.path.join(save_dir, f"lazy_client_{client_id}_round_{round_num+90}.pth")
                torch.save(local_model.state_dict(), lazy_save_path)
                print(f"Lazy client {client_id}'s parameters saved to {lazy_save_path}")

            local_updates.append(local_model.state_dict())

        # Aggregate local updates (simple average for demonstration purposes)
        global_state_dict = global_model.state_dict()
        num_clients = len(client_loaders)
        for key in global_state_dict.keys():
            global_state_dict[key] = torch.stack([local_update[key] for local_update in local_updates]).mean(dim=0)

        global_model.load_state_dict(global_state_dict)
        print(f"[Round {round_num+1} Completed]")

# Load CIFAR-10 dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)

# 데이터셋 나누기
num_clients = 8
split_indices = split_dataset(train_dataset, num_clients)
batch_size = 32
client_loaders = create_client_dataloaders(train_dataset, split_indices, batch_size)

# Define model, loss, and optimizer
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Perform federated training
save_dir = "./worthless_parameters"
# Perform federated training with 3 epochs per round
# Define malicious client and noise level
# noise_level = 0.1  # Add 10% noise to parameters

# 연합 학습 실행
save_dir = "./fl_parameters2/lazy_client_parameters"
federated_training(
    client_loaders=client_loaders,
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    num_rounds=10,
    save_dir=save_dir,
    epochs=10,
    lazy_client_id=2,  # 클라이언트 2는 게으른 클라이언트로 설정
    lazy_epochs=3
)

