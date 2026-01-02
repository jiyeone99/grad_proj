import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.cluster import KMeans
import joblib
import random

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Simple CNN model for CIFAR-10
class SimpleCNN(nn.Module):
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

# DAGMM Model with Improvements
class DAGMM(nn.Module):
    def __init__(self, input_dim, latent_dim=10):
        super(DAGMM, self).__init__()

        # Encoder: Deeper Network with BatchNorm and Dropout
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.2),
            nn.Linear(128, latent_dim)
        )

        # Decoder: Deeper Network with BatchNorm and Dropout
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.2),
            nn.Linear(128, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.2),
            nn.Linear(256, input_dim)
        )

        # Energy-based Estimation Layer
        self.estimation = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.Tanh(),
            nn.Linear(32, 16), nn.Tanh(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        energy = self.estimation(latent)
        return latent, reconstructed, energy

    def compute_loss(self, x, reconstructed, energy):
        recon_loss = F.mse_loss(reconstructed, x, reduction="mean")
        gmm_loss = energy.mean()
        total_loss = recon_loss + 0.1 * gmm_loss  # Adjust weight scaling
        return total_loss

# Load dataset and split into clients
def load_data(num_clients=10):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    num_samples = len(dataset)
    indices = list(range(num_samples))
    np.random.shuffle(indices)
    split_size = num_samples // num_clients
    return [Subset(dataset, indices[i * split_size:(i + 1) * split_size]) for i in range(num_clients)]

# Custom FL Client classes
class FederatedClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, epochs=10):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs
    
    def get_parameters(self, config):
        return [param.cpu().detach().numpy() for param in self.model.parameters()]
    
    def set_parameters(self, parameters):
        for param, new_param in zip(self.model.parameters(), parameters):
            param.data = torch.tensor(new_param, dtype=torch.float32, device=device)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        for _ in range(self.epochs):
            for images, labels in self.train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                loss.backward()
                optimizer.step()
        return self.get_parameters(config), len(self.train_loader.dataset), {}
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        accuracy = 100.0 * correct / total
        return float(1 - accuracy / 100), total, {"accuracy": accuracy}

from typing import Dict, List, Tuple

# Define malicious clients
class RandomClient(FederatedClient):
    def fit(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[List[np.ndarray], int, Dict[str, float]]:
        """Simulate training by generating random updates."""
        print("[INFO] Simulating training with random updates.")
        updated_params = [np.random.rand(*param.shape).astype(np.float32) for param in parameters]

        for epoch in range(self.epochs):
            print(f"[DEBUG] Random Client Epoch {epoch + 1}/{self.epochs}")
            
            # Check and fix NaN values in each parameter
            for i, param in enumerate(updated_params):
                param_tensor = torch.tensor(param)
                if torch.isnan(param_tensor).any():
                    print(f"[ERROR] Found NaN values in parameter {i}. Replacing with zeros.")
                    param_tensor = torch.nan_to_num(param_tensor)
                    updated_params[i] = param_tensor.numpy()  # Convert back to numpy

        # Generate random metrics for the simulation
        metrics = {"loss": np.random.uniform(0, 1), "accuracy": np.random.uniform(0, 1)}
        return updated_params, len(updated_params), metrics

# Dataset loading functions
def split_dataset(dataset, num_clients):
    num_samples = len(dataset)
    indices = list(range(num_samples))
    np.random.seed(57)
    np.random.shuffle(indices)

    split_size = num_samples // num_clients
    split_indices = [indices[i * split_size: (i + 1) * split_size] for i in range(num_clients)]
    return [Subset(dataset, idx) for idx in split_indices]

def create_limited_dataloader(partition, batch_size=32):
    train_size = int(0.4 * len(partition))  # 기존 80% → 40%
    test_size = len(partition) - train_size  # 기존 20% → 10%
    
    train_partition, test_partition = torch.utils.data.random_split(partition, [train_size, test_size])
    trainloader = DataLoader(train_partition, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_partition, batch_size=batch_size, shuffle=False)
    
    return trainloader, testloader

class SmallClient(FederatedClient):
    def __init__(self, model, partition):
        train_loader, test_loader = create_limited_dataloader(partition)
        super().__init__(model, train_loader, test_loader, epochs=10)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        for _ in range(self.epochs):
            for images, labels in self.train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                loss.backward()
                optimizer.step()
        return self.get_parameters(config), len(self.train_loader.dataset), {}

class LazyClient(FederatedClient):
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        for _ in range(2):
            for images, labels in self.train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                loss.backward()
                optimizer.step()
        return self.get_parameters(config), len(self.train_loader.dataset), {}

class EchoClient(FederatedClient):
    def fit(self, parameters, config):
        """Always return the received server parameters without modification."""
        print("[EchoClient] Returning received parameters unchanged")
        return parameters, len(self.train_loader.dataset), {}

import torch.optim as optim

def train_dagmm_supervised(data, labels, input_dim, num_classes=6, epochs=100, lr=0.001):
    model = DAGMM(input_dim).to(device)
    classifier = nn.Linear(model.encoder[-1].out_features, num_classes).to(device)
    optimizer = optim.AdamW(list(model.parameters()) + list(classifier.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)  # Reduce LR every 30 epochs
    criterion = nn.CrossEntropyLoss()
    
    data_tensor = torch.tensor(data, dtype=torch.float32, device=device)
    label_tensor = torch.tensor(labels, dtype=torch.long, device=device)

    for epoch in range(epochs):
        optimizer.zero_grad()
        latent, reconstructed, energy = model(data_tensor)
        outputs = classifier(latent)
        loss = criterion(outputs, label_tensor) + model.compute_loss(data_tensor, reconstructed, energy)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == label_tensor).sum().item() / len(labels) * 100
            print(f"[INFO] DAGMM Epoch {epoch+1}/{epochs} - Loss: {loss.item():.6f}, Accuracy: {accuracy:.2f}%")

    return model, classifier

# Train KMeans with PCA & Gaussian Mixture Option
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
def train_kmeans(latent_representations, n_clusters=6, use_gmm=False):
    reduced_data = PCA(n_components=min(20, latent_representations.shape[1])).fit_transform(latent_representations)
    if use_gmm:
        model = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=42)
    else:
        model = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42)
    model.fit(reduced_data)
    return model

# Evaluate KMeans using True Labels
def evaluate_kmeans_supervised(kmeans_model, data, labels):
    predictions = kmeans_model.predict(data)
    accuracy = (predictions == labels).sum() / len(labels) * 100
    print(f"[INFO] K-Means Supervised Accuracy: {accuracy:.2f}%")
    return accuracy


from flwr.common import parameters_to_ndarrays

def classify_client(parameters, dagmm_model, kmeans_model):
    # Convert Parameters object to a list of NumPy arrays
    parameter_list = parameters_to_ndarrays(parameters)

    # Flatten and concatenate all parameter arrays
    numeric_parameters = np.concatenate([p.flatten() for p in parameter_list])

    params_tensor = torch.tensor(numeric_parameters, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        latent, _ = dagmm_model(params_tensor)
    cluster = kmeans_model.predict(latent.cpu().numpy())[0]
    
    labels = {0: "normal", 1: "lazy", 2: "noised", 3: "random", 4: "server", 5: "echo"}
    return labels.get(cluster, "unknown")

# Custom FedAvg strategy
import os

class CustomFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, load_saved_models=True):  # 🔹 모델을 로드할지 여부 추가
        super().__init__()
        self.client_embeddings = []  
        self.client_labels = []  
        self.input_dim = None  

        self.dagmm_model_path = "dagmm_tester.pth"
        self.kmeans_model_path = "kmeans_tester.pkl"

        # 기존 모델 불러오기
        if load_saved_models and os.path.exists(self.dagmm_model_path) and os.path.exists(self.kmeans_model_path):
            print("[INFO] Loading existing DAGMM and K-Means models...")
            
            self.dagmm_model = DAGMM(268650)  # 입력 차원 확인 후 수정 필요
            self.dagmm_model.load_state_dict(torch.load(self.dagmm_model_path, map_location=device))
            self.dagmm_model.to(device)
            self.dagmm_model.eval()

            self.kmeans_model = joblib.load(self.kmeans_model_path)
            print("[INFO] DAGMM and K-Means models loaded successfully.")
        else:
            self.dagmm_model = None
            self.kmeans_model = None
            print("[INFO] No existing models found. Training new models.")


    def aggregate_fit(self, rnd, results, failures):
        print(f"[INFO] Aggregating results for round {rnd}...")

        for client, fit_res in results:
            params = parameters_to_ndarrays(fit_res.parameters)
            numeric_params = np.concatenate([p.flatten() for p in params])

            if self.input_dim is None:
                self.input_dim = numeric_params.shape[0]
                print(f"[INFO] Set DAGMM input_dim to {self.input_dim}")

            self.client_embeddings.append(numeric_params)
            self.client_labels.append(random.randint(0, 5))  

        if len(self.client_embeddings) >= 10:
            print("[INFO] Training DAGMM and K-Means models...")

            # 지도학습 DAGMM 학습
            dagmm_model, classifier = train_dagmm_supervised(
                np.array(self.client_embeddings),
                np.array(self.client_labels),
                self.input_dim
            )

            # 학습된 모델 저장
            torch.save(dagmm_model.state_dict(), self.dagmm_model_path)
            print("[INFO] DAGMM Supervised model saved.")

            # K-Means 학습 및 평가
            latent_representations = dagmm_model.encoder(
                torch.tensor(self.client_embeddings, dtype=torch.float32, device=device)
            ).detach().cpu().numpy()

            kmeans_model = train_kmeans(latent_representations, n_clusters=6)
            kmeans_accuracy = evaluate_kmeans_supervised(kmeans_model, latent_representations, self.client_labels)
            joblib.dump(kmeans_model, self.kmeans_model_path)
            print("[INFO] K-Means Supervised model saved.")

            self.dagmm_model = dagmm_model  # 새 모델 저장
            self.kmeans_model = kmeans_model

        return super().aggregate_fit(rnd, results, failures)

def assign_fixed_clients(partitions):
    """고정된 비율로 클라이언트 유형을 설정하여 학습"""
    client_distribution = [
        ("normal", 7),  # 정상 클라이언트 6명
        ("lazy", 1),  # 게으른 클라이언트 2명
        ("random", 1),  # 랜덤 업데이트 클라이언트 1명
        ("echo", 1)  # 에코 클라이언트 1명
    ]

    clients = []
    client_mapping = {
        "normal": FederatedClient,
        "lazy": LazyClient,
        "random": RandomClient,
        "echo": EchoClient,
    }

    partition_idx = 0
    for client_type, num in client_distribution:
        for _ in range(num):
            model = SimpleCNN()
            train_loader = DataLoader(partitions[partition_idx], batch_size=32, shuffle=True)
            test_loader = DataLoader(partitions[partition_idx], batch_size=32, shuffle=False)
            clients.append(client_mapping[client_type](model, train_loader, test_loader))
            partition_idx += 1

    return clients

# Start FL learning
def start_federated_learning():
    num_clients = 10
    partitions = load_data(num_clients)

    clients = assign_fixed_clients(partitions)

    # 🔹 기존 모델 로드 가능하도록 CustomFedAvg 설정
    fl.simulation.start_simulation(
        client_fn=lambda cid: clients[int(cid)].to_client(),
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=40),
        strategy=CustomFedAvg(load_saved_models=True)  # 🔹 저장된 모델 불러오기 활성화
    )


if __name__ == "__main__":
    start_federated_learning()