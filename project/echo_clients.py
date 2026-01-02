import argparse
from typing import Dict, List, Tuple
import flwr as fl
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from cifar import SimpleCNN
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformations for CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# CIFAR-10 dataset
full_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)

# Split dataset into partitions
def split_dataset(dataset, num_clients):
    num_samples = len(dataset)
    indices = list(range(num_samples))
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(indices)

    split_size = num_samples // num_clients
    split_indices = [
        indices[i * split_size: (i + 1) * split_size] for i in range(num_clients)
    ]
    return [Subset(dataset, idx) for idx in split_indices]

# Create DataLoader for each partition
def create_dataloader(partition, batch_size=32):
    train_size = int(0.8 * len(partition))
    test_size = len(partition) - train_size
    train_partition, test_partition = torch.utils.data.random_split(partition, [train_size, test_size])
    
    trainloader = DataLoader(train_partition, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_partition, batch_size=batch_size, shuffle=False)
    return trainloader, testloader

# Echo Client
class EchoClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, testloader, epochs):
        self.model = model.to(DEVICE)
        self.trainloader = trainloader
        self.testloader = testloader
        self.epochs = epochs
        self.previous_parameters = []  # 서버에서 받은 파라미터 저장 (최대 2개)

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[List[np.ndarray], int, Dict]:
        # 현재 받은 파라미터를 저장 (최대 2개까지만 유지)
        if len(self.previous_parameters) == 2:
            self.previous_parameters.pop(0)  # 가장 오래된 값 삭제
        self.previous_parameters.append(parameters)  # 새 파라미터 추가

        # n-2 번째 라운드 파라미터 반환 (최대 2개 저장하므로 인덱스 0)
        if len(self.previous_parameters) < 2:
            # 초기 2라운드에서는 랜덤한 파라미터 반환
            print("[EchoClient] 초기 라운드, 랜덤 파라미터 전송")
            return [np.random.rand(*p.shape).astype(np.float32) for p in parameters], len(self.trainloader.dataset), {}
        
        # Simulate computation time for each epoch
        time.sleep(3)
        print("[EchoClient] 이전 n-2 라운드 파라미터 전송")
        return self.previous_parameters[0], len(self.trainloader.dataset), {}

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[float, int, Dict]:
        self.set_parameters(parameters)
        self.model.eval()
        loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in self.testloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = self.model(images)
                loss += torch.nn.CrossEntropyLoss()(outputs, labels).item() * images.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

        accuracy = 100.0 * correct / total
        loss /= total
        print(f"[EchoClient] Evaluation - Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")
        return loss, total, {"accuracy": accuracy}

# 실행 코드
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition-id", type=int, required=True, help="Partition ID for the client")
    parser.add_argument("--num-clients", type=int, required=True, help="Total number of clients")
    parser.add_argument("--epochs", type=int, default=10, help="Number of local epochs")
    args = parser.parse_args()

    # 데이터셋을 클라이언트 수만큼 분할
    partitions = split_dataset(full_dataset, args.num_clients)
    trainloader, testloader = create_dataloader(partitions[args.partition_id])

    # 모델 생성 및 EchoClient 초기화
    model = SimpleCNN().to(DEVICE)
    client = EchoClient(model, trainloader, testloader, args.epochs)

    # Flower 클라이언트 시작
    fl.client.start_numpy_client(
        server_address="192.168.0.40:8080",
        client=client.to_client()
    )

if __name__ == "__main__":
    main()
