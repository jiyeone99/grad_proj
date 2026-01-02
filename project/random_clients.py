import argparse
import time
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformations for CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

class RandomClient(fl.client.NumPyClient):
    def __init__(self, model: torch.nn.Module, input_shape: torch.Size, device: torch.device, epochs: int):
        self.model = model
        self.input_shape = input_shape
        self.device = device
        self.epochs = epochs

    def get_parameters(self, config: Dict[str, str] = None) -> List[np.ndarray]:
        print("[INFO] Generating initial random parameters.")
        random_params = [
            np.random.uniform(-1, 1, size=param.size()).astype(np.float32)
            for param in self.model.parameters()
        ]
        return random_params

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
            
            # Simulate computation time for each epoch
            time.sleep(3)

        # Generate random metrics for the simulation
        metrics = {"loss": np.random.uniform(0, 1), "accuracy": np.random.uniform(0, 1)}
        return updated_params, len(updated_params), metrics

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[float, int, Dict[str, float]]:
        """Simulate evaluation with random loss."""
        print("[INFO] Simulating evaluation with random loss.")
        loss = np.random.uniform(0, 10)  # Random loss between 0 and 10
        return loss, 1, {}

from cifar import SimpleCNN

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition-id", type=int, required=True, help="Partition ID for the client")
    parser.add_argument("--num-clients", type=int, required=True, help="Total number of clients")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training")
    args = parser.parse_args()

    # Initialize random client
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = SimpleCNN().to(DEVICE)
    input_shape = next(model.parameters()).shape
    client = RandomClient(model=model, input_shape=input_shape, device=device, epochs=args.epochs)

    # Start the Flower client
    fl.client.start_numpy_client(
        server_address="192.168.0.40:8080",
        client=client.to_client()
    )

if __name__ == "__main__":
    main()
