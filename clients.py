import argparse
import json
import os
from collections import OrderedDict
from typing import Dict, List, Tuple

import cifar
import flwr as fl
import numpy as np
import torch
from datasets.utils.logging import disable_progress_bar
from torch.utils.data import DataLoader

disable_progress_bar()

USE_FEDBN: bool = True
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CifarClient(fl.client.NumPyClient):
    """Flower client implementing CIFAR-10 image classification with partition tracking."""

    def __init__(self, model, trainloader, testloader, epochs, partition_id):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.epochs = epochs
        self.partition_id = partition_id  # Store the partition ID for tracking

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        self.model.train()
        if USE_FEDBN:
            return [
                val.cpu().numpy()
                for name, val in self.model.state_dict().items()
                if "bn" not in name
            ]
        else:
            return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        self.model.train()
        if USE_FEDBN:
            keys = [k for k in self.model.state_dict().keys() if "bn" not in k]
            params_dict = zip(keys, parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=False)
        else:
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[List[np.ndarray], int, Dict]:
        self.set_parameters(parameters)

        # Train the model
        for epoch in range(self.epochs):
            cifar.train(self.model, self.trainloader, epochs=1, device=DEVICE)

        # Evaluate the model
        accuracy, loss = cifar.test(self.model, self.testloader, device=DEVICE)
        metrics = {"accuracy": accuracy, "loss": loss, "partition_id": self.partition_id}

        return self.get_parameters(config={}), len(self.trainloader.dataset), metrics

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[float, int, Dict]:
        # Set model parameters and evaluate on local test data
        self.set_parameters(parameters)
        loss, accuracy = cifar.test(self.model, self.testloader, device=DEVICE)
        return float(loss), len(self.testloader.dataset), {"accuracy": float(accuracy)}

def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 FL Client with partition tracking")
    parser.add_argument("--partition-id", type=int, required=True, choices=range(0, 10), help="Client partition ID")
    parser.add_argument("--epochs", type=int, default=1, help="Number of local epochs per training round")
    args = parser.parse_args()

    # Load data
    trainloader, testloader = cifar.load_data(args.partition_id)

    # Load model
    model = cifar.Net().to(DEVICE).train()

    # Perform a single forward pass to initialize BatchNorm
    _ = model(next(iter(trainloader))["img"].to(DEVICE))

    # Start the client
    client = CifarClient(model, trainloader, testloader, epochs=args.epochs, partition_id=args.partition_id)
    fl.client.start_client(server_address="192.168.0.40:8080", client=client.to_client())

if __name__ == "__main__":
    main()