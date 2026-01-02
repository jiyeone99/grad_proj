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

# Create directory to store parameters
SAVE_DIR = "local_parameters"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

PARAMETERS_FILE = os.path.join(SAVE_DIR, "all_parameters.json")

class CifarClient(fl.client.NumPyClient):
    """Flower client implementing CIFAR-10 image classification using PyTorch."""

    def __init__(
        self,
        model: cifar.Net,
        trainloader: DataLoader,
        testloader: DataLoader,
        epochs: int = 1,
    ) -> None:
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.epochs = epochs  # Number of epochs for training

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

    def save_parameters(self, parameters: List[np.ndarray], round_number: int) -> None:
        """Save the current parameters to a JSON file, consolidating all rounds."""
        try:
            # Load existing data from file, or start a new list if file doesn't exist
            if os.path.exists(PARAMETERS_FILE):
                with open(PARAMETERS_FILE, "r") as f:
                    all_parameters = json.load(f)
            else:
                all_parameters = []

            # Convert parameters to lists for JSON serialization and append round number
            parameters_list = {
                "round": round_number,
                "parameters": [param.tolist() for param in parameters]
            }
            all_parameters.append(parameters_list)

            # Save updated list of parameters back to file
            with open(PARAMETERS_FILE, "w") as f:
                json.dump(all_parameters, f)

            print(f"Parameters for round {round_number} saved to {PARAMETERS_FILE}")
        except Exception as e:
            print(f"Failed to save parameters: {e}")

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        # Set model parameters, train model, return updated model parameters
        self.set_parameters(parameters)

        # Determine number of epochs for training, with a default of the client's own value
        local_epochs = int(config.get("local_epochs", self.epochs))

        # Train model for the specified number of epochs
        cifar.train(self.model, self.trainloader, epochs=local_epochs, device=DEVICE)

        # Get the updated parameters after training
        updated_parameters = self.get_parameters(config={})

        # Save updated parameters after training
        round_number = int(config.get("round_number", 0))  # Add round number to the config
        self.save_parameters(updated_parameters, round_number)

        return updated_parameters, len(self.trainloader.dataset), {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)
        loss, accuracy = cifar.test(self.model, self.testloader, device=DEVICE)
        return float(loss), len(self.testloader.dataset), {"accuracy": float(accuracy)}

def main() -> None:
    """Load data, start CifarClient."""
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--partition-id", type=int, required=True, choices=range(0, 10))
    parser.add_argument("--round-number", type=int, default=0, help="Round number for training")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs for training")
    args = parser.parse_args()

    # Load data
    trainloader, testloader = cifar.load_data(args.partition_id)

    # Load model
    model = cifar.Net().to(DEVICE).train()

    # Perform a single forward pass to properly initialize BatchNorm
    _ = model(next(iter(trainloader))["img"].to(DEVICE))

    # Start client with specified number of epochs
    client = CifarClient(model, trainloader, testloader, epochs=args.epochs).to_client()
    fl.client.start_client(server_address="192.168.0.40:8080", client=client)

if __name__ == "__main__":
    main()
