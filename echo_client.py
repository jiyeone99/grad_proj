import flwr as fl
import numpy as np
from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays


class EchoClient(fl.client.NumPyClient):
    def __init__(self, model, dataset_size):
        self.model = model
        self.dataset_size = dataset_size
        self.received_parameters = []  # To store received global parameters (FIFO)
        self.max_storage = 2  # To store only the last 2 rounds of parameters

    def get_parameters(self, config):
        """Generate random parameters for the initial rounds."""
        return [np.random.rand(*param.shape).astype(np.float32) for param in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        """Load parameters into the model."""
        state_dict = {k: np.array(v) for k, v in zip(self.model.state_dict().keys(), parameters)}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: Parameters, config):
        try:
            # Convert received parameters
            global_parameters = parameters_to_ndarrays(parameters)

            # Store received parameters
            self.received_parameters.append(global_parameters)
            if len(self.received_parameters) > self.max_storage:
                self.received_parameters.pop(0)  # Keep only the last two rounds

            # Determine parameters to send back
            if len(self.received_parameters) < self.max_storage:
                # Return random parameters for the first two rounds
                return_parameters = self.get_parameters(config)
            else:
                # Echo parameters received two rounds ago
                return_parameters = self.received_parameters[0]

            # Return parameters and metrics
            return ndarrays_to_parameters(return_parameters), self.dataset_size, {"accuracy": 0.0}

        except Exception as e:
            print(f"Error in fit: {e}")
            return ndarrays_to_parameters([]), 0, {"error": str(e)}

    def evaluate(self, parameters: Parameters, config):
        """Dummy evaluation."""
        return 0.0, self.dataset_size, {"accuracy": 0.0}


def main():
    # Initialize model (dummy model for parameter shapes)
    import torch.nn as nn

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            return x

    model = DummyModel()
    dataset_size = 1000  # Example dataset size

    # Print model parameter shapes
    print("Model initialized with layers:")
    for name, param in model.state_dict().items():
        print(f"{name}: {param.shape}")

    # Start the client
    client = EchoClient(model=model, dataset_size=dataset_size)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)


if __name__ == "__main__":
    main()
