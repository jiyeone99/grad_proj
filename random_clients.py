from flwr.common import FitRes, Status, Parameters, ndarrays_to_parameters, parameters_to_ndarrays
import argparse
import numpy as np
import torch
import flwr as fl
from typing import Dict, List, Tuple
from collections import deque
import cifar

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

status = Status(code=0, message="Success")
status_error = Status(code=1, message="An Error Occured")

class RandomClient(fl.client.NumPyClient):
    def __init__(self, model, dataset_size):
        self.model = model
        self.dataset_size = dataset_size
        # Initialize received parameters as a deque to store history
        self.received_parameters = deque(maxlen=2)  # Store only the last 2 rounds of parameters
    
    def get_parameters(self, config):
        # Generate random parameters with the same shape as model parameters
        random_parameters = []
        for param in self.model.parameters():
            random_parameters.append(np.random.rand(*param.shape).astype(np.float32))
        return random_parameters
    
    def fit(self, parameters, config):
        try:
            # 랜덤 파라미터 생성
            random_parameters = [
                np.random.rand(*shape).astype(np.float32) for shape in self.parameter_shapes
            ]

            # 디버깅: 생성된 파라미터 형태 출력
            print(f"[DEBUG] Generated random parameters with shapes: {[param.shape for param in random_parameters]}")

            # Flower 형식으로 반환
            return ndarrays_to_parameters(random_parameters), self.dataset_size, {"accuracy": np.random.rand()}

        except Exception as e:
            # 에러 발생 시 기본값 반환
            print(f"[ERROR] Error in fit: {e}")
            dummy_parameters = [
                np.zeros((1,), dtype=np.float32)  # 기본 더미 파라미터
            ]
            return ndarrays_to_parameters(dummy_parameters), 0, {"error": str(e)}
        
    def evaluate(self, parameters, config):
        # Simulate evaluation
        loss = 1.0     # Example loss value
        accuracy = 0.0    # Example accuracy value
        return loss, self.dataset_size, {"accuracy": accuracy}
    
def main():
    parser = argparse.ArgumentParser(description="Random FL  Client")
    parser.add_argument("--partition-id", type=int, required=True, help="Client partition ID")
    args = parser.parse_args()

    # Initialize the model
    model = cifar.Net()
    print("Model initialized with layers:")
    for name, param in model.state_dict().items():
        print(f"{name}: {param.shape}")   # Log the model's parameters shape
    
    # Partition ID and mock dataset size
    partition_id = args.partition_id
    dataset_size = 1000

    # Initialize RandomClient
    client = RandomClient(model, dataset_size).to_client()

    # Start the Flower Client
    fl.client.start_client(server_address="192.168.0.40:8080", client=client)

if __name__ == "__main__":
    main()