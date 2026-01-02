from typing import List, Tuple, Dict
from flwr.common import parameters_to_ndarrays, Metrics
import joblib
import torch
from torch import nn
from sklearn.cluster import KMeans
import numpy as np
import flwr as fl

# DAGMM model definition
class DAGMM(nn.Module):
    def __init__(self, input_dim, latent_dim=2):
        super(DAGMM, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed

# Load models
def load_models():
    """
    Load DAGMM and K-Means models.
    Returns:
        dagmm_model, kmeans_model
    """
    import os
    import torch

    # Load K-Means model
    with open("kmeans_model54.pkl", "rb") as f:
        kmeans_model = joblib.load(f)
    print("[INFO] K-Means model loaded successfully.")

    # Load DAGMM model state_dict
    state_dict = torch.load("dagmm_model54.pth", map_location=torch.device("cpu"), weights_only=True)
    
    # Detect input_dim dynamically
    input_dim = None
    for key, value in state_dict.items():
        if key.startswith("encoder.0.weight"):
            input_dim = value.size(1)  # Extract the input dimension
            print(f"[INFO] Detected input dimension from state_dict: {input_dim}")
            break

    if input_dim is None:
        raise ValueError("Input dimension could not be detected from the state_dict.")

    # Initialize and load the DAGMM model
    dagmm_model = DAGMM(input_dim)  # 올바르게 모델 초기화
    current_dim = dagmm_model.encoder[0].in_features  # 현재 encoder의 입력 크기

    if current_dim != input_dim:
        print(f"[INFO] Adjusting DAGMM model input dimension from {current_dim} to {input_dim}")
        dagmm_model = DAGMM(input_dim)  # 입력 크기를 조정하여 DAGMM 모델 초기화

    dagmm_model.load_state_dict(state_dict)
    dagmm_model.eval()
    print("[INFO] DAGMM model loaded successfully.")

    return dagmm_model, kmeans_model

# Classify client
import torch
import numpy as np
from flwr.common import parameters_to_ndarrays
def classify_client(parameters, dagmm_model, kmeans_model) -> str:
    """Classify client based on their parameters using DAGMM and KMeans models."""
    from flwr.common import parameters_to_ndarrays

    # 클러스터 ID와 라벨 매핑
    cluster_labels = {
        0: "normal",
        1: "lazy",
        2: "noised",
        3: "random",
        4: "server",
        5: "echo"
    }

    # Convert Parameters object to a list of Numpy arrays
    parameter_list = parameters_to_ndarrays(parameters)
    print(f"Parameter list length: {len(parameter_list)}")

    # Flatten and concatenate all parameter arrays
    numeric_parameters = np.concatenate([param.flatten() for param in parameter_list])
    print(f"Numeric parameters shape: {numeric_parameters.shape}")

    # Prepare data for DAGMM
    params_tensor = torch.tensor(numeric_parameters, dtype=torch.float32).unsqueeze(0)
    print(f"Input tensor shape for DAGMM: {params_tensor.shape}")

    input_dim = params_tensor.shape[1]
    if dagmm_model.encoder[0].in_features != input_dim:
        print(f"[INFO] Adjusting DAGMM model for new input dimension: {input_dim}")
        dagmm_model = DAGMM(input_dim)
        state_dict = torch.load("dagmm_model54.pth", map_location=torch.device("cpu"))
        dagmm_model.load_state_dict(state_dict)
        dagmm_model.eval()

    # Use the DAGMM model for anomaly detection
    with torch.no_grad():
        latent, _ = dagmm_model(params_tensor)

    # Use the K-Means model for classification
    cluster = kmeans_model.predict(latent.numpy())[0]

    # Return the label corresponding to the cluster ID
    return cluster_labels.get(cluster, "unknown")

# Update main server workflow
def main():
    # Determine input_dim dynamically based on client data or predefined size
    dagmm_model, kmeans_model = load_models()

    # Federated server logic
    class CustomFedAvg(fl.server.strategy.FedAvg):
        def aggregate_fit(
            self,
            rnd: int,
            results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
            failures: List[BaseException],
        ) -> Tuple[List[np.ndarray], Dict[str, Metrics]]:
            print(f"Aggregating results for round {rnd}...")
            print(f"Received {len(results)} results, {len(failures)} failures.")
            
            if not results:
                print("[ERROR] No results to aggregate.")
                return None, {}

            try:
                # 분류 및 로깅 시작
                for client_proxy, fit_res in results:
                    # 클라이언트 ID 가져오기
                    client_id = client_proxy.cid

                    # 클라이언트의 파라미터를 사용하여 분류
                    parameters = fit_res.parameters
                    classification = classify_client(parameters, dagmm_model, kmeans_model)
                    with open("classification_results1.txt", "a") as file:
                        file.write(f"Client {client_id}: {classification}\n")
                    # 분류 결과 로그 출력
                    print(f"Client {client_id} classified as: {classification}")
                # 분류 및 로깅 종료

                # 부모 클래스의 aggregate_fit 호출
                return super().aggregate_fit(rnd, results, failures)
            
            except Exception as e:
                print(f"[ERROR] Exception during aggregation: {e}")
                raise


    strategy = CustomFedAvg()

    # Corrected `start_server` call
    fl.server.start_server(
        server_address="192.168.0.40:8080",  # Explicit keyword argument
        config=fl.server.ServerConfig(num_rounds=40),  # Explicit keyword argument
        strategy=strategy  # Explicit keyword argument
    )

if __name__ == "__main__":
    main()
