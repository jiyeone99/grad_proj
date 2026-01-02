import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import torch
import joblib
from torch import nn, optim
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd

# DAGMM 네트워크 정의
class DAGMM(nn.Module):
    def __init__(self, input_dim, latent_dim=2):
        super(DAGMM, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        # Decoder
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

# DAGMM 학습 함수
def train_dagmm(model, data, epochs=50, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        latent, reconstructed = model(data)
        loss = criterion(reconstructed, data)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    return model

def load_and_process_parameters(folder_path, max_length):
    processed_parameters = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        try:
            param = np.load(file_path, allow_pickle=True)
            if isinstance(param, dict):
                param = np.concatenate([v.flatten() for v in param.values()])
            elif isinstance(param, np.ndarray) and param.ndim == 0:
                param = param.item()
                if isinstance(param, dict):
                    param = np.concatenate([v.flatten() for v in param.values()])
            # Pad to max_length
            param_padded = np.pad(param, (0, max_length - len(param)), mode='constant')
            processed_parameters.append(param_padded)
        except Exception as e:
            print(f"[ERROR] Skipped {file_path}: {e}")
    return np.array(processed_parameters)

# Determine the maximum length across all datasets
folders = [
    "./organized_normal_parameters",
    "./organized_lazy_parameters",
    "./organized_noised_parameters",
    "./organized_random_parameters",
    "./organized_server_parameters",
    "./organized_echo_parameters"
]

# Calculate the maximum parameter length
max_length = 0
for folder in folders:
    for file_name in os.listdir(folder):
        file_path = os.path.join(folder, file_name)
        try:
            param = np.load(file_path, allow_pickle=True)
            if isinstance(param, dict):
                param = np.concatenate([v.flatten() for v in param.values()])
            elif isinstance(param, np.ndarray) and param.ndim == 0:
                param = param.item()
                if isinstance(param, dict):
                    param = np.concatenate([v.flatten() for v in param.values()])
            max_length = max(max_length, len(param))
        except Exception as e:
            print(f"[ERROR] Skipped {file_path}: {e}")

print(f"[INFO] Maximum parameter length determined: {max_length}")

# Load and process parameters
normal_params = load_and_process_parameters("./organized_normal_parameters", max_length)
lazy_params = load_and_process_parameters("./organized_lazy_parameters", max_length)
noisy_params = load_and_process_parameters("./organized_noised_parameters", max_length)
random_params = load_and_process_parameters("./organized_random_parameters", max_length)
server_params = load_and_process_parameters("./organized_server_parameters", max_length)
echo_params = load_and_process_parameters("./organized_echo_parameters", max_length)

# Combine all data
all_data = np.vstack([
    normal_params,
    lazy_params,
    noisy_params,
    random_params,
    server_params,
    echo_params
])

# 데이터 전처리
scaler = StandardScaler()
all_data_scaled = scaler.fit_transform(all_data)

# PyTorch Tensor로 변환
all_data_tensor = torch.tensor(all_data_scaled, dtype=torch.float32)

def update_dagmm_model(dagmm_model, input_dim):
    dagmm_model.encoder[0] = nn.Linear(input_dim, 128)
    dagmm_model.decoder[-1] = nn.Linear(128, input_dim)
    return dagmm_model

# DAGMM 초기화 및 학습
input_dim = all_data_tensor.shape[1]  # Dynamically determine input dimension
dagmm_model = DAGMM(input_dim=268650, latent_dim=2)
dagmm_model = update_dagmm_model(dagmm_model, input_dim)
dagmm_model = train_dagmm(dagmm_model, all_data_tensor, epochs=50)

# DAGMM 잠재 공간 추출
dagmm_model.eval()
with torch.no_grad():
    latent_space, _ = dagmm_model(all_data_tensor)
    print(f"[DEBUG] Latent vector: {latent_space}")
    if torch.isnan(latent_space).any():
        print("[ERROR] Latent vector contains NaN values.")
        latent = torch.nan_to_num(latent_space)
    # 기존 코드
    # latent_space, _ = dagmm_model(all_data_tensor)
    # assert latent_space is not None and latent_space.shape[0] > 0, "Latent space is empty or invalid."


# K-Means 클러스터링
kmeans = KMeans(n_clusters=6, random_state=42)
labels = kmeans.fit_predict(latent_space.numpy())

# 결과 출력
print(f"K-Means Cluster Centers:\n{kmeans.cluster_centers_}")
print(f"Assigned Labels:\n{labels}")

silhouette_avg = silhouette_score(latent_space.numpy(), labels)
db_index = davies_bouldin_score(latent_space.numpy(), labels)

print(f"Silhouette Score: {silhouette_avg}")
print(f"Davies-Bouldin Index: {db_index}")

# DAGMM 모델 저장
torch.save(dagmm_model.state_dict(), "dagmm_model2.pth")
print("Model state_dict saved successfully.")

# K-Means 모델 저장
try:
    joblib.dump(kmeans, "kmeans_model2.pkl")
    print("K-Means model saved successfully.")
except Exception as e:
    print(f"Failed to save K-Means model: {e}")
