import os
import numpy as np
import torch
from torch import nn, optim
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import joblib

# DAGMM 네트워크 정의
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

# 데이터 로드 및 라벨링
def load_and_label_parameters(base_dir, label_map, max_length):
    all_data = []
    all_labels = []

    for folder, label in label_map.items():
        folder_path = os.path.join(base_dir, folder)
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)

            if not file_path.endswith(".pth"):
                print(f"[WARNING] Skipped non-pth file: {file_path}")
                continue

            try:
                # .pth 파일 로드 (weights_only=True)
                state_dict = torch.load(file_path, weights_only=True)
                param = np.concatenate([v.cpu().numpy().flatten() for v in state_dict.values()])

                # 패딩 처리
                if len(param) > max_length:
                    raise ValueError(f"Parameter length exceeds max_length: {len(param)} > {max_length}")
                param_padded = np.pad(param, (0, max_length - len(param)), mode="constant")
                all_data.append(param_padded)
                all_labels.append(label)
            except Exception as e:
                print(f"[ERROR] Skipped {file_path}: {e}")
                continue

    if not all_data:
        raise ValueError("No valid data was loaded. Please check the input files and paths.")

    return np.array(all_data, dtype=np.float32), np.array(all_labels, dtype=np.int32)

def get_max_length(base_dir, label_map):
    max_length = 0
    for folder in label_map.keys():
        folder_path = os.path.join(base_dir, folder)
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)

            # .pth 파일만 처리
            if not file_path.endswith(".pth"):
                continue

            try:
                # 파라미터 로드
                param = torch.load(file_path)

                # 파라미터 길이 계산
                if isinstance(param, dict):
                    param_length = sum(v.numel() for v in param.values())
                elif isinstance(param, torch.Tensor):
                    param_length = param.numel()
                else:
                    continue

                max_length = max(max_length, param_length)
            except Exception as e:
                print(f"[ERROR] Failed to process {file_path}: {e}")
                continue

    if max_length == 0:
        raise ValueError("Failed to calculate max_length. No valid files were processed.")
    
    print(f"[INFO] Calculated max_length: {max_length}")
    return max_length

# 데이터 로드
base_dir = "./fl_parameters"
label_map = {
    "global_parameters": 0,
    "lazy_client_parameters": 1,
    "noised_client_parameters": 2,
    "random_parameters": 3,
    "worthless_parameters": 4,
    "normal_client_parameters": 5,
}

# 최대 길이 계산
max_length = get_max_length(base_dir, label_map)

# 데이터 로드
all_data, all_labels = load_and_label_parameters(base_dir, label_map, max_length)

# 데이터 전처리 및 학습 진행
scaler = StandardScaler()
all_data_scaled = scaler.fit_transform(all_data)
all_data_tensor = torch.tensor(all_data_scaled, dtype=torch.float32)

# DAGMM 학습
input_dim = all_data_tensor.shape[1]
dagmm_model = DAGMM(input_dim=input_dim, latent_dim=2)

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

train_dagmm(dagmm_model, all_data_tensor)

# K-Means 학습
dagmm_model.eval()
with torch.no_grad():
    latent_space, _ = dagmm_model(all_data_tensor)

kmeans = KMeans(n_clusters=6, random_state=42)
labels = kmeans.fit_predict(latent_space.numpy())

# 평가 지표
silhouette_avg = silhouette_score(latent_space.numpy(), labels)
db_index = davies_bouldin_score(latent_space.numpy(), labels)

print(f"Silhouette Score: {silhouette_avg}")
print(f"Davies-Bouldin Index: {db_index}")

# 모델 저장
# torch.save(dagmm_model.state_dict(), os.path.join(base_dir, "dagmm_model.pth"))
# 모델 저장
torch.save(dagmm_model.state_dict(), "model_state.pth")
joblib.dump(kmeans, os.path.join(base_dir, "kmeans_model.pkl"))
print("Models saved successfully.")
