import numpy as np
import os

# 정상 클라이언트 파라미터가 저장된 디렉토리
client_dir = "./client_parameters15"
save_dir = "./server_parameters_npy"
os.makedirs(save_dir, exist_ok=True)

# FedAvg를 흉내내기 위한 함수
def fedavg_parameters(client_dir):
    client_files = [os.path.join(client_dir, f) for f in os.listdir(client_dir) if f.endswith(".npy")]
    if not client_files:
        raise ValueError("No client parameter files found.")
    
    # 첫 번째 클라이언트의 파라미터를 기반으로 초기화
    first_params = np.load(client_files[0], allow_pickle=True).item()
    avg_params = {key: np.zeros_like(value) for key, value in first_params.items()}
    num_clients = len(client_files)
    
    # 각 클라이언트의 파라미터를 합산
    for file in client_files:
        client_params = np.load(file, allow_pickle=True).item()
        for key in avg_params:
            avg_params[key] += client_params[key]
    
    # 평균 계산
    for key in avg_params:
        avg_params[key] /= num_clients
    
    return avg_params

# FedAvg 수행 및 서버 파라미터 저장
server_parameters = fedavg_parameters(client_dir)
server_save_path = os.path.join(save_dir, "server_parameters15.npy")
np.save(server_save_path, server_parameters)
print(f"Server parameters saved to {server_save_path}")
