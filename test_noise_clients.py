import numpy as np
import os

# Echo Client 생성 함수
def create_noisy_parameters(server_parameters, noise_level=0.01, num_clients=5):
    echo_clients = []
    for i in range(num_clients):
        noisy_params = {key: value + np.random.normal(0, noise_level, value.shape) for key, value in server_parameters.items()}
        echo_clients.append(noisy_params)
    return echo_clients

# 서버 파라미터 불러오기
server_parameters_path = "./test_parameters/server_parameters_npy/server_params"
server_param_num = 11
save_dir = "./test_parameters/test_echo_parameters"
os.makedirs(save_dir, exist_ok=True)

while (server_param_num <= 100):
    # 서버 파라미터 로드
    server_params_path = f"{server_parameters_path}{server_param_num}.npy"
    server_parameters = np.load(server_params_path, allow_pickle=True).item()

    # Echo Client 생성
    num_echo_clients = 1  # 생성할 클라이언트 수
    noise_level = 0.05  # 노이즈 수준 (표준편차)
    echo_clients = create_noisy_parameters(server_parameters, noise_level=noise_level, num_clients=num_echo_clients)

    # Echo Client 파라미터 저장
    for i, client_params in enumerate(echo_clients):
        save_path = os.path.join(save_dir, f"echo_client_{server_param_num}.npy")
        np.save(save_path, client_params)
        print(f"Echo client parameters saved to {save_path}")
    
    server_param_num += 1
