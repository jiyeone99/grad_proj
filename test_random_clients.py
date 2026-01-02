import numpy as np
import os

# 랜덤 파라미터 생성 함수
def generate_random_parameters():
    parameters = {
        "conv1.weight": np.random.randn(16, 3, 3, 3).astype(np.float32),
        "conv1.bias": np.random.randn(16).astype(np.float32),
        "conv2.weight": np.random.randn(32, 16, 3, 3).astype(np.float32),
        "conv2.bias": np.random.randn(32).astype(np.float32),
        "fc1.weight": np.random.randn(128, 2048).astype(np.float32),
        "fc1.bias": np.random.randn(128).astype(np.float32),
        "fc2.weight": np.random.randn(10, 128).astype(np.float32),
        "fc2.bias": np.random.randn(10).astype(np.float32),
    }
    return parameters

# 저장 경로
save_dir = "./random_parameters_npy9"
os.makedirs(save_dir, exist_ok=True)

# 랜덤 파라미터 생성 및 저장
for i in range(10):  # 예시로 10개의 랜덤 파라미터 세트 생성
    parameters = generate_random_parameters()
    save_path = os.path.join(save_dir, f"random_params_{i}.npy")
    # np.save는 딕셔너리를 바로 저장할 수 있음
    np.save(save_path, parameters)
    print(f"Random parameters saved to {save_path}")
