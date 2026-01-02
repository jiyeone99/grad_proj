import os
import shutil

def organize_parameters(input_dir, output_dir):
    """
    여러 폴더에 분산된 클라이언트 파라미터 파일을 하나의 폴더로 모으고 파일명을 재정렬하여 저장합니다.
    
    Parameters:
        input_dir (str): 클라이언트 파라미터 폴더가 모인 상위 디렉토리.
        output_dir (str): 통합된 파라미터 파일을 저장할 디렉토리.
    """
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    file_counter = 0
    for folder in sorted(os.listdir(input_dir)):
        folder_path = os.path.join(input_dir, folder)

        if os.path.isdir(folder_path):
            for filename in sorted(os.listdir(folder_path)):
                file_path = os.path.join(folder_path, filename)

                if filename.endswith(".npy"):
                    # 새로운 파일명 설정
                    new_filename = f"noisy_client_{file_counter}_params.npy"
                    output_path = os.path.join(output_dir, new_filename)

                    # 파일 이동 및 이름 변경
                    shutil.copy(file_path, output_path)
                    print(f"Copied: {file_path} -> {output_path}")
                    file_counter += 1

# 실행 설정
input_dir = "./test_parameters/test_insufficient_or_noisy_client_parameters"  # 기존 파라미터 폴더들이 위치한 디렉토리
output_dir = "./test_parameters/organized_noisy_parameters"   # 통합된 파라미터 저장 디렉토리

organize_parameters(input_dir, output_dir)
