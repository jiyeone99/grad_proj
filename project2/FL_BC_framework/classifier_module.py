import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import joblib
import torch
import torch.optim as optim
from collections import Counter
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from models import SimpleCNN, DAGMM
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_dagmm_supervised(data, labels, input_dim, num_classes=6, epochs=100, lr=0.001):
    model = DAGMM(input_dim).to(device)
    classifier = nn.Linear(model.latent_dim, num_classes).to(device)
    optimizer = optim.AdamW(list(model.parameters()) + list(classifier.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    # 클래스별 오버샘플 비율 설정 (예: 클래스 1, 5, 6을 더 많이 보정)
    oversample_ratios = {1: 3, 5: 3, 6: 2}  # 필요시 조정 가능

    # 원본 클래스 분포
    class_counts = Counter(labels)
    augmented_data, augmented_labels = [], []

    for cls in np.unique(labels):
        cls_indices = np.where(labels == cls)[0]
        cls_data = data[cls_indices]
        cls_labels = labels[cls_indices]
        repeat_factor = oversample_ratios.get(cls, 1)
        for _ in range(repeat_factor):
            augmented_data.append(cls_data)
            augmented_labels.append(cls_labels)

    data_aug = np.vstack(augmented_data)
    labels_aug = np.concatenate(augmented_labels)

    print("[INFO] Sample count per class (after oversampling):", dict(Counter(labels_aug)))

    # 클래스 가중치 적용
    unique_labels = np.unique(labels_aug)
    class_weights = compute_class_weight(class_weight="balanced", classes=unique_labels, y=labels_aug)
    weight_tensor = torch.tensor([class_weights[list(unique_labels).index(i)] if i in unique_labels else 1.0 for i in range(num_classes)], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    data_tensor = torch.tensor(data_aug, dtype=torch.float32, device=device)
    label_tensor = torch.tensor(labels_aug, dtype=torch.long, device=device)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        latent, reconstructed, energy = model(data_tensor)
        outputs = classifier(latent)
        loss = criterion(outputs, label_tensor) + model.compute_loss(data_tensor, reconstructed, energy)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            _, predicted = torch.max(outputs, 1)
            acc = (predicted == label_tensor).sum().item() / len(labels_aug) * 100
            f1 = f1_score(label_tensor.cpu(), predicted.cpu(), average="macro")
            print(f"[INFO] DAGMM Epoch {epoch+1}/{epochs} - Loss: {loss.item():.6f}, Accuracy: {acc:.2f}%, F1 Score: {f1:.4f}")

    return model, classifier


# 이진 분류를 위한 라벨 변환 함수
def convert_to_binary(labels, normal_class=0):
    return np.array([0 if label == normal_class else 1 for label in labels])

# DAGMM 학습 (이진 분류 라벨 적용)
def train_dagmm_binary(data, labels, input_dim, latent_dim=10, epochs=100, lr=0.001):
    binary_labels = convert_to_binary(labels)

    model = DAGMM(input_dim, latent_dim=latent_dim).to(device)
    classifier = nn.Linear(latent_dim, 2).to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(classifier.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    # 클래스 가중치 계산
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(binary_labels), y=binary_labels)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    data_tensor = torch.tensor(data, dtype=torch.float32, device=device)
    label_tensor = torch.tensor(binary_labels, dtype=torch.long, device=device)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        latent, reconstructed, energy = model(data_tensor)
        outputs = classifier(latent)
        loss = criterion(outputs, label_tensor) + model.compute_loss(data_tensor, reconstructed, energy)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            _, predicted = torch.max(outputs, 1)
            acc = (predicted == label_tensor).sum().item() / len(binary_labels) * 100
            f1 = f1_score(label_tensor.cpu(), predicted.cpu(), average="macro")
            print(f"[INFO] Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}, Accuracy: {acc:.2f}%, F1 Score: {f1:.4f}")

    return model, classifier


# SVM 학습
def train_svm_with_hard_negative_mining(latent, labels, rounds=3):
    # 초기 SVM 학습
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
    svm.fit(latent, labels)

    for r in range(rounds):
        # 현재 예측 결과 확인
        preds = svm.predict(latent)
        hard_negatives = []

        for i in range(len(labels)):
            if labels[i] == 1 and preds[i] == 0:  # 비정상인데 정상으로 잘못 분류된 경우
                hard_negatives.append(i)

        if not hard_negatives:
            print(f"[HNM] No hard negatives found at round {r+1}.")
            break

        # hard negatives 샘플들만 추출하여 다시 학습에 추가
        hard_X = latent[hard_negatives]
        hard_y = labels[hard_negatives]

        print(f"[HNM] Round {r+1}: Found {len(hard_y)} hard negatives")

        # 원래 데이터 + hard negatives 오버샘플링하여 학습
        X_aug = np.vstack([latent, hard_X, hard_X])
        y_aug = np.concatenate([labels, hard_y, hard_y])  # 오버샘플링

        svm.fit(X_aug, y_aug)

    return svm


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# 추적용 리스트 전역 선언 (또는 클래스 내에 위치해도 OK)
f1_history = []
f1_class_history = []
conf_matrices = []

def evaluate_svm(svm_model, data, labels, round_num=None):
    predictions = svm_model.predict(data)
    acc = accuracy_score(labels, predictions) * 100
    f1_macro = f1_score(labels, predictions, average="macro")
    f1_per_class = f1_score(labels, predictions, average=None)

    print(f"[INFO] SVM Supervised Accuracy: {acc:.2f}%")
    print(f"[INFO] SVM Macro F1 Score: {f1_macro:.4f}")
    print("[INFO] Classification Report:")
    print(classification_report(labels, predictions))

    # 기록 저장 (시각화용)
    if round_num is not None:
        f1_history.append((round_num, f1_macro))
        f1_class_history.append((round_num, f1_per_class))
        cm = confusion_matrix(labels, predictions)
        conf_matrices.append((round_num, cm))

        # 시각화 출력 (원하면 저장 가능)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title(f"Confusion Matrix (Round {round_num})")
        plt.show()

    return acc


def plot_f1_trend():
    import matplotlib.pyplot as plt
    import numpy as np

    rounds = [r for r, _ in f1_history]
    macro_f1s = [f for _, f in f1_history]

    plt.figure(figsize=(8, 5))
    plt.plot(rounds, macro_f1s, marker='o', label="Macro F1")
    plt.xlabel("Round")
    plt.ylabel("F1 Score")
    plt.title("Macro F1 Score Over Rounds")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_per_class_f1():
    import matplotlib.pyplot as plt
    import numpy as np

    rounds = [r for r, _ in f1_class_history]
    per_class = np.array([f for _, f in f1_class_history])  # shape: [num_rounds, num_classes]

    plt.figure(figsize=(10, 6))
    for i in range(per_class.shape[1]):
        plt.plot(rounds, per_class[:, i], marker='o', label=f"Class {i}")
    plt.xlabel("Round")
    plt.ylabel("F1 Score")
    plt.title("Per-Class F1 Score Over Rounds")
    plt.grid(True)
    plt.legend()
    plt.show()

from sklearn.model_selection import train_test_split

class MetaClassifier:
    def __init__(self, feature_dim=7):
        self.feature_dim = feature_dim
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.training_data = []
        self.labels = []

    def add_training_sample(self, x, y):
        x = np.array(x)
        if len(x.shape) == 1:
            self.training_data.append(x)
        else:
            # (1, n_features) 같은 거라면 그냥 첫번째꺼만 가져오도록 보정
            self.training_data.append(x.flatten())
        self.labels.append(y)



    def train(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def evaluate(self, X, y):
        y_pred = self.model.predict(X)
        return f1_score(y, y_pred, average="binary")

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)
    
    def retrain(self):
        if len(self.training_data) >= 50:  # 데이터가 충분할 때만 학습
            X = np.array(self.training_data)
            y = np.array(self.labels)

            # ✅ 데이터가 1차원으로 들어갔다면 자동 reshape
            if len(X.shape) == 1:
                X = X.reshape(1, -1)
            elif len(X.shape) == 3:
                # 혹시 실수로 (n, 1, f) 꼴일 때도 방어
                X = X.reshape(X.shape[0], -1)

            self.train(X, y)
            print(f"[RETRAIN] Meta-classifier retrained on {len(y)} samples.")


labels = {
    0: "normal",
    1: "lazy",
    2: "noised",
    3: "random",
    4: "server",
    5: "echo",
    6: "small"
}

def set_parameters(model, parameters):
    """Flower Parameters 객체를 PyTorch 모델에 세팅하는 함수"""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)

