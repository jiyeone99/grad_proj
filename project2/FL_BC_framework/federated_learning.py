import math
import datasets
import joblib
from matplotlib import pyplot as plt, transforms
import pandas as pd
import torch
import torch.nn as nn
from tracker import ClientTracker, PunishedClientPool, make_client
from classifier_module import MetaClassifier, set_parameters
from nodes import FederatedClient, LazyClient, EchoClient, RandomClient, SmallClient
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from flwr.common import parameters_to_ndarrays
import numpy as np
from flwr.common import ndarrays_to_parameters
from scipy.spatial.distance import cosine
import flwr as fl
from torch.utils.data import DataLoader, Subset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomFedAvgWithDetection(fl.server.strategy.FedAvg):
    def __init__(self, model, pool, tracker, penalty_threshold=3, penalty_mode: str = "accumulated",
                 switch_round=15, eval_dataset=None, decay=2):
        super().__init__()
        self.model = model
        self.pool = pool
        self.tracker = tracker
        self.penalty_threshold = penalty_threshold
        self.penalty_mode = penalty_mode
        self.switch_round = switch_round
        # self.input_dim = 268650
        self.input_dim = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.eval_dataset = eval_dataset  # ← 이 줄이 중요!
        self.decay = decay

        self.window = 3
        self.a = 2.0
        self.b = 1.0

        self.loss_per_round = []      # 글로벌 validation loss per round
        self.normal_abnormal_count = []  # (normal_count, abnormal_count) per round

    # ✅ F1 기반 패널티 조건
    def is_f1_penalty(self, cid, threshold=0.7, window=3):
        trues, preds = self.tracker.get_recent_predictions(cid, num_rounds=window)
        if len(trues) < window:
            return False  # 데이터 부족 시 보류
        f1 = f1_score(trues, preds, average="macro", zero_division=0)
        print(f"[F1 Check] CID: {cid} - Macro F1: {f1:.4f}")
        return f1 < threshold

    def evaluate_global_model(self, parameters):
        model = self.model
        set_parameters(model, parameters)
        model.eval()

        test_loader = DataLoader(self.eval_dataset, batch_size=32)
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)

        return correct / total

    def should_remove_client_consecutive(self, cid: str) -> bool:
        return self.tracker.is_consecutively_penalized(cid, self.penalty_threshold)

    def should_remove_client_accumulated(self, cid: str) -> bool:
        return self.tracker.penalty_log.get(cid, []).count(True) >= self.penalty_threshold

    # ✅ 패널티 기준 판단 함수
    def is_penalized(self, cid, threshold):
        history = self.tracker.penalty_log[cid]
        return sum(history) >= threshold
    
    def is_f1_below_threshold(self, cid, threshold):
        f1 = self.tracker.get_client_f1(cid)
        if f1 is None:
            return False
        return f1 < threshold

    def is_consecutive_f1_below_threshold(self, cid, threshold):
        history = self.tracker.get_client_f1_history(cid)
        if not history or len(history) < threshold:
            return False
        # 최근 N개가 모두 f1 < threshold 인지 확인
        return all(f1 < threshold for f1 in history[-threshold:])


    # ✅ 클라이언트 제거 여부 판단
    def should_kick(self, cid, current_round, threshold, mode):
        if mode == "hybrid":
            if current_round < self.switch_round:
                return self.is_penalized(cid, threshold)
            else:
                return self.tracker.is_consecutively_penalized(cid, threshold)
        elif mode == "consecutive":
            return self.tracker.is_consecutively_penalized(cid, threshold)
        elif mode == "accumulated":
            return self.is_penalized(cid, threshold)
        else:
            raise ValueError(f"Unknown penalty mode: {mode}")

    def evaluate_global_loss(self, params):
        """Global model의 validation loss를 계산 (정상적인 파라미터 적용)"""
        model = self.model
        model.load_state_dict(self.ndarrays_to_state_dict(params))  # 직접 state_dict 변환
        model.eval()

        val_loader = DataLoader(self.eval_dataset, batch_size=32, shuffle=False)
        criterion = nn.CrossEntropyLoss()

        total_loss = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)

        avg_loss = total_loss / total_samples
        return avg_loss

    def ndarrays_to_state_dict(self, ndarrays):
        """Flower parameters를 SimpleCNN의 state_dict로 변환"""
        model = self.model
        state_dict = model.state_dict()
        new_state_dict = {}

        # ndarrays는 리스트로 들어온다 (list of np.ndarray)
        idx = 0
        for key in state_dict.keys():
            arr = ndarrays[idx]
            new_state_dict[key] = torch.tensor(arr)
            idx += 1
        return new_state_dict

    def aggregate_fit_internal(self, results, failures):
        """Aggregate parameters from clients."""
        if not results:
            return None, {}
        
        # (num_examples, [np.ndarray, np.ndarray, ...]) 리스트
        weights_results = []
        for client_proxy, fit_res in results:
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            weights_results.append((fit_res.num_examples, ndarrays))
        
        # 각 레이어별로 가중합을 계산
        num_layers = len(weights_results[0][1])
        total_examples = sum(num_examples for num_examples, _ in weights_results)

        # layer별로 가중 평균
        weighted_parameters = []
        for layer_idx in range(num_layers):
            layer_sum = np.zeros_like(weights_results[0][1][layer_idx])
            for num_examples, ndarrays in weights_results:
                layer_sum += num_examples * ndarrays[layer_idx]
            layer_avg = layer_sum / total_examples
            weighted_parameters.append(layer_avg)

        aggregated_parameters = ndarrays_to_parameters(weighted_parameters)
        aggregated_metrics = {}
        return aggregated_parameters, aggregated_metrics

    def get_linear_dynamic_thresholds(self, current_round, total_rounds=30, min_lower=0.0010, min_upper=0.0015, max_upper=0.35):
        """
        라운드가 진행됨에 따라 penalty 임계값을 점진적으로 줄임.
        - current_round: 현재 라운드 번호
        - total_rounds: 전체 라운드 수
        """
        # 예: 선형 감소
        decay_ratio = current_round / total_rounds
        dynamic_upper = max_upper - (max_upper - min_upper) * decay_ratio

        # lower는 고정 혹은 살짝만 변화 (optional)
        dynamic_lower = min_lower  # 필요시 여기도 줄여도 됨

        return dynamic_lower, dynamic_upper
    
    def get_dynamic_thresholds(
        self,
        current_round,
        total_rounds=30,
        min_min=0.0005,    # 최소 최소값
        max_min=0.01,      # 초기 최소 임계값
        min_upper=0.0015,
        max_upper=0.35
    ):
        """
        지수적 감소 방식을 적용해 상·하 임계값을 동적으로 조정.
        self.decay: 지수 감쇠 강도 (값이 클수록 초반에 급격히 감소)
        """
        decay_ratio = current_round / total_rounds

        # upper threshold (정상 노드 포함 기준)
        dynamic_upper = min_upper + (max_upper - min_upper) * np.exp(-self.decay * decay_ratio)

        # lower threshold (비정상 노드 걸러낼 하한)
        dynamic_lower = min_min + (max_min - min_min) * np.exp(-self.decay * decay_ratio)

        return dynamic_lower, dynamic_upper


    def aggregate_fit(self, rnd, results, failures):
        print(f"[INFO] Aggregating results for round {rnd}...")

        true_labels = []
        pred_labels = []

        # 🔵 Global model 파라미터 벡터 준비
        aggregated_params_ndarrays = parameters_to_ndarrays(
            self.aggregate_fit_internal(results, failures)[0]
        )
        global_vector = np.concatenate([p.flatten() for p in aggregated_params_ndarrays])
        self.tracker.add_param_log("server", "server", global_vector, rnd)

        # 🔵 클라이언트별 처리
        for client_proxy, fit_res in results:
            cid = fit_res.metrics["cid"]
            ctype = fit_res.metrics.get("type", "unknown")  # 클라이언트 타입 가져오기
            client = self.pool.get_client_by_id(cid)
            if client is None:
                continue

            # 클라이언트 벡터화
            client_params = parameters_to_ndarrays(fit_res.parameters)
            client_vector = np.concatenate([p.flatten() for p in client_params])

            self.tracker.add_param_log(cid, ctype, client_vector, rnd)
            
            # 벡터 거리 계산
            distance = cosine(global_vector, client_vector)
            self.tracker.update_distance(cid, distance)

            true_label = getattr(client, "true_label", "unknown")
            print(f"[DEBUG] CID: {cid}, True: {true_label}, Distance: {distance:.4f}")
            true_labels.append(1 if true_label != "normal" else 0)

            # 거리 기반 패널티 부여 (벡터 거리가 특정 기준 이하/이상일 때 페널티)
            penalty_lower, penalty_upper = self.get_dynamic_thresholds(rnd)
            # 기존 distance 계산 후
            penalized = self.tracker.record_distance_penalty(
                cid,
                distance,
                penalty_lower,
                penalty_upper
            )
            if penalized:
                print(f"[⚠️] Client {cid} penalized based on distance: {distance:.4f}")

            # 패널티 부여 및 클라이언트 교체
            if self.should_kick(cid, rnd, self.penalty_threshold, self.penalty_mode):
                print(f"[⚠️] Client {cid} penalized and kicked based on mode {self.penalty_mode}")
                self.tracker.log_kick(rnd, cid, self.penalty_mode, true_label)
                new_client = self.pool.replace_client(self.model, cid, rnd, self.switch_round)
                self.tracker.log_add(rnd, new_client.cid, new_client.get_type())
                self.tracker.remove_from_suspect_pool(cid)


        # ✅ 클라이언트 분포 및 상태 기록
        self.tracker.log_distribution(rnd, [
            {"type": self.pool.get_client_by_id(cid).true_label}
            for cid in self.pool.clients
        ])
        self.tracker.log_client_pool_status(rnd, self.pool)
        self.tracker.save_all_logs()

        # ✅ Global FL Accuracy 기록
        last_params = parameters_to_ndarrays(results[-1][1].parameters)
        fl_acc = self.evaluate_global_model(last_params)
        self.tracker.log_fl_accuracy(rnd, fl_acc)

        # 🔵 Global model loss 측정
        aggregated_parameters, _ = self.aggregate_fit_internal(results, failures)
        ndarrays = parameters_to_ndarrays(aggregated_parameters)
        loss = self.evaluate_global_loss(ndarrays)
        self.tracker.log_global_loss(rnd, loss)

        # 🔵 C-LSS 기록
        losses = self.tracker.get_global_loss_list()
        alpha = self.tracker.get_last_abnormal_ratio()
        beta = self.tracker.get_last_added_ratio()

        if rnd >= self.window:
            recent_losses = losses[rnd - self.window:rnd]
            avg_recent = np.mean(recent_losses)
            current_loss = losses[rnd - 1]
            gamma = (1 + self.a * alpha + self.b * beta) / np.log(rnd + 2)
            clss = (current_loss - avg_recent) / gamma
            self.tracker.log_clss(rnd, clss)
        
        if rnd % 5 == 0:
            self.tracker.visualize_client_vectors_by_round(rnd)
            self.tracker.visualize_client_vectors_3d()

        return super().aggregate_fit(rnd, results, failures)


def plot_fl_accuracy_comparison(trackers, labels):
    plt.figure(figsize=(12, 6))
    for tracker, label in zip(trackers, labels):
        df = pd.DataFrame(tracker.fl_accuracies)  # <- 정확한 속성명 사용!
        if not df.empty and "fl_accuracy" in df.columns:
            plt.plot(df["round"], df["fl_accuracy"], label=label)
        else:
            print(f"[WARN] No 'fl_accuracy' data in tracker: {label}")
    plt.title("Global Model Accuracy Over Rounds")
    plt.xlabel("Round")
    plt.ylabel("FL Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

