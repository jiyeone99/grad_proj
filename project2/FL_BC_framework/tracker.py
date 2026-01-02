import csv
import random
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# --- 클라이언트 풀 + 페널티 관리 ---
from collections import defaultdict
from nodes import FederatedClient, LazyClient, RandomClient, EchoClient, SmallClient
from torch.utils.data import DataLoader, Subset
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde
from matplotlib import cm


def make_client(cid, model, partition, label):
    model = model
    train = DataLoader(partition, batch_size=32, shuffle=True)
    test = DataLoader(partition, batch_size=32, shuffle=False)

    if label == "lazy":
        client = LazyClient(cid, model, train, test)
    elif label == "random":
        client = RandomClient(cid, model, train, test)
    elif label == "echo":
        client = EchoClient(cid, model, train, test)
    elif label == "small":
        client = SmallClient(cid, model, train, test)
    else:
        client = FederatedClient(cid, model, train, test)

    client.true_label = label  # 🔥 반드시 추가!
    return client

class ClientTracker:
    def __init__(self):
        self.history = []
        self.kick_log = []
        self.add_log = []
        self.prediction_log = []
        self.distribution_log = []
        self.penalty_log = defaultdict(list)  # 연속 패널티 추적용
        self.round_accuracies = []  # 정확도 기록용
        self.fl_accuracies = []
        self.type_accuracy_log = []  # 추가
        self.type_accuracies = []
        self.type_fl = []
        self.clss_log = []
        self.global_loss_log = []  # ✅ 추가: 글로벌 서버 loss 기록용
        self.suspect_pool = set()  # 🔥 거리 이상 클라이언트 ID 저장
        self.client_distances = {}  # 🔵 각 클라이언트별 거리 기록
        self.param_logs = []

    def update_distance(self, cid, distance):
        """클라이언트별 글로벌 서버와의 거리 기록"""
        self.client_distances[cid] = distance

    def add_to_suspect_pool(self, cid):
        """의심 노드 풀에 추가"""
        self.suspect_pool.add(cid)

    def remove_from_suspect_pool(self, cid):
        """의심 풀에서 제거"""
        self.suspect_pool.discard(cid)

    def is_in_suspect_pool(self, cid):
        """의심 풀 안에 있는지 확인"""
        return cid in self.suspect_pool
    
    def log_distribution(self, round_num, clients):
        counts = {
            "normal": 0, "lazy": 0, "noised": 0, "random": 0,
            "server": 0, "echo": 0, "small": 0
        }
        for client_type in clients:
            label = client_type["type"]
            if label not in counts:
                counts[label] = 0
            counts[label] += 1
        counts["round"] = round_num
        self.distribution_log.append(counts)

    def record_distance_penalty(self, cid, distance, lower, upper):
        """거리 기반으로 페널티 여부를 기록."""
        penalized = distance < lower or distance > upper
        self.penalty_log[cid].append(1 if penalized else 0)
        return penalized

    def log_kick(self, round_num, client_id, predicted_type, actual_type):
        self.kick_log.append({
            "round": round_num,
            "client_id": client_id,
            "predicted": predicted_type,
            "actual": actual_type
        })

    def log_add(self, round_num, client_id, added_type):
        self.add_log.append({
            "round": round_num,
            "client_id": client_id,
            "added_type": added_type
        })

    def log_prediction(self, round_num, client_id, predicted_type, actual_type):
        self.prediction_log.append({
            "round": round_num,
            "client_id": client_id,
            "predicted": predicted_type,
            "actual": actual_type
        })

    def log_type_accuracy(self, round_num, f1):
        self.type_accuracies.append({"round": round_num, "type_accuracy": f1})
    
    def log_type_f1(self, round_num, fl_macro):
        self.type_fl.append({"round": round_num, "type_fl": fl_macro})
        
    def get_recent_predictions(self, client_id, num_rounds):
        """지정된 client_id의 최근 num_rounds에 해당하는 예측과 실제값 반환"""
        preds, trues = [], []
        for entry in reversed(self.prediction_log):
            if entry["client_id"] == client_id:
                preds.append(entry["predicted"])
                trues.append(entry["actual"])
                if len(preds) >= num_rounds:
                    break
        return trues[::-1], preds[::-1]  # 오래된 순으로 반환
    
    def log_client_pool_status(self, round_num, pool):
        counts = {"normal": 0, "abnormal": 0, "round": round_num}
        
        active_client_ids = list(pool.clients.keys())  # 현재 참여중인 클라이언트 ID
        
        for cid in active_client_ids:
            client = pool.original_clients[cid]  # 원본에서 type을 읽음 (참여중인 애들만)
            label = getattr(client, "true_label", "unknown")
            if label == "normal":
                counts["normal"] += 1
            else:
                counts["abnormal"] += 1

        self.history.append(counts)

    def log_fl_accuracy(self, round_num, acc):
        self.fl_accuracies.append({"round": round_num, "fl_accuracy": acc})

    def save_accuracy_csv(self, filename):
        df = pd.DataFrame({
            "round": list(range(1, len(self.round_accuracies)+1)),
            "fl_accuracy": self.round_accuracies
        })
        df.to_csv(filename, index=False)
    
    def record_penalty(self, cid, is_penalized):
        self.penalty_log[cid].append(1 if is_penalized else 0)

    def log_clss(self, round_num, clss_score):
        self.clss_log.append({"round": round_num, "clss": clss_score})

    def plot_clss_over_rounds(self):
        df = pd.DataFrame(self.clss_log)
        plt.plot(df["round"], df["clss"], marker="o", color="purple")
        plt.title("Context-Aware Loss Stability Score (C-LSS) Over Rounds")
        plt.xlabel("Round")
        plt.ylabel("C-LSS")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def get_last_abnormal_ratio(self):
        if not self.history:
            return 0.0
        last = self.history[-1]
        total = last["normal"] + last["abnormal"]
        return last["abnormal"] / total if total > 0 else 0.0

    def get_last_added_ratio(self):
        if not self.add_log:
            return 0.0
        round_num = self.add_log[-1]["round"]
        count = sum(1 for log in self.add_log if log["round"] == round_num)
        total = len(self.history[-1]) if self.history else 1
        return count / total

    def get_global_loss_list(self):
        return [entry["global_loss"] for entry in self.global_loss_log]
    
    def log_global_loss(self, round_num, loss):
        """Global model loss 기록"""
        self.global_loss_log.append({"round": round_num, "global_loss": loss})

    def save_global_loss_csv(self, filename="global_loss_log.csv"):
        """Global model loss csv 저장"""
        df = pd.DataFrame(self.global_loss_log)
        df.to_csv(filename, index=False)

    def save_distribution_csv(self, filename="fl_client_distribution.csv"):
        """라운드별 클라이언트 분포를 CSV로 저장."""
        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            
            # 헤더를 클라이언트 타입별로 모두 포함시킬 수 있어!
            headers = ["Round", "Normal", "Lazy", "Noised", "Random", "Server", "Echo", "Small"]
            writer.writerow(headers)
            
            for entry in self.distribution_log:
                writer.writerow([
                    entry.get("round", 0),
                    entry.get("normal", 0),
                    entry.get("lazy", 0),
                    entry.get("noised", 0),
                    entry.get("random", 0),
                    entry.get("server", 0),
                    entry.get("echo", 0),
                    entry.get("small", 0),
                ])

    def is_consecutively_penalized(self, cid, threshold):
        history = self.penalty_log[cid]
        if len(history) < threshold:
            return False
        return all(p == 1 for p in history[-threshold:])

    def plot_loss_and_client_status(self):
        """Global loss + 클라이언트 비율을 하나로 합친 그래프"""

        # 📊 데이터 준비
        df_loss = pd.DataFrame(self.global_loss_log)
        df_clients = pd.DataFrame(self.history)

        # 🔄 병합 (left join)
        df = pd.merge(df_loss, df_clients, on="round", how="left").sort_values("round")

        # 📈 그래프 시작
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # ---- (1) Global Loss Plot (좌측 Y축)
        ax1.plot(df["round"], df["global_loss"], color="tab:blue", marker="o", label="Global Loss")
        ax1.set_xlabel("Round")
        ax1.set_ylabel("Global Validation Loss", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax1.grid(True, which='both', axis='y', linestyle='--', alpha=0.5)

        # ---- (2) Normal/Abnormal Clients Plot (우측 Y축)
        ax2 = ax1.twinx()
        ax2.bar(df["round"], df["normal"], color="tab:green", alpha=0.6, label="Normal Clients")
        ax2.bar(df["round"], df["abnormal"], bottom=df["normal"], color="tab:orange", alpha=0.6, label="Abnormal Clients")
        ax2.set_ylabel("Number of Clients", color="tab:gray")
        ax2.tick_params(axis="y", labelcolor="tab:gray")

        # 🏷️ 범례 합치기
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

        plt.title("Global Validation Loss & Client Status Over Rounds")
        fig.tight_layout()
        plt.show()


    def save_all_logs(self):
        pd.DataFrame(self.kick_log).to_csv("removed_clients.csv", index=False)
        pd.DataFrame(self.add_log).to_csv("added_clients.csv", index=False)
        pd.DataFrame(self.history).to_csv("client_distribution.csv", index=False)
        pd.DataFrame(self.distribution_log).to_csv("accuracy_log.csv", index=False)
        pd.DataFrame(self.fl_accuracies).to_csv("fl_accuracy_log.csv", index=False)

    def plot_client_distribution(self):
        df = pd.read_csv("client_distribution.csv")
        df.plot(x="round", y=["normal", "abnormal"], kind="bar", stacked=True)
        plt.title("Client Type Ratio per Round")
        plt.xlabel("Round")
        plt.ylabel("Number of Clients")
        plt.tight_layout()
        plt.show()

    def plot_accuracy_over_rounds(self, filename="accuracy_penalty3.csv"):
        df = pd.read_csv(filename)
        plt.plot(df["round"], df["accuracy"], marker="o")
        plt.title("Federated Learning Accuracy Over Rounds")
        plt.xlabel("Round")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def add_param_log(self, cid, ctype, vector, round_num):
        self.param_logs.append({
            "cid": cid,
            "type": ctype,
            "vector": vector,
            "round": round_num
        })

    def plot_convex_hulls(reduced, labels, color_map):
        unique_labels = set(labels)
        for label in unique_labels:
            points = reduced[np.array(labels) == label]
            if len(points) >= 3:  # 최소 3개 이상이어야 Hull 가능
                hull = ConvexHull(points)
                for simplex in hull.simplices:
                    plt.plot(points[simplex, 0], points[simplex, 1], color=color_map.get(label, "gray"), alpha=0.3)

    def visualize_client_vectors_by_round(self, target_round):
        # 해당 라운드만 필터링
        round_logs = [entry for entry in self.param_logs if entry["round"] == target_round]
        if not round_logs:
            print(f"No data found for round {target_round}.")
            return

        vectors = np.array([entry["vector"] for entry in round_logs])
        labels = [entry["type"] for entry in round_logs]

        # PCA로 2D 축소
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(vectors)

        # 색상/마커 매핑
        color_map = {
            "normal": "blue",
            "server": "green",
            "lazy": "orange",
            "noised": "red",
            "random": "purple",
            "echo": "brown",
            "small": "pink",
            "unknown": "gray"
        }
        marker_map = {
            "normal": "o",
            "server": "^",
            "lazy": "D",
            "random": "s",
            "small": "P",
            "echo": "X",
            "unknown": "x"
        }

        plt.figure(figsize=(8, 6))
        for idx, label in enumerate(labels):
            plt.scatter(
                reduced[idx, 0],
                reduced[idx, 1],
                color=color_map.get(label, "gray"),
                marker=marker_map.get(label, "o"),
                label=label if label not in plt.gca().get_legend_handles_labels()[1] else ""
            )
        
        # 밀도 덧칠 코드 (기존 scatter 아래)
        sns.kdeplot(
            x=reduced[:, 0],
            y=reduced[:, 1],
            cmap="Reds",
            fill=True,
            alpha=0.3,
            levels=20,  # 밀도 단계 조절
            thresh=0.05  # 표시할 최소 밀도
        )
        
        # Convex Hull 덧칠
        unique_labels = set(labels)
        for label in unique_labels:
            points = reduced[np.array(labels) == label]
            if len(points) >= 3:
                try:
                    hull = ConvexHull(points)
                    hull_points = np.append(hull.vertices, hull.vertices[0])
                    plt.plot(points[hull_points, 0], points[hull_points, 1], color=color_map.get(label, "gray"), alpha=0.3, linewidth=2)
                except Exception as e:
                    print(f"ConvexHull error for {label}: {e}")

        plt.title(f"Client Parameter Vectors (PCA 2D) - Round {target_round}")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend()
        plt.grid(True)
        plt.show()

    def visualize_client_vectors_3d(self):
        vectors = np.array([entry["vector"] for entry in self.param_logs])
        labels = [entry["type"] for entry in self.param_logs]

        pca = PCA(n_components=3)
        reduced = pca.fit_transform(vectors)

        color_map = {
            "normal": "blue",
            "server": "green",
            "lazy": "orange",
            "noised": "red",
            "random": "purple",
            "echo": "brown",
            "small": "pink",
            "unknown": "gray"
        }
        marker_map = {
            "normal": "o",
            "server": "^",
            "lazy": "D",
            "noised": "X",
            "random": "s",
            "echo": "P",
            "small": "h",
            "unknown": "x"
        }

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        unique_labels = set(labels)
        for label in unique_labels:
            indices = [i for i, l in enumerate(labels) if l == label]
            cluster_points = reduced[indices]
            
            # 산점도 찍기
            for idx in indices:
                ax.scatter(
                    reduced[idx, 0], reduced[idx, 1], reduced[idx, 2],
                    color=color_map.get(label, "gray"),
                    marker=marker_map.get(label, "o"),
                    label=label if label not in ax.get_legend_handles_labels()[1] else ""
                )

            # 밀도 플롯 (히트맵)
            if len(cluster_points) > 3:
                try:
                    # 스무딩 조정
                    bw = 0.2 if len(cluster_points) > 10 else 0.4
                    kde = gaussian_kde(cluster_points.T, bw_method=0.4)
                    
                    # 3D 격자 생성
                    x, y, z = np.mgrid[
                        cluster_points[:, 0].min()-10:cluster_points[:, 0].max()+10:30j,
                        cluster_points[:, 1].min()-10:cluster_points[:, 1].max()+10:30j,
                        cluster_points[:, 2].min()-10:cluster_points[:, 2].max()+10:30j
                    ]
                    coords = np.vstack([x.ravel(), y.ravel(), z.ravel()])
                    density = kde(coords).reshape(x.shape)
                    
                    cmap = cm.get_cmap('Blues_r') if label == 'random' else cm.get_cmap('Oranges')
                    
                    # 세 방향 슬라이스 출력
                    z_middle = density.shape[2] // 2
                    ax.contour3D(x[:, :, z_middle], y[:, :, z_middle], density[:, :, z_middle],
                                levels=10, cmap=cmap, alpha=1.0)

                    # XZ 슬라이스
                    y_middle = density.shape[1] // 2
                    ax.contour3D(x[:, y_middle, :], density[:, y_middle, :], z[:, y_middle, :],
                                levels=10, cmap=cmap, alpha=1.0)

                    # YZ 슬라이스
                    x_middle = density.shape[0] // 2
                    ax.contour3D(density[x_middle, :, :], y[x_middle, :, :], z[x_middle, :, :],
                                levels=10, cmap=cmap, alpha=1.0)

                    
                except Exception as e:
                    print(f"Skipping density plot for {label} due to error: {e}")

        ax.set_title("Client Parameter Vectors (PCA 3D) + Density")
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.set_zlabel("PC 3")
        ax.legend()
        plt.show()


class PunishedClientPool:
    def __init__(self, clients, partitions, consecutive_mode):
        self.clients = {str(i): c.to_client() for i, c in enumerate(clients)}  # SimulatedClient
        self.original_clients = {str(i): c for i, c in enumerate(clients)}     # 원래의 CustomClient
        self.penalties = defaultdict(int)
        self.partitions = partitions  # ✅ 요 줄이 누락되었음
        self.labels = ["normal", "lazy", "random", "echo", "small", "noised", "server"]
        self.consecutive_mode = consecutive_mode

    def get_client_by_id(self, cid):
        return self.original_clients.get(str(cid), None)

    def get_client_fn(self):
        return lambda cid: self.clients[str(cid)]

    def penalize_client(self, cid):
        self.penalties[cid] += 1

    def replace_client(self, model, cid, current_round, swtich_round=15):
        # 1. 기존 클라이언트 삭제
        if str(cid) in self.clients:
            del self.clients[str(cid)]
        if str(cid) in self.original_clients:
            del self.original_clients[str(cid)]

        # 2. 새 클라이언트 생성
        if current_round < swtich_round:
            prob_dist = ["normal"] * 6 + ["lazy", "echo", "small", "random"]
        else:
            prob_dist = ["normal", "lazy", "echo", "small", "random"]
        new_type = random.choice(prob_dist)
        new_original_client = make_client(cid, model, self.partitions[cid], new_type)  # CustomClient
        new_simulated_client = new_original_client.to_client()  # SimulatedClient로 변환

        # 3. 완전히 교체
        self.clients[str(cid)] = new_simulated_client
        self.original_clients[str(cid)] = new_original_client

        print(f"[DEBUG] After replacement, total clients: {len(self.clients)} (Simulated), {len(self.original_clients)} (Original)")

        return new_original_client

def plot_accuracy_comparison(trackers, labels):
    plt.figure(figsize=(12, 6))
    for tracker, label in zip(trackers, labels):
        df = pd.DataFrame(tracker.type_accuracies)
        if not df.empty and "accuracy" in df.columns:
            plt.plot(df["round"], df["accuracy"], label=label)
        else:
            print(f"[WARN] No 'accuracy' column found in tracker: {label}")
    plt.title("Client Type Classification Accuracy Across Strategies")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()
