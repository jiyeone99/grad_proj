
import flwr as fl
import torch
import torch.nn as nn
import numpy as np
from flwr.common import parameters_to_ndarrays
from typing import List, Dict, Tuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- FL 클라이언트들 ---
class FederatedClient(fl.client.NumPyClient):
    def __init__(self, cid, model, train_loader, test_loader, epochs=10, type="normal"):
        self.cid = cid
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_fn = nn.CrossEntropyLoss()
        self.model.to(self.device)
        self.type = type

    def get_type(self):
        return self.__class__.__name__.replace("Client", "").lower()

    def get_parameters(self, config): return [p.cpu().detach().numpy() for p in self.model.parameters()]
    
    def set_parameters(self, parameters):
        for p, new_p in zip(self.model.parameters(), parameters):
            p.data = torch.tensor(new_p, dtype=torch.float32, device=device)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        for _ in range(self.epochs):
            for x, y in self.train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = nn.CrossEntropyLoss()(self.model(x), y)
                loss.backward()
                optimizer.step()
        return self.get_parameters(config), len(self.train_loader.dataset), {"cid": self.cid, "type": self.type}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)
                loss += self.loss_fn(outputs, y).item() * x.size(0)
                predicted = outputs.argmax(1)
                correct += (predicted == y).sum().item()
                total += y.size(0)

        avg_loss = loss / total
        accuracy = correct / total

        return avg_loss, total, {"accuracy": accuracy, "cid": self.cid}  # ✅ 반드시 3개 반환


# --- 클라이언트 행동 변종 ---
class LazyClient(FederatedClient):
    def __init__(self, cid, model, train_loader, test_loader, type="lazy"):
        super().__init__(cid, model, train_loader, test_loader, epochs=5)
        self.type = type
    
    def get_type(self):
        return self.__class__.__name__.replace("Client", "").lower()

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        for _ in range(5):  # 적게 학습
            for x, y in self.train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = nn.CrossEntropyLoss()(self.model(x), y)
                loss.backward()
                optimizer.step()
        return self.get_parameters(config), len(self.train_loader.dataset), {"cid": self.cid, "type": self.type}
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(device), y.to(device)
                outputs = self.model(x)
                loss += self.loss_fn(outputs, y).item() * x.size(0)
                predicted = outputs.argmax(1)
                correct += (predicted == y).sum().item()
                total += y.size(0)

        avg_loss = loss / total
        accuracy = correct / total
        return avg_loss, total, {"accuracy": accuracy, "cid": self.cid}

class RandomClient(FederatedClient):
    def __init__(self, cid, model, train_loader, test_loader, type="random"):
        super().__init__(cid, model, train_loader, test_loader)
        self.type = type
        
    def get_type(self):
        return self.__class__.__name__.replace("Client", "").lower()

    def fit(self, parameters, config):
        shape_params = [np.random.randn(*p.shape).astype(np.float32) for p in parameters]
        return shape_params, len(self.train_loader.dataset), {"cid": self.cid, "type": self.type}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(device), y.to(device)
                outputs = self.model(x)
                loss += self.loss_fn(outputs, y).item() * x.size(0)
                predicted = outputs.argmax(1)
                correct += (predicted == y).sum().item()
                total += y.size(0)

        avg_loss = loss / total
        accuracy = correct / total
        return avg_loss, total, {"accuracy": accuracy, "cid": self.cid}
    
class EchoClient(FederatedClient):
    def __init__(self, cid, model, train_loader, test_loader, type="echo"):
        super().__init__(cid, model, train_loader, test_loader)
        self.buffer = []
        self.round = 0
        self.type = type
    def get_type(self):
        return self.__class__.__name__.replace("Client", "").lower()

    def fit(self, parameters, config):
        self.round += 1
        self.buffer.append(parameters)

        if self.round <= 2:
            print(f"[EchoClient] Round {self.round}: returning current parameters (initial phase)")
            return parameters, len(self.train_loader.dataset), {"cid": self.cid}
        else:
            echoed = self.buffer[-3]  # 2 라운드 이전의 파라미터 반환
            print(f"[EchoClient] Round {self.round}: returning parameters from round {self.round - 2}")
            return echoed, len(self.train_loader.dataset), {"cid": self.cid, "type": self.type}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(device), y.to(device)
                outputs = self.model(x)
                loss += self.loss_fn(outputs, y).item() * x.size(0)
                predicted = outputs.argmax(1)
                correct += (predicted == y).sum().item()
                total += y.size(0)

        avg_loss = loss / total
        accuracy = correct / total
        return avg_loss, total, {"accuracy": accuracy, "cid": self.cid}

class SmallClient(FederatedClient):
    def __init__(self, cid, model, train_loader, test_loader, type="small"):
        # train_loader의 데이터 수가 적어야 비정상으로 간주되도록!
        small_train_loader = torch.utils.data.DataLoader(
            list(train_loader.dataset)[:200],  # 작은 부분만 사용
            batch_size=32, shuffle=True
        )
        super().__init__(cid, model, small_train_loader, test_loader, epochs=10)
        self.type = type

    def get_type(self):
        return self.__class__.__name__.replace("Client", "").lower()

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        for _ in range(self.epochs):
            for x, y in self.train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = nn.CrossEntropyLoss()(self.model(x), y)
                loss.backward()
                optimizer.step()
        return self.get_parameters(config), len(self.train_loader.dataset), {"cid": self.cid, "type": self.type}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)
                loss += self.loss_fn(outputs, y).item() * x.size(0)
                predicted = outputs.argmax(1)
                correct += (predicted == y).sum().item()
                total += y.size(0)

        avg_loss = loss / total
        accuracy = correct / total

        return avg_loss, total, {"accuracy": accuracy, "cid": self.cid}  # ✅ 반드시 3개 반환

