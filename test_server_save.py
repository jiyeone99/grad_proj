import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def train_client(client_id, client_loader, save_dir, epochs=10):
    """Train a client model and save its parameters."""
    # Example model (define your model here)
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(32 * 32 * 3, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 10)
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    # Train the model
    for epoch in range(epochs):
        for images, labels in client_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Save model parameters in the specified directory for the current round
    os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists
    save_path = os.path.join(save_dir, f"client_{client_id}_params.npy")
    client_params = {name: param.data.numpy() for name, param in model.state_dict().items()}
    np.save(save_path, client_params)
    print(f"[INFO] Saved client {client_id} parameters to {save_path}")
    return client_params

def aggregate_parameters(all_client_params):
    """Aggregate parameters from all clients (FedAvg)."""
    aggregated_params = {}
    for key in all_client_params[0].keys():
        aggregated_params[key] = np.mean([params[key] for params in all_client_params], axis=0)
    return aggregated_params

def main(
        num_clients,
        train_epochs,
        client_save_dir,
        server_save_dir,
        num
        ):
    # Create save directories
    save_dir = os.path.join(client_save_dir, f"{num}/")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(server_save_dir, exist_ok=True)

    # Load CIFAR-10 dataset
    transform = transforms.Compose([transforms.ToTensor()])
    full_dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)

    # Split dataset randomly for clients
    total_size = len(full_dataset)
    client_sizes = [total_size // num_clients] * num_clients
    leftover = total_size % num_clients
    for i in range(leftover):
        client_sizes[i] += 1
    client_datasets = random_split(full_dataset, client_sizes)

    # Train clients and collect parameters
    all_client_params = []
    for client_id, client_dataset in enumerate(client_datasets):
        client_loader = DataLoader(client_dataset, batch_size=32, shuffle=True)
        print(f"[INFO] Training client {client_id + 1}/{num_clients}...")
        client_params = train_client(client_id, client_loader, save_dir, epochs=train_epochs)
        all_client_params.append(client_params)

    # Aggregate parameters and save
    print("[INFO] Aggregating parameters...")
    aggregated_params = aggregate_parameters(all_client_params)
    server_save_path = os.path.join(server_save_dir, f"server_params{num}.npy")
    np.save(server_save_path, aggregated_params)
    print(f"[INFO] Server parameters saved to {server_save_path}")

if __name__ == "__main__":
    num = 41
    while (num <= 100):
        main(num_clients=8,
             train_epochs=10,
             client_save_dir="./test_parameters/server_parameters_npy/client_parameters/",
             server_save_dir="./test_parameters/server_parameters_npy",
             num=num)
        num += 1
