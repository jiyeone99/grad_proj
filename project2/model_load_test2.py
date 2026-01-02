import torch
import numpy as np
from torch import nn

# Define the DAGMM model structure
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

# Test function for DAGMM
def test_dagmm_model(model_path, input_dim):
    try:
        # Initialize the model with the correct input_dim
        dagmm_model = DAGMM(input_dim=input_dim, latent_dim=2)
        print("Model initialized successfully.")

        # Load state dictionary
        state_dict = torch.load(model_path)
        if not isinstance(state_dict, dict):
            raise ValueError("The loaded object is not a state dictionary. Please save the model using `model.state_dict()`.")
        dagmm_model.load_state_dict(state_dict)
        dagmm_model.eval()
        print("Model weights loaded successfully.")

        # Test the model with a dummy input
        dummy_input = torch.randn(1, input_dim, dtype=torch.float32)  # Create a random input tensor
        latent, reconstructed = dagmm_model(dummy_input)
        print(f"Latent vector: {latent}")
        print(f"Reconstructed vector: {reconstructed}")

    except Exception as e:
        print(f"Error during testing DAGMM model: {e}")

# Path to the model file and input dimension
model_path = "dagmm_model.pth"
input_dim = 1055082  # Set to the original dimension used during model saving

# Run the test
test_dagmm_model(model_path, input_dim)

