import torch

state_dict = torch.load("dagmm_model.pth")
print("Encoder Weight Shape:", state_dict['encoder.4.weight'].shape)
print("Decoder Weight Shape:", state_dict['decoder.0.weight'].shape)
