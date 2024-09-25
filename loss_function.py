import torch
from torch import nn

class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()

    def forward(self, model_output, targets):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, _ = model_output
        gate_out = gate_out.view(-1, 1)

        # Mel loss
        mel_loss = nn.MSELoss()(mel_out, mel_target) + nn.MSELoss()(mel_out_postnet, mel_target)

        # Gate loss
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target.float())

        return mel_loss + gate_loss
    
class UnitYLoss(nn.Module):
    def __init__(self):
        super(UnitYLoss, self).__init__()

    def forward(self, model_output, targets):
        unit_target, gate_target = targets[0], targets[1]
        unit_target.requires_grad = False
        gate_target.requires_grad = False

        # Extract dimensions from model output
        batch_size, seq_len, num_features = model_output.size()

        # Reshape gate_out for loss calculation
        gate_out = model_output[:, :, -1].reshape(batch_size * seq_len, 1)
        gate_target = gate_target.view(-1)
        expected_gate_size = gate_out.size(0)
        actual_gate_size = gate_target.size(0)

        if actual_gate_size > expected_gate_size:
            gate_target = gate_target[:expected_gate_size]
        elif actual_gate_size < expected_gate_size:
            repeat_factor = expected_gate_size // actual_gate_size
            remaining_size = expected_gate_size % actual_gate_size
            gate_target = torch.cat([gate_target.repeat(repeat_factor), gate_target[:remaining_size]])

        gate_target = gate_target.reshape_as(gate_out)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target.float())

        # Flatten model output for unit loss (excluding the gate output)
        unit_out = model_output[:, :, :-1].reshape(batch_size * seq_len, num_features - 1)

        # Debugging print statements
        print(f"unit_out shape: {unit_out.shape}")
        print(f"Original unit_target shape: {unit_target.shape}")
        print(f"Number of elements in unit_out: {unit_out.numel()}")
        print(f"Number of elements in unit_target: {unit_target.numel()}")

        # Calculate the required number of elements
        required_elements = unit_out.numel()
        available_elements = unit_target.numel()

        # Adjust unit_target size to match unit_out
        if available_elements > required_elements:
            # Truncate if unit_target has more elements
            unit_target = unit_target[:required_elements]
        elif available_elements < required_elements:
            # Pad with zeros if unit_target has fewer elements
            padding = required_elements - available_elements
            padding_tensor = torch.zeros(padding, device=unit_target.device)
            unit_target = torch.cat([unit_target.view(-1), padding_tensor]).view_as(unit_out)

        # Try to reshape unit_target to match unit_out
        try:
            unit_target = unit_target.reshape_as(unit_out)
        except RuntimeError as e:
            print(f"Error reshaping unit_target: {e}")
            print(f"unit_out shape: {unit_out.shape}, unit_target shape: {unit_target.shape}")
            raise e

        print(f"unit_target shape after reshape: {unit_target.shape}")

        # Calculate unit loss
        unit_loss = nn.MSELoss()(unit_out, unit_target)

        # Return combined loss
        return unit_loss + gate_loss