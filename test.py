import torch
from model import Tacotron2
from hparams import create_hparams
from text import text_to_sequence  # Ensure this import is correct for your setup
import numpy as np
# Load the hyperparameters
hparams = create_hparams()

# Modify the checkpoint path to point to your saved checkpoint
checkpoint_path = r"E:\Rapid Innovation\Tacotron\tacotron2\outdir\checkpoint_epoch_42"

# Create the model and load the state dictionary
model = Tacotron2(hparams).cuda()
checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint_dict['state_dict'])

# Set the model to evaluation mode
model.eval()

# Function to load data for inference
debug = False  # Set to True to enable debug prints

# Function to load data for inference
def load_data(text, text_cleaners):
    text_norm = torch.IntTensor(text_to_sequence(text, text_cleaners))
    text_norm = text_norm.unsqueeze(0).cuda()  # Add batch dimension
    input_lengths = torch.IntTensor([text_norm.size(1)]).cuda()  # Length of the input text sequence
    b = text_norm.cpu().numpy()
    np.save("sample1.npy",b)
    if debug:
        print(f"text_norm: {text_norm}")
        print(f"text_norm shape: {text_norm.shape}")
    return text_norm, input_lengths


# Example input text
input_text = "printing in the only sense with which we are at present concerned differs from most if not from all the Arts and Crafts represented in the exhibition"

# Load input data
input_data, input_lengths = load_data(input_text, hparams.text_cleaners)

print(input_data)

# Inference
with torch.no_grad():
    embedded_inputs = model.embedding(input_data).transpose(1, 2)
    encoder_outputs = model.encoder(embedded_inputs, input_lengths)
    mel_outputs, gate_outputs, alignments = model.decoder.inference(encoder_outputs)
    mel_outputs_postnet = model.postnet(mel_outputs)
    mel_outputs_postnet = mel_outputs + mel_outputs_postnet

print(mel_outputs_postnet)
print(mel_outputs_postnet.shape)
mc = mel_outputs_postnet.cpu()
a = mc.numpy()
np.save("sample.npy",a)

# The mel_outputs_postnet can be converted to audio using a vocoder like WaveGlow or Griffin-Lim algorithm
print("Inference completed successfully.")
