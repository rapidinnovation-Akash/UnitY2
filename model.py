import torch
import torchaudio
from torch import nn
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram
from hparams import HParams  # Importing HParams class
from waveglow.denoiser import Denoiser
import sys
sys.path.append('E:/Rapid Innovation/Tacotron/tacotron2/waveglow')
hparams = HParams()
import math
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Helper function to process audio into mel-spectrograms
class AudioProcessor:
    def __init__(self, hparams):
        self.sampling_rate = hparams.sampling_rate
        self.mel_transform = MelSpectrogram(
            sample_rate=hparams.sampling_rate,
            n_fft=hparams.filter_length,
            hop_length=hparams.hop_length,
            n_mels=hparams.n_mel_channels,
            win_length=hparams.win_length
        )

    def process(self, audio_path):
        waveform, sr = torchaudio.load(audio_path)
        if sr != self.sampling_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sampling_rate)(waveform)
        mel_spectrogram = self.mel_transform(waveform)
        
        print(f"Mel-spectrogram shape: {mel_spectrogram.shape}")  # Debugging print statement
        
        return mel_spectrogram.squeeze(0)  # Convert to 2D tensor (time, n_mels)


class ConformerBlock(nn.Module):
    def __init__(self, model_dim, num_heads, ff_dim, conv_dim, dropout=0.1):
        super(ConformerBlock, self).__init__()
        
        # Multi-head Self Attention
        self.self_attn = nn.MultiheadAttention(model_dim, num_heads, dropout=dropout, batch_first=True)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, model_dim)
        )
        
        # Convolutional layers
        self.conv = nn.Sequential(
            nn.Conv1d(model_dim, conv_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_dim),
            nn.ReLU(),
            nn.Conv1d(conv_dim, model_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(model_dim)
        )
        
        # Layer Normalization
        self.layer_norm1 = nn.LayerNorm(model_dim)
        self.layer_norm2 = nn.LayerNorm(model_dim)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Debugging prints for shapes
        print(f"Input shape to conformer block: {x.shape}")
        if mask is not None:
            print(f"Mask shape: {mask.shape}")

        # Self-attention
        attn_output, _ = self.self_attn(x, x, x, attn_mask=mask)

        # Ensure shapes are correct after each operation
        x = self.layer_norm1(x + self.dropout(attn_output))
        print(f"Shape after self-attention and layer norm: {x.shape}")

        # Convolution Module
        conv_output = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.layer_norm2(x + self.dropout(conv_output))
        print(f"Shape after convolution and layer norm: {x.shape}")

        # Feed Forward Network
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        print(f"Shape after feed-forward network: {x.shape}")

        return x


# class MultiHeadAttentionModule(nn.Module):
#     def __init__(self, model_dim, n_heads):
#         super(MultiHeadAttentionModule, self).__init__()
#         self.model_dim = model_dim
#         self.n_heads = n_heads
#         self.head_dim = model_dim // n_heads

#         assert self.head_dim * n_heads == model_dim, "model_dim must be divisible by n_heads"

#         self.multihead_attn = nn.MultiheadAttention(embed_dim=model_dim, num_heads=n_heads)

#     def forward(self, x, memory):
#         # x: (seq_len, batch_size, model_dim)
#         # memory: (seq_len, batch_size, model_dim)

#         # Transpose for MultiheadAttention, which expects (seq_len, batch_size, model_dim)
#         x = x.transpose(0, 1)  # Now (batch_size, seq_len, model_dim)

#         # Apply multi-head attention

#         attn_output, attn_weights = self.multihead_attn(x, memory, memory)
#         return attn_output.transpose(0, 1)  # Return to original shape (seq_len, batch_size, model_dim)

# Positional Encoding for Transformers
# Simplified Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.model_dim = model_dim

        # Create the position encodings
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(1))  # Shape: (max_len, 1, model_dim)

    def forward(self, x):
        seq_len = x.size(0)
        x = x + self.pe[:seq_len, :].to(x.device)
        return x

class SpeechEncoderConformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_layers=6, num_heads=8):
        super(SpeechEncoderConformer, self).__init__()
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(model_dim=model_dim, num_heads=num_heads, ff_dim=1024, conv_dim=256, dropout=0.1)
            for _ in range(num_layers)
        ])
        self.input_projection = nn.Linear(input_dim, model_dim)
        self.positional_encoding = PositionalEncoding(model_dim)
        self.num_heads = num_heads

    def forward(self, x, input_lengths):
        print(f"x shape before projection: {x.shape}")

        x = x.float().to(x.device)

        # Ensure input is properly shaped for projection
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # Shape: (batch_size, seq_len, 1)
            x = x.expand(-1, -1, self.input_projection.in_features)  # Shape: (batch_size, seq_len, input_dim)

        if x.shape[-1] != self.input_projection.in_features:
            raise ValueError(f"Expected input with {self.input_projection.in_features} features, but got {x.shape[-1]} features")

        # Reshape to (batch_size * seq_len, input_dim) for projection
        x = x.view(-1, self.input_projection.in_features)
        x = self.input_projection(x)
        
        # Reshape back to (seq_len, batch_size, model_dim)
        x = x.view(-1, input_lengths.size(0), self.input_projection.out_features)
        print(f"x shape after projection: {x.shape}")

        # Apply positional encoding
        x = self.positional_encoding(x)
        print(f"x shape after positional encoding: {x.shape}")

        input_lengths = input_lengths.to(x.device)

        # Updated based on your expected sequence length (32)
        max_seq_len = 32  # Ensure this matches the model's expected seq_len
        batch_size = x.size(1)
        num_heads = self.num_heads

        # If input_lengths indicate a longer sequence, truncate/pad to max_seq_len
        if x.size(0) > max_seq_len:
            x = x[:max_seq_len]  # Truncate sequences to max_seq_len
        elif x.size(0) < max_seq_len:
            padding = torch.zeros((max_seq_len - x.size(0), x.size(1), x.size(2)), device=x.device)
            x = torch.cat([x, padding], dim=0)  # Pad sequences to max_seq_len

        print(f"x shape after adjustment for seq_len: {x.shape}")

        # Create a mask based on the adjusted seq_len
        mask = (torch.arange(max_seq_len, device=x.device).expand(batch_size, max_seq_len) >= input_lengths.unsqueeze(1)).unsqueeze(1).unsqueeze(2)
        print(f"Mask shape after unsqueeze: {mask.shape}")  # Should be (batch_size, 1, 1, seq_len)

        # Correct the expansion for the attention heads
        mask = mask.expand(batch_size, num_heads, max_seq_len, max_seq_len)  # (batch_size, num_heads, seq_len, seq_len)
        mask = mask.reshape(batch_size * num_heads, max_seq_len, max_seq_len)  # (batch_size * num_heads, seq_len, seq_len)

        print(f"Mask shape after expanding for attention heads: {mask.shape}")

        for block in self.conformer_blocks:
            x = block(x.transpose(0, 1), mask=mask)  # Transpose x to (batch_size, seq_len, model_dim) for batch_first=True
            print(f"x shape after conformer block: {x.shape}")

        return x.transpose(0, 1)  # (seq_len, batch_size, model_dim)

# Text Decoder (First Pass)
class TextDecoderFirstPass(nn.Module):
    def __init__(self, model_dim, vocab_size, num_layers=6):
        super(TextDecoderFirstPass, self).__init__()
        self.model_dim = model_dim
        self.vocab_size = vocab_size
        
        # Define transformer decoder layers
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=model_dim, nhead=8, dim_feedforward=1024, dropout=0.1)
            for _ in range(num_layers)
        ])
        
        # Embedding and output projection
        self.embedding = nn.Embedding(vocab_size, model_dim)
        self.positional_encoding = PositionalEncoding(model_dim)
        self.output_projection = nn.Linear(model_dim, vocab_size)

    def forward(self, encoded_speech, target_subwords=None, tgt_mask=None):
        if target_subwords is not None:
            target_subwords = target_subwords.long()

            if target_subwords.dim() == 3:
                target_subwords = target_subwords.float().mean(dim=-1)
                target_subwords = target_subwords.round().long()

            target_subwords = torch.clamp(target_subwords, min=0, max=self.vocab_size - 1)
            tgt = self.embedding(target_subwords)
            tgt = self.positional_encoding(tgt.transpose(0, 1))  # (seq_len, batch_size, model_dim)

            # Ensure that the sequence lengths match for tgt and encoded_speech
            tgt_seq_len = tgt.size(0)
            mem_seq_len = encoded_speech.size(0)

            if tgt_seq_len != mem_seq_len:
                if tgt_seq_len > mem_seq_len:
                    tgt = tgt[:mem_seq_len, :, :]
                else:
                    padding = torch.zeros(mem_seq_len - tgt_seq_len, tgt.size(1), tgt.size(2)).to(tgt.device)
                    tgt = torch.cat((tgt, padding), dim=0)

            for i, layer in enumerate(self.decoder_layers):
                tgt = layer(tgt, encoded_speech, tgt_mask=tgt_mask)

            logits = self.output_projection(tgt)
            return logits.transpose(0, 1)  # (batch_size, seq_len, vocab_size)

class T2UEncoder(nn.Module):
    def __init__(self, hparams, num_layers=6):
        super(T2UEncoder, self).__init__()
        # Adjust the input projection to match the actual input dimension
        self.input_projection = nn.Linear(1000, hparams.model_dim)
        self.positional_encoding = PositionalEncoding(hparams.model_dim)
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(
                model_dim=hparams.model_dim, 
                num_heads=hparams.n_heads, 
                ff_dim=hparams.ff_dim, 
                conv_dim=hparams.conv_dim,
                dropout=hparams.dropout
            ) 
            for _ in range(num_layers)
        ])

    def forward(self, text_representation):
        print(f"Shape before input_projection: {text_representation.shape}")
        
        # Apply the input projection
        text_representation = self.input_projection(text_representation)

        # Apply positional encoding
        text_representation = self.positional_encoding(text_representation)

        # Pass through conformer blocks
        for conformer_block in self.conformer_blocks:
            text_representation = conformer_block(text_representation)

        return text_representation


# Unit Decoder (Second Pass)
class UnitDecoderSecondPass(nn.Module):
    def __init__(self, hparams):
        super(UnitDecoderSecondPass, self).__init__()
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=hparams.model_dim, nhead=hparams.n_heads, dim_feedforward=hparams.ff_dim, dropout=hparams.dropout)
            for _ in range(hparams.n_layers_unit_decoder)
        ])
        self.embedding = nn.Embedding(hparams.unit_vocab_size, hparams.model_dim)
        self.output_projection = nn.Linear(hparams.model_dim, hparams.unit_vocab_size)
        self.positional_encoding = PositionalEncoding(hparams.model_dim)

    def forward(self, t2u_output, target_units=None, tgt_mask=None):
        tgt = self.embedding(target_units)
        tgt = self.positional_encoding(tgt.transpose(0, 1))  # (seq_len, batch_size, dim)

        for layer in self.decoder_layers:
            tgt = layer(tgt, t2u_output, tgt_mask=tgt_mask)

        logits = self.output_projection(tgt)
        return logits.transpose(0, 1)  # (batch_size, seq_len, num_units)


# Placeholder for unit vocoder (implement your vocoder here)
def load_waveglow(waveglow_path):
    waveglow = torch.load(waveglow_path)['model']
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow.cuda().eval()
try:
    waveglow = load_waveglow('E:/Rapid Innovation/Tacotron/tacotron2/waveglow/waveglow_256channels_universal_v5.pt')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

# Define the unit_vocoder function
def unit_vocoder(mel_spectrogram, denoise=True):
    # Ensure input is on the same device as WaveGlow
    mel_spectrogram = mel_spectrogram.cuda()

    # Generate waveform using WaveGlow
    with torch.no_grad():
        audio = waveglow.infer(mel_spectrogram)

    # Optionally apply denoiser
    if denoise:
        denoiser = Denoiser(waveglow).cuda()
        audio = denoiser(audio, strength=0.01)  # Tune strength as necessary

    return audio

# GLAT Unit Decoder (Second Pass)
class GLATUnitDecoder(nn.Module):
    def __init__(self, hparams):
        super(GLATUnitDecoder, self).__init__()
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=hparams.model_dim, nhead=hparams.n_heads, dim_feedforward=hparams.ff_dim, dropout=hparams.dropout)
            for _ in range(hparams.n_layers_unit_decoder)
        ])
        self.embedding = nn.Embedding(hparams.unit_vocab_size, hparams.model_dim)
        self.output_projection = nn.Linear(hparams.model_dim, hparams.unit_vocab_size)
        self.positional_encoding = PositionalEncoding(hparams.model_dim)

    def forward(self, t2u_output, target_units=None, tgt_mask=None, alpha=0.5):
        if target_units is not None:
            # Debugging: Print the initial shape of target_units
            print(f"Initial target_units shape: {target_units.shape}")

            # Handling different shapes
            if target_units.dim() == 1:
                # If target_units is 1D, reshape it to 2D (batch_size, seq_len)
                print("1D target_units detected, reshaping...")
                target_units = target_units.unsqueeze(0)
            elif target_units.dim() == 3:
                # If 3D, average across the third dimension
                print("3D target_units detected, averaging...")
                target_units = target_units.float().mean(dim=-1).round().long()

            # Re-check shape after processing
            print(f"Processed target_units shape: {target_units.shape}")

            # Ensure it is a 2D tensor (batch_size, seq_len)
            if target_units.dim() != 2:
                raise ValueError(f"Expected target_units to be 2D, but got {target_units.dim()}D tensor")

            batch_size, seq_len = target_units.shape

            # Continue with the GLAT logic
            tgt = self.embedding(target_units)
            tgt = self.positional_encoding(tgt.transpose(0, 1))  # (seq_len, batch_size, model_dim)

            # GLAT logic
            glance_mask = (torch.rand(batch_size, seq_len) < alpha).long().to(target_units.device)
            tgt_glance = tgt * glance_mask.unsqueeze(-1)  # Masked target embeddings
            
            for layer in self.decoder_layers:
                tgt = layer(tgt_glance, t2u_output, tgt_mask=tgt_mask)

            logits = self.output_projection(tgt)
            return logits.transpose(0, 1)  # (batch_size, seq_len, num_units)
        else:
            # Inference mode
            tgt = torch.zeros(t2u_output.size(0), 1, dtype=torch.long, device=t2u_output.device)
            tgt = self.embedding(tgt)
            tgt = self.positional_encoding(tgt.transpose(0, 1))

            for layer in self.decoder_layers:
                tgt = layer(tgt, t2u_output, tgt_mask=tgt_mask)

            logits = self.output_projection(tgt)
            return logits.transpose(0, 1)


# Main UnitY Model
class UnitY(nn.Module):
    def __init__(self, hparams):
        super(UnitY, self).__init__()
        self.speech_encoder = SpeechEncoderConformer(
            input_dim=hparams.n_mel_channels,
            model_dim=hparams.model_dim,
            num_layers=hparams.n_layers_enc
        )
        self.text_decoder = TextDecoderFirstPass(
            model_dim=hparams.model_dim, 
            vocab_size=hparams.vocab_size
        )
        self.t2u_encoder = T2UEncoder(hparams)
        self.unit_decoder = GLATUnitDecoder(hparams)
        
        # New intermediate projection layer to ensure dimensionality matches
        self.intermediate_projection = nn.Linear(1000, hparams.model_dim)  # Adjust 1000 to match actual intermediate size
        self.output_projection = nn.Linear(hparams.model_dim, 858)  # Adjusted to match target size
        
        self.audio_processor = AudioProcessor(hparams)

    def forward(self, src, src_lengths, tgt_text, tgt_text_lengths=None, tgt_unit=None, tgt_unit_lengths=None):
        encoded_speech = self.speech_encoder(src, src_lengths)
        text_outputs = self.text_decoder(encoded_speech, tgt_text)
        t2u_outputs = self.t2u_encoder(text_outputs)
        unit_outputs = self.unit_decoder(t2u_outputs, tgt_unit)
        
        # Apply the intermediate projection to match dimensions
        unit_outputs = self.intermediate_projection(unit_outputs)
        unit_outputs = self.output_projection(unit_outputs)  # Ensure output size matches the target size
        
        return text_outputs, unit_outputs

    def inference(self, audio_paths, hparams):
        mel_spectrograms = [self.audio_processor.process(audio_path) for audio_path in audio_paths]
        mel_spectrograms = nn.utils.rnn.pad_sequence(mel_spectrograms, batch_first=True)
        input_lengths = torch.tensor([mel.size(1) for mel in mel_spectrograms])

        encoded_speech = self.speech_encoder(mel_spectrograms, input_lengths)

        subword_predictions = self.two_pass_beam_search(self, encoded_speech, hparams.start_token_id_text, hparams.start_token_id_unit, hparams.end_token_id, hparams.max_len, hparams.B1st, hparams.B2nd, hparams.device)
        
        return subword_predictions

    def beam_search(self, decoder, memory, start_token_id, end_token_id, max_len, beam_size, vocab_size, device):
        beams = [(torch.tensor([start_token_id], device=device).unsqueeze(0), 0.0)]

        for _ in range(max_len):
            new_beams = []
            
            for beam, score in beams:
                if beam[0, -1] == end_token_id:
                    new_beams.append((beam, score))
                    continue
                
                output = decoder(beam, memory)
                log_probs = torch.log_softmax(output[:, -1, :], dim=-1)
                topk_log_probs, topk_ids = log_probs.topk(beam_size)
                
                for i in range(beam_size):
                    new_beam = torch.cat([beam, topk_ids[:, i].unsqueeze(0)], dim=1)
                    new_score = score + topk_log_probs[:, i].item()
                    new_beams.append((new_beam, new_score))
            
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
        
        return beams

    def two_pass_beam_search(self, model, memory, start_token_id_text, start_token_id_unit, end_token_id, max_len, B1st, B2nd, device):
        model.eval()

        with torch.no_grad():
            first_pass_beams = self.beam_search(model.text_decoder, memory, start_token_id_text, end_token_id, max_len, B1st, model.vocab_size, device)
            best_first_pass_beam = max(first_pass_beams, key=lambda x: x[1])
            Y_hat = best_first_pass_beam[0]
            
            D_text = model.text_decoder.embedding(Y_hat)
            Z = model.t2u_encoder(D_text)
            
            second_pass_beams = self.beam_search(model.unit_decoder, Z, start_token_id_unit, end_token_id, max_len, B2nd, model.unit_vocab_size, device)
            best_second_pass_beam = max(second_pass_beams, key=lambda x: x[1])
            U_hat = best_second_pass_beam[0]
            
            W_hat = unit_vocoder(U_hat)
            
            return W_hat
    
    def parse_batch(self, batch):
        src = batch[0].cuda()
        src_lengths = batch[1].cuda()
        tgt_text = batch[2].cuda()

        tgt_text_lengths = batch[3].cuda() if batch[3].numel() > 0 else None

        if len(batch) > 4 and batch[4].numel() > 0:
            tgt_unit = batch[4].cuda()
            tgt_unit_lengths = batch[4].cuda()
            return src, src_lengths, tgt_text, tgt_text_lengths, tgt_unit, tgt_unit_lengths
        else:
            return src, src_lengths, tgt_text, tgt_text_lengths