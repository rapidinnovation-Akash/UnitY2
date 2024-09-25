import torch
class HParams:
    def __init__(self):
        ##############################
        # Data Parameters
        ##############################
        self.training_files = 'E:/Rapid Innovation/Tacotron/tacotron2/filelists/ljs_audio_text_train_filelist.txt'
        self.validation_files = 'E:/Rapid Innovation/Tacotron/tacotron2/filelists/ljs_audio_text_val_filelist.txt'
        self.text_cleaners = ['english_cleaners']
        self.max_wav_value = 32768.0
        self.sampling_rate = 22050
        self.filter_length = 1024
        self.hop_length = 256
        self.win_length = 1024
        self.n_mel_channels = 80
        self.mel_fmin = 0.0
        self.mel_fmax = 8000.0
        self.load_unit_from_disk = True
        self.load_mel_from_disk = False
        self.hidden_dim = 512  # Add this line to define hidden_dim
        # Number of frames in the input mel-spectrogram
        self.input_dim = self.n_mel_channels  # Number of mel channels
        
        ##############################
        # Model Parameters
        ##############################
        self.model_dim = 512  # Dimension of the model (used in Conformer and others)
        self.conv_dim = 256  # Dimension of the convolutional layers in the Conformer block
        self.n_heads = 8  # Number of attention heads
        self.n_layers_enc = 6  # Number of encoder layers (Conformer)
        self.n_layers_dec = 6  # Number of layers in the first-pass decoder
        self.n_layers_t2u = 4  # Number of layers in the T2U encoder
        self.n_layers_unit_decoder = 4  # Number of layers in the unit decoder
        self.ff_dim = 2048  # Feed-forward network dimension
        self.dropout = 0.1  # Dropout probability
        
        # Embeddings and Vocabularies
        self.vocab_size = 1000  # Vocabulary size for text decoder
        self.unit_vocab_size = 1000  # Vocabulary size for unit decoder
        
        # Audio/Vocoder Parameters
        self.n_frames_per_step = 1  # Number of frames processed per step
        self.unit_denoise_strength = 0.01  # Denoising strength for vocoder

        ##############################
        # Optimization Parameters
        ##############################
        self.learning_rate = 0.001  # Initial learning rate
        self.weight_decay = 1e-6  # Weight decay for optimizer
        self.batch_size = 32  # Batch size
        self.epochs = 50  # Number of training epochs
        self.grad_clip_thresh = 1.0  # Gradient clipping threshold
        self.fp16_run = False  # Enable mixed precision (FP16) training
        self.iters_per_checkpoint = 500  # Iterations per checkpoint

        ##############################
        # Distributed Training Parameters
        ##############################
        self.distributed_run = False  # Enable distributed training
        self.dist_backend = 'nccl'  # Backend for distributed training
        self.dist_url = 'tcp://localhost:54321'  # URL for distributed training
        self.seed = 1234  # Random seed for reproducibility

        ##############################
        # Beam Search Parameters
        ##############################
        self.beam_size = 5  # Beam size for beam search
        self.max_decoder_steps = 1000  # Maximum steps for the decoder
        self.gate_threshold = 0.5  # Gate threshold for decoder in beam search

        ##############################
        # Logging and Checkpointing
        ##############################
        self.log_directory = 'logs'  # Directory for saving logs and checkpoints
        self.cudnn_enabled = True  # Enable cuDNN optimizations
        self.cudnn_benchmark = False  # Enable cuDNN benchmarking for performance

        ##############################
        # Inference Parameters
        ##############################
        self.max_len = 100  # Maximum length for inference sequences
        self.start_token_id_text = 0  # Start token for text decoding
        self.start_token_id_unit = 0  # Start token for unit decoding
        self.end_token_id = 1  # End token for both decoders
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  # Automatically set the device


    def create_hparams():
        return HParams()