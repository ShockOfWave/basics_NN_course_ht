from pathlib import Path
import torch


def get_project_root() -> Path:
    return Path(__file__).parent.parent


config = {
        'epochs': 100,
        'learning_rate_encoder': 1e-2,
        'learning_rate_decoder': 1e-2,
        'learning_rate_discriminator': 1e-2,
        'alpha': 0.1,
        'gamma': 15,
        'batch_size': 256,
        'latent_dim': 128
    }
