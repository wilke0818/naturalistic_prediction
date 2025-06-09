import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import torch.optim as optim
import random

# ---- Simple Model ----
class SimpleSurfacePredictor(nn.Module):

    MODEL_NAME = "simple_model"

    """A lightweight baseline model that is compatible with the tuning script.
    Expects inputs from collate_fn:
        fmri: (B, T, surface_dim)
        stimulus: (B,)
    Returns:
        (B, surface_dim) prediction per trial.
    """
    def __init__(self, aggregation_method='mean', input_projection_size=128,embedding_dim=32, num_heads=2, transformer_layers=2,
                 fc_output_size=512, surface_dim=1056, num_stimuli=10, **kwargs):
        super().__init__()
        self.aggregation_method = aggregation_method
        self.input_projection_size = input_projection_size
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.transformer_layers = transformer_layers
        self.fc_output_size = fc_output_size
        self.surface_dim = surface_dim
        self.num_stimuli = num_stimuli

        self.config = {
            'aggregation_method': aggregation_method,
            'input_projection_size': input_projection_size,
            'embedding_dim': embedding_dim,
            'num_heads': num_heads,
            'transformer_layers': transformer_layers,
            'fc_output_size': fc_output_size,
            'surface_dim': surface_dim,
            'num_stimuli': num_stimuli
        }

        # Simple linear projection per time step
        self.input_proj = nn.Linear(surface_dim, input_projection_size)

        # Optional Transformer Encoder (ignored for mean pooling baseline)
        #encoder_layer = nn.TransformerEncoderLayer(d_model=input_projection_size, nhead=num_heads)
        #self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        # Stimulus embedding
        #self.stim_emb = nn.Embedding(num_stimuli, embedding_dim)

        # Final prediction head
        self.fc = nn.Sequential(
            nn.Linear(input_projection_size, fc_output_size),
            nn.ReLU(),
            nn.Linear(fc_output_size, surface_dim),
            nn.Sigmoid()
        )

        

    @staticmethod
    def _get_stimuli_embeddings(friends_stimuli_dir, season, episode_num, episode_part):
        return 0
    
    
    @staticmethod
    def _sample_optuna_hyperparams(trial):
        return {
                'aggregation_method': trial.suggest_categorical('aggregation_method', ['mean']),
                'lr': trial.suggest_float('lr_exp', 10**(-5), 10**(-3)),
                'embedding_dim': trial.suggest_categorical('embedding_dim', [16]),
                'num_heads': trial.suggest_categorical('num_heads', [2]),
                'transformer_layers': trial.suggest_categorical('transformer_layers', [2]),
                'weight_decay': trial.suggest_categorical('weight_decay', [0.0, 1e-5, 1e-4]),
                'input_projection_size': trial.suggest_categorical('input_projection_size', [128,256,512]),
                'dropout_rate': trial.suggest_float('dropout_rate', .01, .25),
                'fc_output_size': trial.suggest_categorical('fc_output_size', [256,512, 1024]),
            }
    
        
    @staticmethod
    def _sample_hyperparams():
        return {
            'aggregation_method': 'mean',
            'lr': 10 ** random.uniform(-5, -3),
            'embedding_dim': random.choice([16, 32, 64]),
            'num_heads': random.choice([2, 4, 8]),
            'transformer_layers': random.choice([2, 4, 6]),
            'weight_decay': random.choice([0.0, 1e-5, 1e-4]),
            'input_projection_size': random.choice([128,256]),
            'fc_output_size': random.choice([256,512,1024])
        }


    def forward(self, fmri, stimulus, lengths):
        """fmri: (B, T, D_pad)  | lengths: (B,) actual lengths"""
        x = self.input_proj(fmri)  # (B, T, E)

        # Create mask where True = padded element
        max_T = fmri.size(1)
        mask = torch.arange(max_T, device=lengths.device).expand(len(lengths), max_T) >= lengths.unsqueeze(1)

        if self.aggregation_method == 'attention':
            x = self.transformer(x, src_key_padding_mask=mask)  # (B, T, E)
            # masked mean
            mask_f = (~mask).float().unsqueeze(2)  # (B,T,1)
            x = (x * mask_f).sum(dim=1) / lengths.unsqueeze(1)
        else:  # simple mean with masking
            mask_f = (~mask).float().unsqueeze(2)
            x = (x * mask_f).sum(dim=1) / lengths.unsqueeze(1)

        #stim_vec = self.stim_emb(stimulus)
        #x = torch.cat([x, stim_vec], dim=1)
        return self.fc(x)
