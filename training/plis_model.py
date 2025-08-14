import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import torch.optim as optim
import random

# ---- Simple Model ----

class ResidualBlock(nn.Module):
    """Residual block"""

    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x: torch.Tensor):
        return self.block(x) + x


class PlisSimpleSurfacePredictor(nn.Module):

    MODEL_NAME = "simple_plis_model"

    """A lightweight baseline model that is compatible with the tuning script.
    Expects inputs from collate_fn:
        fmri: (B, T, surface_dim)
        stimulus: (B,)
    Returns:
        (B, surface_dim) prediction per trial.
    """
    def __init__(self, input_projection_size=128,
                  dropout_rate=.2, surface_dim=1056,  output_dim=None, num_layers=0,**kwargs):
        super().__init__()
        self.input_projection_size = input_projection_size
        #self.embedding_dim = embedding_dim
        #self.num_heads = num_heads
        #self.transformer_layers = transformer_layers
        self.dropout_rate = dropout_rate
        #self.fc_output_size = fc_output_size
        self.surface_dim = surface_dim
        self.output_dim = output_dim if output_dim else surface_dim
        self.num_layers = num_layers
        #self.num_stimuli = num_stimuli

        self.config = {
            'input_projection_size': input_projection_size,
            #'embedding_dim': embedding_dim,
            #'num_heads': num_heads,
            #'transformer_layers': transformer_layers,
            'dropout_rate': dropout_rate,
            'surface_dim': surface_dim,
            'output_dim': output_dim,
            'num_layers': num_layers
            #'num_stimuli': num_stimuli
        }

        # Simple linear projection per time step
        #self.input_proj = nn.Linear(surface_dim, input_projection_size)

        # Optional Transformer Encoder (ignored for mean pooling baseline)
        #encoder_layer = nn.TransformerEncoderLayer(d_model=input_projection_size, nhead=num_heads)
        #self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        # Stimulus embedding
        #self.stim_emb = nn.Embedding(num_stimuli, embedding_dim)

        # Final prediction head

        layers = [
            nn.Linear(surface_dim, input_projection_size),
            nn.LayerNorm(input_projection_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        ]

        for _ in range(num_layers):
            layers.append(
                nn.Sequential(
                    ResidualBlock(
                        nn.Sequential(
                            nn.Linear(input_projection_size, input_projection_size),
                            nn.LayerNorm(input_projection_size),
                        )
                    ),
                    nn.ReLU(),
                    nn.Dropout(p=dropout_rate),
                ),
            )

        # output block
        layers.append(
            nn.Linear(input_projection_size, self.output_dim),
        )

        self.fc = nn.Sequential(*layers)

        """
        self.fc = nn.Sequential(
            nn.Linear(surface_dim, input_projection_size),
            nn.LayerNorm(input_projection_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(input_projection_size, self.output_dim)
        )
        self.sigmoid = nn.Sigmoid()
        """
        

    @staticmethod
    def _get_stimuli_embeddings(friends_stimuli_dir, season, episode_num, episode_part):
        return 0


    @staticmethod
    def _found_hyperparams(study):
        return {
            'lr': study.best_trial.params['lr_exp'],
            'weight_decay': study.best_trial.params['weight_decay'],
            'input_projection_size': study.best_trial.params['input_projection_size'],
            'dropout_rate': study.best_trial.params['dropout_rate'],
            'num_layers': study.best_trial.params['num_layers'],
        }

    @staticmethod
    def _sample_optuna_hyperparams(trial):
        return {
                'lr': trial.suggest_float('lr_exp', 10**(-5),10**(-2),log=True),
                'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2,log=True),
                'input_projection_size': trial.suggest_int('input_projection_size', 32,512),
                'dropout_rate': trial.suggest_float('dropout_rate', .01, .9),
                'num_layers': trial.suggest_int('num_layers',0,4),
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
            'input_projection_size': random.choice([128,256,512]),
            'dropout_rate': random.uniform(.01, .25),
            'fc_output_size': random.choice([256,512,1024])
        }

    def forward(self, x: torch.Tensor, stimuli: torch.Tensor, lengths=None, introspection=False,output_temporal_attention=False,output_spatial_attention=False):
        bs, tl, fs = x.shape  # [batch_size, time_length, input_feature_size]

        fc_output = self.fc(x.view(-1, fs))
        fc_output = fc_output.view(bs, tl, -1)

        logits = fc_output.mean(1)

        if introspection:
            predictions = torch.argmax(logits, axis=-1)
            return fc_output, predictions

        if output_temporal_attention and output_spatial_attention:
            return logits, None, None
        elif output_temporal_attention and not output_spatial_attention:
            return logits, None
        elif not output_temporal_attention and output_spatial_attention:
            return logits, None
        else:
            return logits

    """
    def forward(self, fmri, stimulus, lengths):
        #fmri: (B, T, D_pad)  | lengths: (B,) actual lengths
        x = self.fc(fmri)  # (B, T, E)

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
        return x
        #stim_vec = self.stim_emb(stimulus)
        #x = torch.cat([x, stim_vec], dim=1)
#        return self.sigmoid(x)
    """
