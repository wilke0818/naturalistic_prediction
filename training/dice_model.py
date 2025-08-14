# pylint: disable=invalid-name, no-member, missing-function-docstring, too-many-branches, too-few-public-methods, unused-argument
""" DICE model from https://github.com/UsmanMahmood27/DICE """
from random import uniform, randint

import torch
from torch import nn
from torch import optim

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class DICESurfacePredictor(nn.Module):
    """
    DICE model for fMRI data.
    Expected input shape: [batch_size, time_length, input_feature_size].
    Output: [batch_size, n_classes]
    """

    MODEL_NAME = "dice_model"

    @staticmethod
    def _sample_optuna_hyperparams(trial):
        lstm_num_layers = trial.suggest_int('lstm_num_layers', 1,3)
        return {
                'lr': trial.suggest_float('lr_exp', 10**(-5), 10**(-3),log=True),
                'lstm_hidden_size': trial.suggest_int('lstm_hidden_size', 32,128), #20 low
                'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-1,log=True),
                'lstm_num_layers': lstm_num_layers,#trial.suggest_int('lstm_num_layers', 1,2),#1 low
                'attn_heads': trial.suggest_int('attn_heads', 1,4),#1 low
                'head_dim': trial.suggest_int('head_dim', 16,64),#16 low
                'attn_dropout': trial.suggest_float('attn_dropout', 0, .3),
                'clf_num_layers': trial.suggest_int('clf_num_layers', 0,3),#0 low
                'clf_hidden_size': trial.suggest_int('clf_hidden_size', 16,512),#128 low
                'lstm_dropout': trial.suggest_float('lstm_dropout', 0,.5) if lstm_num_layers > 1 else 0
            }

    @staticmethod
    def _found_hyperparams(study):
        num_lstm_layers = study.best_trial.params['lstm_num_layers']
        return {
            'lr': study.best_trial.params['lr_exp'],
            'lstm_hidden_size': study.best_trial.params['lstm_hidden_size'],
            'lstm_num_layers': num_lstm_layers,
            'attn_heads': study.best_trial.params['attn_heads'],
            'head_dim': study.best_trial.params['head_dim'],
            'attn_dropout': study.best_trial.params['attn_dropout'],
            'clf_num_layers': study.best_trial.params['clf_num_layers'],
            'clf_hidden_size': study.best_trial.params['clf_hidden_size'],
            'weight_decay': study.best_trial.params['weight_decay'],
            'lstm_dropout': study.best_trial.params['lstm_dropout'] if num_lstm_layers > 1 else 0,
        }

    def __init__(self, surface_dim=1056, output_dim=None, lstm_hidden_size=50, lstm_num_layers=1, bidirectional=True,
                 attn_heads=2, head_dim=48, attn_dropout=0.1,lstm_dropout=.1,
                 clf_num_layers=0,clf_hidden_size=256,
                 embedding_dim=32, num_stimuli=10,
                 output_activation='identity',
                 **kwargs):
        super().__init__()
        self.surface_dim = surface_dim
        self.output_dim = output_dim if output_dim else surface_dim
        self.output_activation = output_activation

        self.config = {
            'lstm_dropout': lstm_dropout,
            'surface_dim': surface_dim,
            'output_dim': self.output_dim,
            'lstm_hidden_size': lstm_hidden_size,
            'lstm_num_layers': lstm_num_layers,
            'bidirectional': bidirectional,
            'attn_heads': attn_heads,
            'head_dim': head_dim,
            'attn_dropout': attn_dropout,
            'embedding_dim': embedding_dim,
            'num_stimuli': num_stimuli,
            'output_activation': output_activation,
            'clf_num_layers': clf_num_layers,
            'clf_hidden_size': clf_hidden_size,
                }

        #super().__init__()

        input_size = self.surface_dim#model_cfg.input_size
        output_size = self.output_dim#model_cfg.output_size

        lstm_hidden_size = lstm_hidden_size #model_cfg.lstm.hidden_size
        lstm_num_layers = lstm_num_layers #model_cfg.lstm.num_layers
        bidirectional = bidirectional #model_cfg.lstm.bidirectional

        self.lstm_output_size = (
            lstm_hidden_size * 2 if bidirectional else lstm_hidden_size
        )

        clf_hidden_size = clf_hidden_size #model_cfg.clf.hidden_size
        clf_num_layers = clf_num_layers #model_cfg.clf.num_layers

        MHAtt_n_heads = attn_heads #model_cfg.MHAtt.n_heads
        MHAtt_hidden_size = MHAtt_n_heads * head_dim #model_cfg.MHAtt.head_hidden_size
        MHAtt_dropout = attn_dropout #model_cfg.MHAtt.dropout

        # LSTM - first block
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0  # Add this line
        )

        # Classifier - last block
        clf = [
            nn.Linear(input_size**2, clf_hidden_size),
            nn.ReLU(),
        ]
        for _ in range(clf_num_layers):
            clf.append(nn.Linear(clf_hidden_size, clf_hidden_size))
            clf.append(nn.ReLU())
        clf.append(
            nn.Linear(clf_hidden_size, output_size),
        )
        self.clf = nn.Sequential(*clf)

        # Multihead attention - second block
        self.key_layer = nn.Sequential(
            nn.Linear(
                self.lstm_output_size,
                MHAtt_hidden_size,
            ),
        )
        self.value_layer = nn.Sequential(
            nn.Linear(
                self.lstm_output_size,
                MHAtt_hidden_size,
            ),
        )
        self.query_layer = nn.Sequential(
            nn.Linear(
                self.lstm_output_size,
                MHAtt_hidden_size,
            ),
        )
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=MHAtt_hidden_size,
            num_heads=MHAtt_n_heads,
            dropout=MHAtt_dropout,
            batch_first=True,
        )

        # Global Temporal Attention - third block
        self.upscale = 0.05
        self.upscale2 = 0.5

        self.HW = torch.nn.Hardswish()
        self.gta_embed = nn.Sequential(
            nn.Linear(
                input_size**2,
                round(self.upscale * input_size**2),
            ),
        )
        self.gta_norm = nn.Sequential(
            nn.BatchNorm1d(round(self.upscale * input_size**2),eps=1e-4),
            nn.ReLU(),
        )
        self.gta_attend = nn.Sequential(
            nn.Linear(
                round(self.upscale * input_size**2),
                round(self.upscale2 * input_size**2),
            ),
            nn.ReLU(),
            nn.Linear(round(self.upscale2 * input_size**2), 1),
        )

        self.init_weight()

    """
    def init_weight(self):
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.(param, mode="fan_in")
        for name, param in self.clf.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_in")
        for name, param in self.query_layer.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_in")
        for name, param in self.key_layer.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_in")
        for name, param in self.value_layer.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_in")
        for name, param in self.multihead_attn.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_in")
        for name, param in self.gta_embed.named_parameters():
            if "weight" in name and param.dim() > 1:
                nn.init.kaiming_normal_(param, mode="fan_in")
        for name, param in self.gta_attend.named_parameters():
            if "weight" in name and param.dim() > 1:
                nn.init.kaiming_normal_(param, mode="fan_in")
    """
    # Place this inside the DICESurfacePredictor class
    def init_weight(self):
        """
        Initializes weights for the model.
        - Uses Kaiming Normal for most feed-forward layers.
        - Uses Orthogonal initialization for LSTM recurrent (hidden-to-hidden) weights,
          which is crucial for preventing exploding/vanishing gradients in RNNs.
        - Initializes all biases to zero.
        """
        for name, param in self.named_parameters():
            # Handle all biases by setting them to zero
            if 'bias' in name:
                nn.init.zeros_(param)

            # Handle the special case for LSTM's recurrent weights (weight_hh)
            # This is the most important part for stability.
            elif 'lstm' in name and 'weight_hh' in name:
                nn.init.orthogonal_(param)

            # Handle all other weights (Linear, Conv, LSTM input weights)
            # The check for param.dim() > 1 ensures this only applies to
            # weight tensors (like matrices), not 1D bias vectors.
            elif 'weight' in name and param.dim() > 1:
                # Using nonlinearity='relu' is good practice when using Kaiming/He init
                # as it's designed for layers followed by a ReLU activation.
                nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='relu')

    def gta_attention(self, x, node_axis=1):
        # x.shape: [batch_size; time_length; input_feature_size * input_feature_size]
        x_readout = x.mean(node_axis, keepdim=True)
        x_readout = x * x_readout

        a = x_readout.shape[0]
        b = x_readout.shape[1]
        x_readout = x_readout.reshape(-1, x_readout.shape[2])
        x_embed = self.gta_norm(self.gta_embed(x_readout))
        x_graphattention = (self.gta_attend(x_embed).squeeze()).reshape(a, b)
        x_graphattention = self.HW(x_graphattention.reshape(a, b))
        final_output =  (x * (x_graphattention.unsqueeze(-1))).mean(node_axis)
        return final_output, x_graphattention

    def multi_head_attention(self, x):
        # x.shape: [time_length * batch_size; input_feature_size; lstm_hidden_size]
        key = self.key_layer(x)
        value = self.value_layer(x)
        query = self.query_layer(x)

        attn_output, attn_output_weights = self.multihead_attn(key, value, query)

        return attn_output, attn_output_weights

    def forward(self, x, stimuli, lengths=None,output_temporal_attention=False, output_spatial_attention=False):
        B, T, C = x.shape  # [batch_size, time_length, input_feature_size]

        # 1. pass input to LSTM; treat each channel as an independent single-feature time series
        x = x.permute(0, 2, 1)  # x.shape: [batch_size; input_feature_size; time_length]
        x = x.reshape(B * C, T, 1)  # x.shape: [batch_size * n_channels; time_length; 1]
        ##########################

        if torch.any(torch.isinf(x)) or torch.any(torch.isnan(x)):
            print("!!! WARNING: Invalid values detected in input to LSTM !!!")

        lstm_output, _ = self.lstm(x)
        # lstm_output.shape: [batch_size * input_feature_size; time_length; lstm_hidden_size]
        ##########################
        lstm_output = lstm_output.reshape(B, C, T, self.lstm_output_size)
        # lstm_output.shape: [batch_size; input_feature_size; time_length; lstm_hidden_size]

        if torch.any(torch.isinf(lstm_output)) or torch.any(torch.isnan(lstm_output)):
            print("!!! WARNING: Invalid values detected in output of LSTM !!!")

        # 2. pass lstm_output at each time point to multihead attention to reveal spatial connctions
        lstm_output = lstm_output.permute(2, 0, 1, 3)
        # lstm_output.shape: [time_length; batch_size; input_feature_size; lstm_hidden_size]
        lstm_output = lstm_output.reshape(T * B, C, self.lstm_output_size)
        # lstm_output.shape: [time_length * batch_size; input_feature_size; lstm_hidden_size]
        ##########################
        _, spatial_attn_weights = self.multi_head_attention(lstm_output)
        # attn_weights.shape: [time_length * batch_size; input_feature_size; input_feature_size]
        ##########################
        attn_weights_for_gta = spatial_attn_weights.reshape(T, B, C, C)
        # attn_weights.shape: [time_length; batch_size; input_feature_size; input_feature_size]
        attn_weights_for_gta = attn_weights_for_gta.permute(1, 0, 2, 3)
        # attn_weights.shape: [batch_size; time_length; input_feature_size; input_feature_size]

        # 3. pass attention weights to a global temporal attention to obrain global graph
        temporal_features = attn_weights_for_gta.reshape(B, T, -1)
        # attn_weights.shape: [batch_size; time_length; input_feature_size * input_feature_size]
        ##########################
        FC, temporal_attention = self.gta_attention(temporal_features)
        # FC.shape: [batch_size; input_feature_size * input_feature_size]
        ##########################

        # 4. Pass learned graph to the classifier to get predictions
        logits = self.clf(FC)
        # logits.shape: [batch_size; n_classes]

        if output_temporal_attention and output_spatial_attention:
            return logits, temporal_attention, attn_weights_for_gta
        elif output_temporal_attention and not output_spatial_attention:
            return logits, temporal_attention
        elif not output_temporal_attention and output_spatial_attention:
            return logits, attn_weights_for_gta
        else:
            return logits
