import torch
import torch.nn as nn

class MiniDiceSurfacePredictor(nn.Module):


    MODEL_NAME = "mini_dice_model"

    def __init__(self, surface_dim=1056, lstm_hidden_size=50, lstm_num_layers=1, bidirectional=True,
                 clf_num_layers=0,clf_hidden_size=256,
                 embedding_dim=32, num_stimuli=10,
                 graph_dim=1024,
                 output_activation='identity',
                 **kwargs):
        super().__init__()
        self.surface_dim = surface_dim
        self.output_activation = output_activation

        self.config = {
            'surface_dim': surface_dim,
            'lstm_hidden_size': lstm_hidden_size,
            'lstm_num_layers': lstm_num_layers,
            'bidirectional': bidirectional,
            'embedding_dim': embedding_dim,
            'num_stimuli': num_stimuli,
            'output_activation': output_activation,
            'clf_num_layers': clf_num_layers,
            'clf_hidden_size': clf_hidden_size,
            'graph_dim': graph_dim
                }

#        print(self.config)
        self.graph_dim = graph_dim

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            bidirectional=bidirectional,
            batch_first=True
        )

        self.lstm_output_size = lstm_hidden_size * (2 if bidirectional else 1)

        # Temporal embedding + attention
        self.temporal_embed = nn.Linear(self.lstm_output_size * surface_dim, graph_dim)
        self.temporal_norm = nn.Sequential(
            nn.BatchNorm1d(graph_dim),
            nn.ReLU()
        )
        self.temporal_attend = nn.Sequential(
            nn.Linear(graph_dim, graph_dim // 2),
            nn.ReLU(),
            nn.Linear(graph_dim // 2, 1)
        )

        #if self.use_stimulus:
        #    self.stim_emb = nn.Embedding(num_stimuli, embedding_dim)
        #    clf_input_dim = graph_dim + embedding_dim
        #else:
        #    clf_input_dim = graph_dim

        clf = [
            nn.Linear(self.graph_dim, clf_hidden_size),
            nn.ReLU(),
        ]
        for _ in range(clf_num_layers):
            clf.append(nn.Linear(clf_hidden_size, clf_hidden_size))
            clf.append(nn.ReLU())
        clf.append(
            nn.Linear(clf_hidden_size, surface_dim),
        )
        self.clf = nn.Sequential(*clf)

        #clf_input_dim = input_size**2
        #if self.use_stimulus:
        #    self.stim_emb = nn.Embedding(num_stimuli, embedding_dim)
        #    clf_input_dim += embedding_dim

        #self.output_layer = nn.Linear(clf_input_dim, input_size)

        if output_activation == 'sigmoid':
            self.final_activation = nn.Sigmoid()
        elif output_activation == 'tanh':
            self.final_activation = nn.Tanh()
        else:
            self.final_activation = nn.Identity()


    @staticmethod
    def _get_stimuli_embeddings(friends_stimuli_dir, season, episode_num, episode_part):
        return 0


    @staticmethod
    def _found_hyperparams(study):
        return {
            'lr': study.best_trial.params['lr_exp'],
            'lstm_hidden_size': study.best_trial.params['lstm_hidden_size'],
            'lstm_num_layers': study.best_trial.params['lstm_num_layers'],
            'clf_num_layers': study.best_trial.params['clf_num_layers'],
            'clf_hidden_size': study.best_trial.params['clf_hidden_size'],
            'graph_dim': study.best_trial.params['graph_dim'],
            'weight_decay': study.best_trial.params['weight_decay']
        }

    @staticmethod
    def _sample_optuna_hyperparams(trial):
        return {
                'lr': trial.suggest_float('lr_exp', 10**(-5), 10**(-3)),
                'lstm_hidden_size': trial.suggest_categorical('lstm_hidden_size', [64]), #20 low
                'weight_decay': trial.suggest_categorical('weight_decay', [0.0, 1e-5, 1e-4]),
                'lstm_num_layers': trial.suggest_categorical('lstm_num_layers', [3]),#1 low
                'clf_num_layers': trial.suggest_categorical('clf_num_layers', [2]),#0 low
                'clf_hidden_size': trial.suggest_categorical('clf_hidden_size', [512]),#128 low
                'graph_dim': trial.suggest_categorical('graph_dim', [1024])#512 low
            }

    def forward(self, fmri, stimulus, lengths,output_temporal=False):
        B, T, C = fmri.shape
        x = fmri.permute(0, 2, 1).reshape(B * C, T, 1)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.repeat_interleave(C).cpu(), batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        x, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        x = x.reshape(B, C, T, self.lstm_output_size).permute(0, 2, 1, 3)  # (B, T, C, H)

        x_flat = x.reshape(B, T, -1)  # (B, T, C*H)
        temporal_embeddings = self.temporal_embed(x_flat.reshape(-1, x_flat.shape[-1]))  # (B*T, graph_dim)
        normed = self.temporal_norm(temporal_embeddings)  # (B*T, graph_dim)
        scores = self.temporal_attend(normed).squeeze().reshape(B, T)  # (B, T)
        scores = torch.nn.functional.hardswish(scores)

        # Apply mask if lengths provided
        if lengths is not None:
            mask = torch.arange(T, device=lengths.device)[None, :] < lengths[:, None]  # (B, T)
            scores = scores * mask
            scores = scores / lengths[:, None].clamp(min=1.0)  # normalize

        scores = scores.unsqueeze(-1)  # (B, T, 1)
        x_summary = (x_flat * scores).sum(dim=1)  # (B, C*H)
        summary = self.temporal_embed(x_summary)  # (B, graph_dim)

        #if self.use_stimulus and stimulus is not None:
        #    stim_vec = self.stim_emb(stimulus)
        #    clf_input = torch.cat([summary, stim_vec], dim=1)
        #else:
        clf_input = summary

        pred = self.clf(clf_input)
        if output_temporal:
            return self.final_activation(pred), scores
        else:
            return self.final_activation(pred)
