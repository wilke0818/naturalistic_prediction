import torch
import torch.nn as nn

class DiceSparseSurfacePredictor(nn.Module):


    MODEL_NAME = "sparse_dice_model"

    def __init__(self, surface_dim=1056, lstm_hidden_size=50, lstm_num_layers=1, bidirectional=True,
                 attn_hidden_size=64,
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
            'attn_hidden_size': attn_hidden_size,
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
        self.attn_hidden_size = attn_hidden_size 
        self.attn_proj = nn.Linear(self.lstm_output_size,self.attn_hidden_size)



        self.gta_embed = nn.Linear(surface_dim*self.attn_hidden_size, self.graph_dim)
        self.gta_norm = nn.Sequential(
            nn.ReLU()
        )
        self.gta_attend = nn.Sequential(
            nn.Linear(self.graph_dim, self.graph_dim//2),
            nn.ReLU(),
            nn.Linear(self.graph_dim//2, 1)
        )
        self.hardswish = nn.Hardswish()


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
            'attn_hidden_size': study.best_trial.params['attn_hidden_size'],
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
                'attn_hidden_size': trial.suggest_categorical('attn_hidden_size', [512]),#1 low
                'clf_num_layers': trial.suggest_categorical('clf_num_layers', [2]),#0 low
                'clf_hidden_size': trial.suggest_categorical('clf_hidden_size', [512]),#128 low
                'graph_dim': trial.suggest_categorical('graph_dim', [1024])#512 low
            }


    def forward(self, fmri, stimulus=None, lengths=None, output_temporal=False):
        B, T, C = fmri.shape
        x = fmri.permute(0, 2, 1).reshape(B * C, T, 1)  # (B*C, T, 1)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.repeat_interleave(C).cpu(), batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        x, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        # sparse attention over time per region
        qkv = self.attn_proj(x)  # (B*C, T, H)
        attn_scores = torch.bmm(qkv, qkv.transpose(1, 2)) / (qkv.shape[-1] ** 0.5)  # (B*C, T, T)
        attn_weights = torch.softmax(attn_scores, dim=-1).detach()  # detach to save memory
        attn_out = torch.bmm(attn_weights, qkv)  # (B*C, T, H)

        attn_out = attn_out.reshape(B, C, T, self.attn_hidden_size).permute(0, 2, 1, 3)
        attn_out = attn_out.reshape(B, T, -1)  # (B, T, C*H)


        emb = self.gta_embed(attn_out.reshape(-1, attn_out.shape[-1]))
        normed = self.gta_norm(emb)
        scores = self.gta_attend(normed).squeeze().reshape(B, T)
        scores = torch.nn.functional.hardswish(scores)

        weighted = (attn_out * scores.unsqueeze(-1)).sum(1)  # (B, C*H)
        summary = self.gta_embed(weighted)

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
