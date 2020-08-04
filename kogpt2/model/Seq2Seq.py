import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from torch.nn import CrossEntropyLoss
from torch.nn.functional import gelu

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, gpt_vocab, args):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.vocab = gpt_vocab
        self.device = args.device
        self.keyword_Module = args.keyword_Module

    def forward(self, source_ids, target_ids, key_score=None, lm_labels=None):
        encoded_layers, pooled_output = self.encoder(source_ids)

        if self.keyword_Module == "add":
            key_score = key_score.tolist()
            for i, score in enumerate(key_score):
                key_score_list = list(filter(lambda a: a != self.vocab[self.vocab.padding_token], score))
                key_score_list = list(map(lambda a: a-min(key_score_list), key_score_list))
                key_score_list = list(map(lambda a: a/max(key_score_list), key_score_list))
                key_score_list = list(map(lambda a: a + 0.5, key_score_list))

                """
                """
                tmp = torch.tensor(key_score_list)
                tmp = F.softmax(tmp, dim=-1)
                tmp = tmp.tolist()
                for k,v in enumerate(tmp):
                    encoded_layers[-1][i][k+1] = encoded_layers[-1][i][k+1]*2

        if lm_labels is not None:
            if self.keyword_Module == "attention":
                size = key_score.size()
                key_score = key_score.unsqueeze(-1)
                key_score = key_score.expand(size[0],size[1], 768)
                outputs = self.decoder(target_ids, encoded_layers[-1], attention_score=key_score, labels=lm_labels)
                return outputs
            else:
                outputs = self.decoder(target_ids, encoded_layers[-1], labels=lm_labels)
                return outputs
        else:
            if self.keyword_Module == "attention":
                size = key_score.size()
                key_score = key_score.unsqueeze(-1)
                key_score = key_score.expand(size[0],size[1], 768)
                outputs = self.decoder(target_ids, encoded_layers[-1], attention_score=key_score)
                return outputs
            else:
                outputs = self.decoder(target_ids, encoded_layers[-1])
                return outputs

class Seq2Seq_gpt2(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, source_ids, target_ids, key_score=None, lm_labels=None):
        _, _, all_hidden_states = self.encoder(source_ids)

        if key_score is not None:
            key_score = key_score.tolist()
            for i, score in enumerate(key_score):
                key_score_list = list(filter(lambda a: a != -100, score))
                key_score_list = key_score_list - min(key_score_list)
                key_score_list = key_score_list/max(key_score_list)
                key_score_list = key_score_list+0.5
                tmp = torch.tensor(key_score_list)
                tmp = F.softmax(tmp, dim=-1)
                tmp = tmp.tolist()
                for k,v in enumerate(tmp):
                    all_hidden_states[-1][i][k+1] = all_hidden_states[-1][i][k+1]*v

        if lm_labels is not None:
            outputs = self.decoder(target_ids, all_hidden_states[-1], labels=lm_labels)
            return outputs
        else:
            outputs = self.decoder(target_ids, all_hidden_states[-1])
            return outputs