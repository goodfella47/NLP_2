import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def lossAugmentedInference(score_matrix, true_heads, device):
    """
        loss augmented inference implementation as per kiperwasser
        Return:
          - score_matrix with ones added to untrue heads
    """
    n = len(true_heads)
    ones = torch.ones([n, n], device=device)
    ones[np.arange(n), true_heads] = 0
    score_matrix += ones


class biaffineEdgeScorer(nn.Module):
    def __init__(self, bilstm_out, hidden_dim):
        super().__init__()

        self.head = nn.Linear(bilstm_out, hidden_dim)
        self.modifier = nn.Linear(bilstm_out, hidden_dim)

        self.dropout_head = nn.Dropout(p=0.333)
        self.dropout_modiefier = nn.Dropout(p=0.333)

        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.b = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, words_encoding):
        # apply linear transform to head and modifier along woth relu activation
        MLP_head = self.head(words_encoding)
        MLP_head = self.dropout_head(F.relu(MLP_head))
        MLP_modiefier = self.modifier(words_encoding)
        MLP_modiefier = self.dropout_modiefier(F.relu(MLP_modiefier))

        biaffine_W = self.W(MLP_head).transpose(1, 2)
        biaffine_W = torch.matmul(MLP_modiefier, biaffine_W)
        biaffine_b = self.b(MLP_head).transpose(1, 2)
        scores = (biaffine_W + biaffine_b).squeeze()

        return scores


class advancedDependencyParser(nn.Module):
    def __init__(self, word_embedding_dim, tag_embedding_dim, hidden_dim, word_vocab_size,
                 tag_vocab_size, bilstm_size, bilstm_dropout, bilstm_layers):
        super(advancedDependencyParser, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.word_embedding = nn.Embedding(word_vocab_size, word_embedding_dim)
        self.pos_embedding = nn.Embedding(tag_vocab_size, tag_embedding_dim)
        blstm_input_dim = self.word_embedding.embedding_dim + self.pos_embedding.embedding_dim
        self.bilstm = nn.LSTM(input_size=blstm_input_dim, hidden_size=bilstm_size, num_layers=3, dropout=bilstm_dropout,
                              bidirectional=True,
                              batch_first=True)
        self.edge_scorer = biaffineEdgeScorer(2 * bilstm_size, hidden_dim)


    def forward(self, word_idx_tensor, pos_idx_tensor, pre_trained_embeds=None, true_heads=None):
        if pre_trained_embeds is not None:
            pre_trained_embeds = pre_trained_embeds.unsqueeze(0)
            if self.training:  # only drop words in training mode
                word_idx_tensor, pre_trained_embeds = self.word_dropout(word_idx_tensor, pre_trained_embeds)
            word_embeds = self.word_embedding(word_idx_tensor.to(self.device)) + pre_trained_embeds.to(self.device)
        else:
            if self.training:  # only drop words in training mode
                word_idx_tensor = self.word_dropout(word_idx_tensor)
            word_embeds = self.word_embedding(word_idx_tensor.to(self.device))
        tag_embeds = self.pos_embedding(pos_idx_tensor.to(self.device))
        x = torch.cat((word_embeds, tag_embeds), dim=-1)
        bilstm_out, _ = self.bilstm(x)
        score_matrix = self.edge_scorer(bilstm_out)
        # use of loss augmented inference in the advanced model
        if true_heads is not None:
            lossAugmentedInference(score_matrix, true_heads, self.device)
        return score_matrix


    # bayesian word and embedding dropout for the advanced model
    def word_dropout(self, word_idx, embeds=None):
        word_mask = (torch.rand(size=word_idx.shape, device=word_idx.device) > 0.333).long()
        if embeds is not None:
            embeds_mask = (torch.rand(size=word_idx.shape, device=word_idx.device) > 0.9).long()
            embeds_mask = (1 - (1 - word_mask) * embeds_mask)
            return word_idx * word_mask, (embeds_mask*embeds[0,:,:].T).T.unsqueeze(0)
        else:
            return word_idx * word_mask


    def save(self, file_path):
        torch.save(self.state_dict(), file_path)


    def load(self, file_path):
        self.load_state_dict(torch.load(file_path))
        self.eval()
