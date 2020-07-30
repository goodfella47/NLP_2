import torch
import torch.nn as nn
import numpy as np

def lossAugmentedInference(score_matrix,true_heads,device):
  n = len(true_heads)
  ones = torch.ones([n,n],device=device)
  ones[np.arange(n),true_heads] = 0
  score_matrix += ones

class edge_scorer(nn.Module):
    def __init__(self, bilstm_out, hidden_dim):
        super(edge_scorer, self).__init__()
        self.head = nn.Linear(bilstm_out, hidden_dim, bias=False)
        self.modifier = nn.Linear(bilstm_out, hidden_dim, bias=True)
        self.fc1 = nn.Linear(hidden_dim, 1, bias=True)

    def forward(self, word_encoding):
        head_linear_encoding = self.head(word_encoding)
        modifier_linear_encoding = self.modifier(word_encoding)
        num_of_words = word_encoding.shape[1]
        score_vec = []
        for i in range(num_of_words):
            X = head_linear_encoding[0, i, :].unsqueeze(0) + modifier_linear_encoding
            X = torch.tanh(X)
            X = self.fc1(X)
            score_vec.append(X)
        scores = torch.stack(score_vec, dim=1).view(num_of_words, num_of_words)
        return scores.transpose(1,0)

class KiperwasserDependencyParser(nn.Module):
    def __init__(self, word_embedding_dim, tag_embedding_dim, hidden_dim, word_vocab_size,
                 tag_vocab_size,advanced = False):
        super(KiperwasserDependencyParser, self).__init__()
        self.advanced = advanced
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.word_embedding = nn.Embedding(word_vocab_size, word_embedding_dim)
        self.pos_embedding = nn.Embedding(tag_vocab_size, tag_embedding_dim)
        self.input_dim = self.word_embedding.embedding_dim + self.pos_embedding.embedding_dim
        self.bilstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.input_dim, num_layers=2, bidirectional=True,
                              batch_first=True)
        self.edge_scorer = edge_scorer(2 * self.input_dim, hidden_dim)

    def forward(self, word_idx_tensor, pos_idx_tensor, true_heads = None):
        word_embeds = self.word_embedding(word_idx_tensor.to(self.device))
        tag_embeds = self.pos_embedding(pos_idx_tensor.to(self.device))
        x = torch.cat((word_embeds, tag_embeds), dim=-1)
        bilstm_out, _ = self.bilstm(x)
        scores = self.edge_scorer(bilstm_out)
        # use of loss augmented inference in the advanced model
        if true_heads is not None:
          lossAugmentedInference(scores,true_heads,self.device)
        return scores

    def save(self,file_path):
      torch.save(self.state_dict(), file_path)

    def load(self,file_path):
      self.load_state_dict(torch.load(file_path))
      self.eval()