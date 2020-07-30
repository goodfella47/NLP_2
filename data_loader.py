import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from random import random
from torchtext.vocab import Vocab
from collections import Counter, defaultdict

ROOT_TOKEN = 'ROOT'
UNKNOWN_TOKEN = '<unk>'
SPECIAL_TOKENS = [UNKNOWN_TOKEN, ROOT_TOKEN]


def split(file_path):
    """
      Split the file into a list of pandas dataframes.
      note that this is not the fastest approach but it is very readable and user friendly

      :file_path: 
      :return: 
    """
    vocab = []
    columns = ['Token Counter', 'Token', 'Token POS', 'Token Head']
    sentence = []
    sentence.append([0, 'ROOT', 'ROOT', 0])  # append root to start of sentence
    with open(file_path) as f:
        for line in f:
            if line.split():
                line = line.rstrip('\n').split('\t')
                sentence.append([int(line[0]), line[1], line[3], int(line[6])])
            else:
                vocab.append(pd.DataFrame(sentence, columns=columns))
                sentence = []
                sentence.append([0, 'ROOT', 'ROOT', 0])  # append root to start of sentence
    return vocab


def generateVocabs(sentences_df):
    """
        Extract vocabs from given datasets. Return a word_to_ix and tag_to_ix.
        Return:
          - word_to_ix
          - tag_to_ix
    """
    word_vocab = []
    pos_vocab = []
    for sentence in sentences_df:
        for word in sentence['Token']:
            word_vocab.append(word)
        for pos_tag in sentence['Token POS']:
            pos_vocab.append(pos_tag)
    word_vocab = list(dict.fromkeys(word_vocab).keys())
    pos_vocab = list(dict.fromkeys(pos_vocab).keys())
    word_to_ix = {word: i + 2 for i, word in enumerate(word_vocab)}
    tag_to_ix = {tag: i for i, tag in enumerate(pos_vocab)}
    word_to_ix[UNKNOWN_TOKEN] = 0 #crucial for word dropping
    word_to_ix[ROOT_TOKEN] = 1
    ix_to_word = {i: word for word, i in word_to_ix.items()}
    return word_to_ix, tag_to_ix, ix_to_word

def wordFrequency(sentences_df):
    """
      Calculates frequency of each unique word in the corpus

      :@param sentences_df: pandas dataframe object of all sentences in the corpus
      :return: word frequency dict
    """
    word_frequency = defaultdict(int)
    for df in sentences_df:
        words = df['Token']
        for word in words:
            if word in word_frequency.keys():
                word_frequency[word] += 1
            else:
                word_frequency[word] = 1
    return word_frequency

def init_word_embeddings(word_dict):
    """
      loads word embeddings for the advanced model from a torchtext

      :@param word_dict: word:frequency dict
      :return:
    """
    glove = Vocab(Counter(word_dict), vectors="glove.840B.300d", specials=SPECIAL_TOKENS)
    return glove.stoi, glove.itos, glove.vectors

def get_pretrained_vector(pretrained_embeds, words_idx_tensor, idx_to_words):
    """
      initializes pre-trained word embeddings for advanced model

      :@param pretrained_embeds: torchtext word embeddings
      :@param words_idx_tensor: word to index tensor
      :@param idx_to_words: index to word tensor
      :return:
    """
    embeds = []
    for i in words_idx_tensor[0]:
        word = idx_to_words[i.item()]
        embed_idx = pretrained_embeds[0][word]
        embed = pretrained_embeds[2][embed_idx]
        embeds.append(list(embed))
    return torch.tensor(embeds, dtype=torch.float32)

class KiperwasserDataset(Dataset):
    """
      Dataset object

      :@param Dataset:
      :return:
    """
    def __init__(self, word_to_ix, tag_to_ix, sentences_df, tp='basic', alpha=0.25, word_embed_to_ix=None):
        super().__init__()
        self.alpha = alpha
        self.tp = tp
        self.sentences_df = sentences_df
        self.word_to_ix = word_to_ix
        self.tag_to_ix = tag_to_ix
        self.word_embed_to_ix = word_embed_to_ix
        self.items = []
        if tp == 'basic':
            self.frequency = self.frequency()
        for df in sentences_df:
            words = [word_to_ix[w] if w in word_to_ix.keys() else word_to_ix[UNKNOWN_TOKEN] for w in df['Token']]
            pos = [tag_to_ix[t] for t in df['Token POS']]
            true_heads_tensor = torch.tensor(list(df['Token Head']), dtype=torch.long)
            words_tensor = torch.tensor(words, dtype=torch.long)
            pos_tensor = torch.tensor(pos, dtype=torch.long)
            if self.word_embed_to_ix:
                words_to_embed = [word_embed_to_ix[w] for w in df['Token']]
                words_to_embed = torch.tensor(words_to_embed, dtype=torch.long)
                self.items.append((words_tensor, pos_tensor, true_heads_tensor, words_to_embed))
            else:
                self.items.append((words_tensor, pos_tensor, true_heads_tensor))

    def __len__(self):
        return len(self.sentences_df)

    def __getitem__(self, index):
        if self.word_embed_to_ix:
            # word_embeds are the pre-trained for the advanced model
            words, pos, true_heads, word_embeds = self.items[index]
            return words, pos, true_heads, word_embeds
        words, pos, true_heads = self.items[index]
        # word drop out for basic model
        if self.tp == 'basic':
            words = self.dropout(words)
        return words, pos, true_heads

    def frequency(self):
        word_frequency = {}
        for df in self.sentences_df:
            words = df['Token']
            for word in words:
                if word in word_frequency.keys():
                    word_frequency[self.word_to_ix[word]] += 1
                else:
                    word_frequency[self.word_to_ix[word]] = 1
        return word_frequency

    # word dropout implementation for basic model
    def dropout(self, words):
        replaced_words = []
        for word in words:
            prob = self.alpha / (self.alpha + self.frequency[word.item()])
            if prob < random():
                replaced_words.append(torch.tensor(self.word_to_ix[UNKNOWN_TOKEN]))
            else:
                replaced_words.append(word)
        return torch.tensor(replaced_words, dtype=torch.long)
