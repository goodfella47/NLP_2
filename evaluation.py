# evaluation
import numpy as np
import pandas as pd
import torch
from chu_liu_edmonds import decode_mst
from train_basic import KiperwasserDataset
from torch.utils.data.dataloader import DataLoader
from data_loader import get_pretrained_vector,generateVocabs, wordFrequency, init_word_embeddings

def evaluate(dataloader, model, pretrained_embeds = None, ix_to_word = None):
    model.eval()  # put model on eval model to avoid dropouts
    true_positives = 0
    total_tokens = 0
    with torch.no_grad():
        for batch_idx, input_data in enumerate(dataloader):
            if len(input_data)==4:
                word_idx, pos_idx, gold, word_embeds_idx = input_data
            else:
                word_idx, pos_idx, gold = input_data
                word_embeds_idx = word_idx
            if pretrained_embeds and ix_to_word:
                external_embeds = get_pretrained_vector(pretrained_embeds, word_embeds_idx, ix_to_word)
                scores = model(word_idx, pos_idx, external_embeds)
            else:
                scores = model(word_idx, pos_idx)
            scores = scores.cpu().detach().numpy().T
            gold = gold.squeeze(0)[1:].detach().numpy()

            predicted_heads, _ = decode_mst(scores, len(scores[0]), False)
            true_positives += np.sum(np.equal(predicted_heads[1:], gold))
            total_tokens += len(gold)
    uas = true_positives / total_tokens
    return uas

def split(file_path):
    """
      Split the file into a list of pandas dataframes where each sentence is represented by a dataframe.
      note that this is not a fast approach but it is very readable and user friendly

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
                sentence.append([int(line[0]), line[1], line[3], 0])
            else:
                vocab.append(pd.DataFrame(sentence, columns=columns))
                sentence = []
                sentence.append([0, 'ROOT', 'ROOT', 0])  # append root to start of sentence
    return vocab

def parser(dataloader, model, pretrained_embeds=None, ix_to_word=None):
    predicted_list = []
    model.eval()  # put model on eval model to avoid dropouts
    with torch.no_grad():
        for batch_idx, input_data in enumerate(dataloader):
            if len(input_data) == 4:
                word_idx, pos_idx, gold, word_embeds_idx = input_data
            else:
                word_idx, pos_idx, gold = input_data
                word_embeds_idx = word_idx
            if pretrained_embeds and ix_to_word:
                external_embeds = get_pretrained_vector(pretrained_embeds, word_embeds_idx, ix_to_word)
                scores = model(word_idx, pos_idx, external_embeds)
            else:
                scores = model(word_idx, pos_idx)
            scores = scores.cpu().detach().numpy().T
            predicted_heads, _ = decode_mst(scores, len(scores[0]), False)
            predicted_list.append(predicted_heads[1:])
    return predicted_list

def parse_to_file(comp_path, parse_path, word_to_ix, tag_to_ix, model, tp = 'basic'):
    comp_sentences_df = split(comp_path)
    word_to_ix_comp, _, ix_to_word_comp = generateVocabs(comp_sentences_df)
    word_dict_comp = wordFrequency(comp_sentences_df)
    comp_pretrained_embeds = init_word_embeddings(word_dict_comp)
    #comp = KiperwasserDataset(word_dict, word_to_ix, tag_to_ix, comp_sentences_df)
    comp = KiperwasserDataset(word_to_ix, tag_to_ix, comp_sentences_df, tp = 'comp', word_embed_to_ix = word_to_ix_comp)
    comp_dataloader = DataLoader(comp, shuffle=False)
    if tp == 'advanced':
        predicted_list = parser(comp_dataloader, model, comp_pretrained_embeds, ix_to_word_comp)
    else:
        predicted_list = parser(comp_dataloader, model)

    predicted_iter = iter(predicted_list)
    i = 0
    new_sentence = True
    with open(parse_path, 'w+') as predict_file:
        with open(comp_path) as f:
            for line in f:
                if new_sentence:
                    heads = next(predicted_iter)
                    new_sentence = False
                    i = 0
                if line.split():
                    line = line.rstrip('\n').split('\t')
                    line[6] = str(heads[i])
                    to_write = '\t'.join(line)
                    predict_file.write(to_write)
                    predict_file.write('\n')
                    i += 1
                else:
                    new_sentence = True
                    predict_file.write('\n')

