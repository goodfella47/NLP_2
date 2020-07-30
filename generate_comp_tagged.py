from evaluation import parse_to_file
from data_loader import split, generateVocabs
from train_basic import emptyModel as basic_model_gen
from train_advanced import emptyModel as advanced_model_gen
import torch

m1_tagged_path = 'comp_m1_305320400.labeled'
m2_tagged_path = 'comp_m2_305320400.labeled'

# initialize empty models
train_path = "train.labeled"
train_sentences_df = split(train_path)
word_to_ix, tag_to_ix, _ = generateVocabs(train_sentences_df)
word_vocab_size = len(word_to_ix)
tag_vocab_size = len(tag_to_ix)

basic_model = basic_model_gen(word_vocab_size, tag_vocab_size)
advanced_model = advanced_model_gen(word_vocab_size, tag_vocab_size)

# put model on cude
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
if use_cuda:
    basic_model.cuda()
    advanced_model.cuda()


# load the trained parameters
basic_model_param = 'basic_model_param'
advanced_model_param = 'advanced_model_param'
basic_model.load(basic_model_param)
advanced_model.load(advanced_model_param)


# parsing for competition
comp_path = 'comp.unlabeled'
parse_to_file(comp_path, m1_tagged_path, word_to_ix, tag_to_ix, basic_model, tp='basic')
parse_to_file(comp_path, m2_tagged_path, word_to_ix, tag_to_ix, advanced_model, tp='advanced')
