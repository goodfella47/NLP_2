# basic training
import torch
import torch.optim as optim
import torch.nn as nn
from advanced_model import advancedDependencyParser
from data_loader import split, generateVocabs, KiperwasserDataset, wordFrequency, init_word_embeddings, \
    get_pretrained_vector
from torch.utils.data.dataloader import DataLoader
import evaluation
from utils import plot_figures

# Hyperparameters
WORD_EMBEDDING_DIM = 300  # must be 300 for external embeddings
POS_EMBEDDING_DIM = 30
BILSTM_DIM = 256
MLP_HIDDEN_DIM = 256
BILSTM_DROPOUT = 0.333
LEARNING_RATE = 0.0025
WEIGHT_DECAY = 1e-5
BILSTM_LAYERS = 3
EPOCHS = 70
accumulate_grad_steps = 25

def emptyModel(word_vocab_size, tag_vocab_size):
    model = advancedDependencyParser(WORD_EMBEDDING_DIM, POS_EMBEDDING_DIM, MLP_HIDDEN_DIM, word_vocab_size,
                                     tag_vocab_size, BILSTM_DIM, BILSTM_DROPOUT, BILSTM_LAYERS)
    return model

def train(plot=False):
    # Data loading
    train_path = "train.labeled"
    test_path = "test.labeled"
    train_sentences_df = split(train_path)
    test_sentences_df = split(test_path)
    word_to_ix, tag_to_ix, ix_to_word = generateVocabs(train_sentences_df)
    word_to_ix_test, _, ix_to_word_test = generateVocabs(test_sentences_df)
    word_dict_train = wordFrequency(train_sentences_df)
    word_dict_test = wordFrequency(test_sentences_df)
    train_pretrained_embeds = init_word_embeddings(word_dict_train)
    test_pretrained_embeds = init_word_embeddings(word_dict_test)

    train = KiperwasserDataset(word_to_ix, tag_to_ix, train_sentences_df, tp='train_')
    train_dataloader = DataLoader(train, shuffle=True)
    test = KiperwasserDataset(word_to_ix, tag_to_ix, test_sentences_df, tp='test', word_embed_to_ix=word_to_ix_test)
    test_dataloader = DataLoader(test, shuffle=False)

    word_vocab_size = len(train.word_to_ix)
    tag_vocab_size = len(train.tag_to_ix)

    loss_function = nn.CrossEntropyLoss()  # NLLLoss
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    model = emptyModel(word_vocab_size, tag_vocab_size)

    if use_cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.9), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Training start
    print("Training Started")
    UAS_train_list = []
    UAS_test_list = []
    loss_list = []
    epochs = EPOCHS
    best_UAS_sf = 0.905
    for epoch in range(epochs):
        current_loss = 0  # To keep track of the loss value
        i = 0
        for input_data in train_dataloader:
            model.train()  # put model on train model to proceed with dropout
            i += 1
            words_idx_tensor, pos_idx_tensor, gold = input_data
            external_embeds = get_pretrained_vector(train_pretrained_embeds, words_idx_tensor, ix_to_word)
            gold = gold.squeeze(0).to(device)
            scores = model(words_idx_tensor, pos_idx_tensor, external_embeds, gold)
            loss = loss_function(scores[1:, :], gold[1:])
            loss = loss / accumulate_grad_steps
            loss.backward()
            if i % accumulate_grad_steps == 0:
                optimizer.step()
                model.zero_grad()
            current_loss += loss.item()

        # below used for plotting
        current_loss = current_loss / len(train)
        loss_list.append(float(current_loss))
        UAS_train = evaluation.evaluate(train_dataloader, model, train_pretrained_embeds, ix_to_word)
        UAS_test = evaluation.evaluate(test_dataloader, model, test_pretrained_embeds, ix_to_word_test)
        UAS_train_list.append(UAS_train)
        UAS_test_list.append(UAS_test)
        if UAS_test > best_UAS_sf:
            model.save('advanced_final_'+str(epoch))
            best_UAS_sf = UAS_test
            plot_figures(loss_list, UAS_train_list, UAS_test_list, 'advanced_'+str(epoch))
        print(
            f"Epoch {epoch + 1}, \tLoss: {current_loss:.7f}, \t UAS_train: {UAS_train:.4f}, \tUAS_test: {UAS_test:.4f}")
    if (plot):
        plot_figures(loss_list, UAS_train_list, UAS_test_list, 'advanced')
    return model

if __name__ == "__main__":
    model_path = 'advanced_model_param'
    model = train(plot=True)
    model.save(model_path)