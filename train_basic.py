# basic training
import torch
import torch.optim as optim
import torch.nn as nn
from basic_model import KiperwasserDependencyParser
from data_loader import split, generateVocabs, KiperwasserDataset
from torch.utils.data.dataloader import DataLoader
import evaluation
from utils import plot_figures

# Hyperparameters
WORD_EMBEDDING_DIM = 100
POS_EMBEDDING_DIM = 20
EPOCHS = 10  # 10
HIDDEN_DIM = 100
accumulate_grad_steps = 50

def emptyModel(word_vocab_size,tag_vocab_size):
    model = KiperwasserDependencyParser(WORD_EMBEDDING_DIM, POS_EMBEDDING_DIM, HIDDEN_DIM, word_vocab_size, tag_vocab_size)
    return model

def train_basic(plot = False):
    # Data loading
    train_path = "train.labeled"
    test_path = "test.labeled"
    train_sentences_df = split(train_path)
    test_sentences_df = split(test_path)
    word_to_ix, tag_to_ix, _ = generateVocabs(train_sentences_df)

    train = KiperwasserDataset(word_to_ix, tag_to_ix, train_sentences_df, tp='train_')
    train_dataloader = DataLoader(train, shuffle=True)
    test = KiperwasserDataset(word_to_ix, tag_to_ix, test_sentences_df, tp='test')
    test_dataloader = DataLoader(test, shuffle=False)

    word_vocab_size = len(train.word_to_ix)
    tag_vocab_size = len(train.tag_to_ix)

    loss_function = nn.CrossEntropyLoss()  # NLLLoss
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    model = emptyModel(word_vocab_size,tag_vocab_size)

    if use_cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training start
    print("Training Started")
    UAS_train_list = []
    UAS_test_list = []
    loss_list = []
    epochs = EPOCHS
    best_UAS_sf = 0.84
    for epoch in range(epochs):
        current_loss = 0  # To keep track of the loss value
        i = 0
        for input_data in train_dataloader:
            model.train()  # put model on train model to procceed with dropout
            i += 1
            words_idx_tensor, pos_idx_tensor, gold = input_data
            gold = gold.squeeze(0).to(device)
            scores = model(words_idx_tensor, pos_idx_tensor, gold)
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
        UAS_train = evaluation.evaluate(train_dataloader, model)
        UAS_test = evaluation.evaluate(test_dataloader, model)
        UAS_train_list.append(UAS_train)
        UAS_test_list.append(UAS_test)
        if UAS_test > best_UAS_sf:
            model.save('basic_final_'+str(epoch))
            best_UAS_sf = UAS_test
            plot_figures(loss_list, UAS_train_list, UAS_test_list, 'basic_'+str(epoch))
        print(f"Epoch {epoch + 1}, \tLoss: {current_loss:.7f}, \t UAS_train: {UAS_train:.4f}, \tUAS_test: {UAS_test:.4f}")
    if plot:
        plot_figures(loss_list, UAS_train_list, UAS_test_list, 'basic')
    return model

if __name__ == "__main__":
    model_path = 'basic_model_param'
    model = train_basic(plot = True)
    model.save(model_path)