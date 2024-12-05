import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import seaborn as sns

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, input):
        embedded = self.embedding(input)
        output, hidden = self.gru(embedded)
        return output, hidden
    
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).unsqueeze(1)
        output = torch.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output.squeeze(1)))
        return output, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        batch_size = source.size(0)
        target_length = target.size(1)
        target_vocab_size = self.decoder.out.out_features

        outputs = torch.zeros(batch_size, target_length, target_vocab_size).to(self.device)

        encoder_output, hidden = self.encoder(source)

        decoder_input = target[:, 0]

        for t in range(1, target_length):
            decoder_output, hidden = self.decoder(decoder_input, hidden)
            outputs[:, t, :] = decoder_output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = decoder_output.argmax(1)
            decoder_input = target[:, t] if teacher_force else top1

        return outputs

def train(model, source_tensor, target_tensor, optimizer, criterion):
    optimizer.zero_grad()
    
    output = model(source_tensor, target_tensor)
    loss = criterion(output.view(-1, output.size(-1)), target_tensor.view(-1))
    
    loss.backward()
    optimizer.step()
    
    return loss.item()

def evaluate(model, source_tensor, target_tensor, criterion):
    with torch.no_grad():
        output = model(source_tensor, target_tensor, 0)
        loss = criterion(output.view(-1, output.size(-1)), target_tensor.view(-1))
    return loss.item()

def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.show()

def plot_attention(attention_weights, source_sentence, predicted_sentence):
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(attention_weights, cmap='YlGnBu', ax=ax)
    ax.set_xticklabels(source_sentence.split(), rotation=90)
    ax.set_yticklabels(predicted_sentence.split())
    plt.title('Attention Weights Visualization')
    plt.show()
    
def create_vocab(size):
    return {i: f'word_{i}' for i in range(size)}

def numbers_to_words(tensor, vocab):
    return [' '.join([vocab[i.item()] for i in sequence]) for sequence in tensor]

import random

def create_realistic_vocab(size):
    common_words = ["the", "be", "to", "of", "and", "a", "in", "that", "have", "I", 
                    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
                    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
                    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what"]
    
    vocab = {i: word for i, word in enumerate(common_words)}
    
    # Fill the rest with numbered words
    for i in range(len(common_words), size):
        vocab[i] = f"word_{i}"
    
    return vocab