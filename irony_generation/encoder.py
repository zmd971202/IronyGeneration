import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Encoder(nn.Module):
    def __init__(self, hps):
        super(Encoder, self).__init__()
        
        vocab_size = hps.vocab_size
        self.batch_size = hps.batch_size
        emb_dim = hps.emb_dim
        self.hid_dim = hps.encoder_hid_dim
        self.n_layers = hps.encoder_n_layers
        dropout = hps.encoder_dropout
        
        self.hps = hps
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.LSTM(emb_dim, self.hid_dim, self.n_layers, dropout=dropout)
    
    def forward(self, inputs, lengths, state):
        #inputs = [batch size, src sent len]
        embed = self.embedding(inputs)
        #embed = self.drop(embed)
        embed = pack_padded_sequence(embed, lengths, batch_first=True)
        output, (hidden, cell) = self.encoder(embed, state)
        
        output, _ = pad_packed_sequence(output)
        #outputs = [src sent len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        return output, (hidden, cell)
    
    def init_hidden(self, batch_size):
        h = Variable(torch.zeros(self.n_layers, batch_size, self.hid_dim)).cuda()
        c = Variable(torch.zeros(self.n_layers, batch_size, self.hid_dim)).cuda()
        return (h, c)
        