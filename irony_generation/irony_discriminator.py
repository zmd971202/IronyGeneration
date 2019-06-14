import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class IronyDiscriminator(nn.Module):
    def __init__ (self, hps):
        super(IronyDiscriminator, self).__init__()
        
        vocab_size = hps.vocab_size
        emb_dim = hps.emb_dim
        self.batch_size = hps.dis_batch_size
        self.hid_dim = hps.dis_hid_dim
        self.max_len = hps.max_len
        dropout = hps.dis_dropout
        self.n_layers = hps.dis_n_layers
        
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(emb_dim, self.hid_dim, self.n_layers, dropout=dropout)
        self.fc = nn.Linear(self.n_layers * self.hid_dim, 2)
    
    def forward(self, inputs, lengths, state):
        # inputs [batch, sent_len]
        #print(inputs.size())
        embed = self.embedding(inputs)
        embed = self.drop(embed)
        embed = pack_padded_sequence(embed, lengths, batch_first=True)
        _, X = self.lstm(embed, state)
        
        X = self.fc(X[0].view(-1, self.n_layers * self.hid_dim))
        X = F.softmax(X)
        return X
 
    def init_hidden(self, batch_size):
        h = Variable(torch.zeros(self.n_layers, batch_size, self.hid_dim)).cuda()
        c = Variable(torch.zeros(self.n_layers, batch_size, self.hid_dim)).cuda()
        return (h, c)