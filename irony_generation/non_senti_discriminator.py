import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class NonSentiDiscriminator(nn.Module):
    def __init__ (self, hps):
        super(NonSentiDiscriminator, self).__init__()
        
        vocab_size = hps.vocab_size
        emb_dim = hps.emb_dim
        self.batch_size = hps.non_senti_batch_size
        self.hid_dim = hps.non_senti_hid_dim
        self.max_len = hps.max_len
        dropout1 = hps.non_senti_dropout1
        dropout2 = hps.non_senti_dropout2
        self.n_layers = hps.non_senti_n_layers
        
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.drop1 = nn.Dropout(dropout1)
        self.lstm = nn.LSTM(emb_dim, self.hid_dim, self.n_layers)
        self.dense = nn.Linear(self.n_layers * self.hid_dim, 64)
        self.drop2 = nn.Dropout(dropout2)
        self.fc = nn.Linear(64, 2)
    
    def forward(self, inputs, lengths, state):
        # inputs [batch, sent_len]
        #print(inputs.size())
        embed = self.embedding(inputs)
        #embed = self.drop1(embed)
        embed = pack_padded_sequence(embed, lengths, batch_first=True)
        _, X = self.lstm(embed, state)
        
        X = self.dense(X[0].view(-1, self.n_layers * self.hid_dim))
        X = self.drop2(X)
        X = self.fc(X)
        X = F.softmax(X)
        return X
 
    def init_hidden(self, batch_size):
        h = Variable(torch.zeros(self.n_layers, batch_size, self.hid_dim)).cuda()
        c = Variable(torch.zeros(self.n_layers, batch_size, self.hid_dim)).cuda()
        return (h, c)