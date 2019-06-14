import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, hps):
        super(Decoder, self).__init__()
        
        emb_dim = hps.emb_dim
        hid_dim = hps.decoder_hid_dim
        n_layers = hps.decoder_n_layers
        dropout = hps.decoder_dropout
        vocab_size = hps.vocab_size
        
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.out = nn.Linear(hid_dim, vocab_size)
    
    def forward(self, inputs, hidden, cell):
        #input = [batch size, 1]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        #print(inputs,size())
        inputs = inputs.unsqueeze(0)
        #print(inputs.size())
        embedded = self.embedding(inputs)
        #embedded = self.dropout(embedded)
        output, (hidden, cell) = self.decoder(embedded, (hidden, cell))
        #output = [sent len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        prediction = self.out(output.squeeze(0))
        #prediction = F.softmax(prediction, dim=1)
        return prediction, hidden, cell