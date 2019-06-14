import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F
from encoder import Encoder
from decoder import Decoder
import random
import numpy as np
        
class Seq2Seq(nn.Module):
    def __init__(self, hps):
        super(Seq2Seq, self).__init__()
        
        self.hps = hps
        self.vocab_size = hps.vocab_size
        self.emb_dim = hps.emb_dim
        self.max_len = hps.max_len
        self.batch_size = hps.batch_size
        self.test_batch_size = hps.test_batch_size
        
        #self.embedding = nn.Embedding(self.vocab_size, self.emb_dim)
        self.encoder = Encoder(hps)
        self.irony_decoder = Decoder(hps)
        self.non_decoder = Decoder(hps)
    
    def forward(self):
        pass
        
    def test_ae(self, src, lengths, tags):
        batch_size = src.size(0)
        max_len = src.size(1)
        trg_vocab_size = self.vocab_size
        
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).cuda()
        
        state = self.encoder.init_hidden(src.size(0))
        _, state = self.encoder(src, lengths, state)
        hidden = state[0]
        cell = state[1]
        
        tag = tags[0].item()
        if tag == 0:
            decoder = self.non_decoder
        elif tag == 1:
            decoder = self.irony_decoder
        
        input = Variable(torch.LongTensor(batch_size).fill_(2)).cuda()
        for t in range(1, max_len):

            output, hidden, cell = decoder(input, hidden, cell) #[batch_size, vocab_size]
            outputs[t] = output
            top1 = output.max(1)[1] # max index #[batch_size, ]
            input = top1
        
        outputs = outputs.argmax(2).squeeze(1) #[max_len, ]
        outputs[0] = 2
        return outputs
    
    def test_tran(self, src, lengths, tags):
        batch_size = src.size(0)
        max_len = src.size(1)
        trg_vocab_size = self.vocab_size
        
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).cuda()
        
        state = self.encoder.init_hidden(src.size(0))
        _, state = self.encoder(src, lengths, state)
        hidden = state[0]
        cell = state[1]
        
        tag = tags[0].item()
        if tag == 0:
            decoder = self.irony_decoder
        elif tag == 1:
            decoder = self.non_decoder
        
        input = Variable(torch.LongTensor(batch_size).fill_(2)).cuda()
        for t in range(1, max_len):

            output, hidden, cell = decoder(input, hidden, cell) #[batch_size, vocab_size]
            outputs[t] = output
            top1 = output.max(1)[1] # max index #[batch_size, ]
            input = top1
        
        outputs = outputs.argmax(2).squeeze(1) #[max_len, ]
        outputs[0] = 2
        return outputs
        
    def batchNLLLoss(self, src, trg, lengths, tags, teacher_forcing_ratio = 0.5):
        #src = [batch size,src sent len]
        #trg = [batch size,trg sent len]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        batch_size = src.size(0)
        max_len = src.size(1)
        trg_vocab_size = self.vocab_size

        #tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).cuda()

        #last hidden state of the encoder is used as the initial hidden state of the decoder
        #src_embedding = self.embedding(src)
        state = self.encoder.init_hidden(src.size(0))
        _, state = self.encoder(src, lengths, state)
        hidden = state[0]
        cell = state[1]
        
        tag = tags[0].item()
        if tag == 0:
            decoder = self.non_decoder
        elif tag == 1:
            decoder = self.irony_decoder
        
        #first input to the decoder is the <sos> tokens
        input = trg[:,0]

        loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        loss = 0
        preds = []
        for t in range(1, max_len):

            output, hidden, cell = decoder(input, hidden, cell) #[batch_size, vocab_size]
            #loss += loss_fn(output, trg[:, t])
            preds.append(output)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1] # max index #[batch_size, 1]
            input = (trg[:, t] if teacher_force else top1)
        
        #loss = loss / (max_len - 1)
        preds = torch.stack(preds, dim=1).view(batch_size * (max_len - 1), -1)
        loss = loss_fn(preds, trg[:, 1:].contiguous().view(-1))
        return outputs, loss
    
    def POS_dis(self):
        pass
    
    def get_reward(self, src, sample_y, base_y, lengths, sample_lengths, base_lengths, dis, senti, non_senti, mode):
        alpha1 = self.hps.alpha1
        alpha2 = self.hps.alpha2
        alpha3 = self.hps.alpha3
        
        threshold1 = 0.3  # irony senti
        threshold2 = 0.6  # non senti
        
        # irony
        state = dis.init_hidden(src.size(0))
        output_src = [o[1] for o in dis(src, lengths, state).data.tolist()]
        
        state = dis.init_hidden(base_y.size(1))
        output_base = [o[1] for o in dis(base_y.transpose(0, 1), base_lengths, state).data.tolist()]
        
        state = dis.init_hidden(sample_y.size(1))
        output_sample = [o[1] for o in dis(sample_y.transpose(0, 1), sample_lengths, state).data.tolist()]
        
        # senti
        if mode == 0:
            h = senti.init_hidden(src.size(0))
            senti_output_src = [o[1] for o in senti(src, lengths, h).data.tolist()]

            h = non_senti.init_hidden(base_y.size(1))
            senti_output_base = [o[1] for o in non_senti(base_y.transpose(0, 1), base_lengths, h).data.tolist()]

            h = non_senti.init_hidden(sample_y.size(1))
            senti_output_sample = [o[1] for o in non_senti(sample_y.transpose(0, 1), sample_lengths, h).data.tolist()]
            
        elif mode == 1:
            h = non_senti.init_hidden(src.size(0))
            senti_output_src = [o[1] for o in non_senti(src, lengths, h).data.tolist()]

            h = senti.init_hidden(base_y.size(1))
            senti_output_base = [o[1] for o in senti(base_y.transpose(0, 1), base_lengths, h).data.tolist()]

            h = senti.init_hidden(sample_y.size(1))
            senti_output_sample = [o[1] for o in senti(sample_y.transpose(0, 1), sample_lengths, h).data.tolist()]
        
        #reward
        if mode == 0:
            #reward_RL_sample = np.array([alpha1 * (os - op) for os, op in zip(output_src, output_sample)])
            #reward_RL_base = np.array([alpha1 * (os - ob) for os, ob in zip(output_src, output_base)])
        
            #print('irony reward sample {} base {}'.format(np.sum(reward_RL_sample) / reward_RL_sample.shape[0], np.sum(reward_RL_base) / reward_RL_base.shape[0]))
            
            tmp1 = np.array([alpha2 * (1 - abs(os - threshold1 - op + threshold2)) for os, op in zip(senti_output_src, senti_output_sample)])
            tmp2 = np.array([alpha2 * (1 - abs(os - threshold1 - ob + threshold2)) for os, ob in zip(senti_output_src, senti_output_base)])
            
            print('senti reward sample {} base {}'.format(np.sum(tmp1) / tmp1.shape[0], np.sum(tmp2) / tmp2.shape[0]))
            
            reward_RL_sample = tmp1
            reward_RL_base = tmp2
        
        elif mode == 1:
            #reward_RL_sample = np.array([alpha1 * (op - os) for os, op in zip(output_src, output_sample)])
            #reward_RL_base = np.array([alpha1 * (ob - os) for os, ob in zip(output_src, output_base)])
        
            #print('irony reward sample {} base {}'.format(np.sum(reward_RL_sample) / reward_RL_sample.shape[0], np.sum(reward_RL_base) / reward_RL_base.shape[0]))
            
            tmp1 = np.array([alpha2 * (1 - abs(os - threshold2 - op + threshold1)) for os, op in zip(senti_output_src, senti_output_sample)])
            tmp2 = np.array([alpha2 * (1 - abs(os - threshold2 - ob + threshold1)) for os, ob in zip(senti_output_src, senti_output_base)])
            
            print('senti reward sample {} base {}'.format(np.sum(tmp1) / tmp1.shape[0], np.sum(tmp2) / tmp2.shape[0]))
            
            reward_RL_sample = tmp1
            reward_RL_base = tmp2
        
        reward_RL_sample = list(reward_RL_sample)
        reward_RL_base = list(reward_RL_base)
        
        return reward_RL_sample, reward_RL_base
    
    # trg
    def batchPGLoss(self, src, lengths, dis, senti, non_senti, mode):
        #src_embedding = self.embedding(src)
        state = self.encoder.init_hidden(src.size(0))
        src_encoding, init_ctx_vec = self.encoder(src, lengths, state)
        
        if mode == 1:
            decoder = self.irony_decoder
        elif mode == 0:
            decoder = self.non_decoder
            
        scores, sample_y = self.decode_sample(decoder, src_encoding, init_ctx_vec)
        base_y = self.decode_baseline(decoder, src_encoding, init_ctx_vec)
        
        sample_y = torch.squeeze(sample_y, 2) #[len, batch]
        base_y = torch.squeeze(base_y, 2) #[len, batch]
        
        sample_lengths = torch.LongTensor(sample_y.size(1)).fill_(self.max_len).cuda()
        base_lengths = torch.LongTensor(base_y.size(1)).fill_(self.max_len).cuda()
        #state = dis.init_hidden()
        #reward_RL_sample = [o[1] for o in dis(sample_y.transpose(0, 1), state).data.tolist()] #
        #reward_RL_base = [o[1] for o in dis(base_y.transpose(0, 1), state).data.tolist()]
        
        # content conservation
        #reward_RL_sample = reward_RL_sample + []
        #reward_RL_base = reward_RL_base + []
        
        reward_RL_sample, reward_RL_base = self.get_reward(src, sample_y, base_y, lengths, sample_lengths, base_lengths, dis, senti, non_senti, mode)
        
        reward_RL_sample = Variable(torch.FloatTensor(reward_RL_sample)).view(1, self.batch_size, 1).cuda()
        reward_RL_base = Variable(torch.FloatTensor(reward_RL_base)).view(1, self.batch_size, 1).cuda()
        
        vocab_mask = torch.ones(self.vocab_size)
        vocab_mask[0] = 0
        cross_entropy_loss = nn.CrossEntropyLoss(weight=vocab_mask, size_average=False, reduce=False).cuda()
        word_loss = cross_entropy_loss(scores.view(-1, scores.size(2)), sample_y.view(-1)).view(scores.size(0), scores.size(1), 1)
        #print(reward_RL_base.size()) #[1,batch,1]
        #print(reward_RL_sample.size())
        #print(word_loss.size()) #[len,batch,1]
        #print(word_loss.data)
        word_loss = -word_loss * (reward_RL_base - reward_RL_sample)
        word_loss = torch.sum(word_loss) # []
        loss = word_loss / self.batch_size
        
        # MLE
        # loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        # loss += loss_fn(scores, trg)
        
        return loss
    
    def decode_sample(self, decoder, src_encoding, init_ctx_vec):
        hidden, cell = init_ctx_vec
        sample_y = []
        scores = []
        for i in range(self.max_len):
            if i == 0:
                y = Variable(torch.LongTensor(self.batch_size).fill_(2)).cuda()
            else:
                y = sample_y[-1].view(-1)
            
            
            score_t, hidden, cell = decoder(y, hidden, cell) # [batch_size, vocab_size]
            cur_y = torch.multinomial(F.softmax(score_t), 1) # [batch_size, 1]
            #print('cur {}'.format(cur_y.size()))
            scores.append(score_t)
            sample_y.append(cur_y.view(-1, 1))
        
        scores = torch.stack(scores) # [len, batch_size, vocab_size]
        sample_y = torch.stack(sample_y) # [len, batch_size, 1]
        return scores, sample_y
    
    
    def decode_baseline(self, decoder, src_encoding, init_ctx_vec):
        hidden, cell = init_ctx_vec
        base_y = []
        for i in range(self.max_len):
            if i == 0:
                y_embed = Variable(torch.LongTensor(self.batch_size).fill_(2)).cuda()
            else:
                y_embed = base_y[-1].view(-1)
            
            
            score_t, hidden, cell = decoder(y_embed, hidden, cell) # [batch_size, vocab_size]
            _, cur_max_y = torch.max(score_t, 1) # [batch_size, 1]
            base_y.append(cur_max_y.view(-1, 1))
        
        base_y = torch.stack(base_y) # [len, batch_size, 1]
        return base_y