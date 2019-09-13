import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F

from model import TransformerModel, SharedTransformerModel, TransformerDecoder, DEFAULT_CONFIG, DEFAULT_SHARED_CONFIG
import random
import numpy as np

import math
import sys

class Seq2Seq(nn.Module):
    def __init__(self, mask, hps):
        super(Seq2Seq, self).__init__()
        
        self.hps = hps
        self.vocab_size = hps.vocab_size
        self.emb_dim = hps.emb_dim
        self.max_len = hps.max_len
        self.batch_size = hps.batch_size
        self.test_batch_size = hps.test_batch_size
        
        self.mask = mask
        
        args = DEFAULT_CONFIG
        shared_args = DEFAULT_SHARED_CONFIG
        self.irony_encoder = TransformerModel(args, self.vocab_size + self.max_len, self.max_len)
        self.non_encoder = TransformerModel(args, self.vocab_size + self.max_len, self.max_len)
        self.shared_encoder = SharedTransformerModel(shared_args, self.vocab_size + self.max_len, self.max_len)
        self.shared_decoder = SharedTransformerModel(shared_args, self.vocab_size + self.max_len, self.max_len)
        self.irony_decoder = TransformerDecoder(args, self.vocab_size + self.max_len, self.max_len, True)
        self.non_decoder = TransformerDecoder(args, self.vocab_size + self.max_len, self.max_len, True)
    
    def forward(self, src, trg, lengths, tags):
        pass
        
    def test_ae(self, src, tags):
        batch_size = src.size(0)
        max_len = src.size(2)
        trg_vocab_size = self.vocab_size
        
        tag = tags[0].item()
        if tag == 0:
            encoder = self.non_encoder
            decoder = self.non_decoder
        elif tag == 1:
            encoder = self.irony_encoder
            decoder = self.irony_decoder
            
        output = encoder(src)
        

        output = self.shared_encoder(output)
        
        output = self.shared_decoder(output)
        
        output, _ = decoder(output)
        
        bias = torch.FloatTensor(self.test_batch_size, self.vocab_size).fill_(0).cuda()
        
        for i in range(output.size(1)):
            output[:, i, :] += bias
            indices = torch.argmax(output[:, i, :], 1)
            for j in range(self.test_batch_size):
                idx = indices[j]
                if self.mask[idx] == 1:
                    bias[j][idx] = -1e30
        
        del bias
        outputs = output.argmax(2).squeeze(0) #[max_len, ]
        
        return outputs
    
    def test_tran(self, src, tags):
        batch_size = src.size(0)
        max_len = src.size(2)
        trg_vocab_size = self.vocab_size
        
        tag = tags[0].item()
        if tag == 0:
            encoder = self.non_encoder
            decoder = self.irony_decoder
        elif tag == 1:
            encoder = self.irony_encoder
            decoder = self.non_decoder
        
        output = encoder(src)
        
        output = self.shared_encoder(output)
        
        output = self.shared_decoder(output)
        
        output, _ = decoder(output)
        
        bias = torch.FloatTensor(self.test_batch_size, self.vocab_size).fill_(0).cuda()
        
        for i in range(output.size(1)):
            output[:, i, :] += bias
            indices = torch.argmax(output[:, i, :], 1)
            for j in range(self.test_batch_size):
                idx = indices[j]
                if self.mask[idx] == 1:
                    bias[j][idx] = -1e30
        
        del bias
        outputs = output.argmax(2).squeeze(0) #[max_len, ]

        return outputs
    
    def find_first_eos(self, sents):
        # sents [batch, len]
        lengths = []
        for s in sents:
            length = len(s)
            idx = 0
            for w in s:
                if w.item() == 3:
                    break
                idx += 1
            if idx != length:
                idx += 1
            lengths.append(idx)
        return lengths
            
    
    def batchBT(self, src, trg, tags):
        batch_size = src.size(0)
        max_len = src.size(2)
        trg_vocab_size = self.vocab_size
        
        
        tag = tags[0].item()
        if tag == 0:
            encoder = self.non_encoder
            decoder = self.irony_decoder
        elif tag == 1:
            encoder = self.irony_encoder
            decoder = self.non_decoder
        
        output = encoder(src)
        

        output = self.shared_encoder(output)
        
        output = self.shared_decoder(output)
        
        _, h = decoder(output)
        
        output = self.shared_decoder(h)
        
        output, _ = decoder(output)
        
        loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        loss = loss_fn(output.view(-1, trg_vocab_size), trg[:,:,:,0].contiguous().view(-1))
        
        return loss

    
    
    def batchNLLLoss(self, src, trg, tags):
        #src = [batch size,src sent len]
        #trg = [batch size,trg sent len]

        batch_size = src.size(0)
        max_len = src.size(2)
        trg_vocab_size = self.vocab_size

        tag = tags[0].item()
        if tag == 0:
            encoder = self.non_encoder
            decoder = self.non_decoder
        elif tag == 1:
            encoder = self.irony_encoder
            decoder = self.irony_decoder
        
        output = encoder(src)
        

        output = self.shared_encoder(output)
        
        output = self.shared_decoder(output)
        
        output, _ = decoder(output)
        
#         bias = torch.FloatTensor(self.batch_size, self.vocab_size).fill_(0).cuda()
        
#         for i in range(output.size(1)):
#             output[:, i, :] += bias
#             indices = torch.argmax(output[:, i, :], 1)
#             for j in range(self.batch_size):
#                 idx = indices[j]
#                 if self.mask[idx] == 1:
#                     bias[j][idx] = -1e30
#         del bias
        loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        loss = loss_fn(output.view(-1, trg_vocab_size), trg[:,:,:,0].contiguous().view(-1))
        
        return loss
    
    def get_reward(self, src, sample_y, base_y, dis, senti, non_senti, idx2word, mode):
        
        # src [batch, len]
        # sample_y base_y [batch, len]
        
        threshold1 = 0.35  # irony senti
        threshold2 = 0.25  # non senti
        
        
        sample_lengths = self.find_first_eos(sample_y)
        base_lengths = self.find_first_eos(base_y)
        
        sample_zeros = torch.LongTensor(self.batch_size, self.max_len).fill_(0)
        base_zeros = torch.LongTensor(self.batch_size, self.max_len).fill_(0)
        
        for i in range(self.batch_size):
            sample_zeros[i, :sample_lengths[i]] = sample_y[i, :sample_lengths[i]]
            base_zeros[i, :base_lengths[i]] = base_y[i, :base_lengths[i]]
        
        
        sample_y = sample_zeros.cuda()
        base_y = base_zeros.cuda()
        
        del sample_zeros
        del base_zeros
        
        
        # irony
        output_src = F.softmax(dis(src), dim=-1)[:, 1]
        
        output_base = F.softmax(dis(base_y), dim=-1)[:, 1]
        
        output_sample = F.softmax(dis(sample_y), dim=-1)[:, 1]
        
        # senti
        if mode == 0:

            senti_output_src = F.softmax(senti(src), dim=-1)[:, 1] 

            senti_output_base = F.softmax(non_senti(base_y), dim=-1)[:, 1] 

            senti_output_sample = F.softmax(non_senti(sample_y), dim=-1)[:, 1] 
            
        elif mode == 1:

            senti_output_src = F.softmax(non_senti(src), dim=-1)[:, 1] 

            senti_output_base = F.softmax(senti(base_y), dim=-1)[:, 1] 

            senti_output_sample = F.softmax(senti(sample_y), dim=-1)[:, 1]
        
        
        reward_RL_sample = 0
        reward_RL_base = 0
        
        #reward
        if mode == 0:
            tmp1 = output_src - output_sample
            tmp2 = output_src - output_base
            
            del output_src
            del output_sample 
            del output_base
            
            
            tmp3 = 1 - abs(senti_output_src - threshold1 - senti_output_sample + threshold2)
            tmp4 = 1 - abs(senti_output_src - threshold1 - senti_output_base + threshold2)

            
            del senti_output_src
            del senti_output_sample
            del senti_output_base
            
        elif mode == 1:
            tmp1 = output_sample - output_src
            tmp2 = output_base - output_src
            
            del output_src
            del output_sample 
            del output_base
            
            tmp3 = 1 - abs(senti_output_src - threshold1 - senti_output_sample + threshold2)
            tmp4 = 1 - abs(senti_output_src - threshold1 - senti_output_base + threshold2)
            
            del senti_output_src
            del senti_output_sample
            del senti_output_base
                

        beta = 0.5
        reward_RL_sample = (1 + beta * beta) * tmp1 * tmp3 / (beta * beta * tmp3 + tmp1)
        reward_RL_base = (1 + beta * beta) * tmp2 * tmp4 / (beta * beta * tmp4 + tmp2)
        
        reward_RL_sample = Variable(reward_RL_sample, requires_grad = False)
        reward_RL_base = Variable(reward_RL_base, requires_grad = False)
        
        irony_rw_sample = (torch.sum(tmp1) / tmp1.shape[0]).item()
        irony_rw_base = (torch.sum(tmp2) / tmp2.shape[0]).item()
        senti_rw_sample = (torch.sum(tmp3) / tmp3.shape[0]).item()
        senti_rw_base = (torch.sum(tmp4) / tmp4.shape[0]).item()
        
        
        return sample_y, reward_RL_sample, reward_RL_base, irony_rw_sample, irony_rw_base, senti_rw_sample, senti_rw_base
    
    # trg
    def batchPGLoss(self, src, dis, senti, non_senti, idx2word, mode):
        # src [batch, 1, len ,2]
        
        if mode == 1:
            encoder = self.non_encoder
            decoder = self.irony_decoder
        elif mode == 0:
            encoder = self.irony_encoder
            decoder = self.non_decoder
            
        output = encoder(src)
        

        output = self.shared_encoder(output)
        
        output = self.shared_decoder(output)
        
        
        scores, sample_y = self.decode_sample(decoder, output) # [batch, len, vocab] [batch, len]
        base_y = self.decode_baseline(decoder, output) # [batch, len]
        

        with torch.no_grad():
            sample_y, reward_RL_sample, reward_RL_base, irony_rw_sample, irony_rw_base, senti_rw_sample, senti_rw_base = self.get_reward(src[:,:,:,0].squeeze(1), sample_y, base_y, dis, senti, non_senti, idx2word, mode)
        
        reward_RL_sample = reward_RL_sample.view(1, self.batch_size, 1)
        reward_RL_base = reward_RL_base.view(1, self.batch_size, 1)
        
        vocab_mask = torch.ones(self.vocab_size)
        vocab_mask[0] = 0
        cross_entropy_loss = nn.CrossEntropyLoss(weight=vocab_mask, size_average=False, reduce=False).cuda()
        word_loss = cross_entropy_loss(scores.view(-1, scores.size(2)), sample_y.view(-1)).view(scores.size(1), scores.size(0), 1)
        
        word_loss = -word_loss * (reward_RL_base - reward_RL_sample)
        word_loss = torch.sum(word_loss)
        loss = word_loss / self.batch_size
        
        
        return loss, irony_rw_sample, irony_rw_base, senti_rw_sample, senti_rw_base
    
    def decode_sample(self, decoder, output):
        
        output, _ = decoder(output)
        
#         bias = torch.FloatTensor(self.batch_size, self.vocab_size).fill_(0).cuda()
        
#         for i in range(output.size(1)):
#             output[:, i, :] += bias
#             indices = torch.argmax(output[:, i, :], 1)
#             for j in range(self.batch_size):
#                 idx = indices[j]
#                 if self.mask[idx] == 1:
#                     bias[j][idx] = -1e30
        
#         del bias


        s1, s2, s3 = output.size()
        sample_y = torch.LongTensor(s1, s2, 1).fill_(0).cuda()
        for i in range(s2):
            sample_y[:, i, :] = torch.multinomial(F.softmax(output[:, i, :], dim=-1), 1)
        
        return output, sample_y.squeeze(2)
    
    
    def decode_baseline(self, decoder, output):
        
        output, _ = decoder(output)
        
#         bias = torch.FloatTensor(self.batch_size, self.vocab_size).fill_(0).cuda()
        
#         for i in range(output.size(1)):
#             output[:, i, :] += bias
#             indices = torch.argmax(output[:, i, :], 1)
#             for j in range(self.batch_size):
#                 idx = indices[j]
#                 if self.mask[idx] == 1:
#                     bias[j][idx] = -1e30
#         del bias


        base_y = torch.argmax(output, 2)
        
        return base_y