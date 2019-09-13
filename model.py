import copy
import json
import math
import re
import collections

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def swish(x):
    return x * torch.sigmoid(x)


ACT_FNS = {
    'relu': nn.ReLU,
    'swish': swish,
    'gelu': gelu
}


class LayerNorm(nn.Module):
    "Construct a layernorm module in the OpenAI style (epsilon inside the square root)."

    def __init__(self, n_state, e=1e-5):
        super(LayerNorm, self).__init__()
        self.g = nn.Parameter(torch.ones(n_state))
        self.b = nn.Parameter(torch.zeros(n_state))
        self.e = e

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.e)
        return self.g * x + self.b


class Conv1D(nn.Module):
    def __init__(self, nf, rf, nx):
        super(Conv1D, self).__init__()
        self.rf = rf
        self.nf = nf
        if rf == 1:  # faster 1x1 conv
            w = torch.empty(nx, nf)
            nn.init.normal_(w, std=0.02)
            self.w = Parameter(w)
            self.b = Parameter(torch.zeros(nf))
        else:  # was used to train LM
            raise NotImplementedError

    def forward(self, x):
        if self.rf == 1:
            size_out = x.size()[:-1] + (self.nf,)
            x = torch.addmm(self.b, x.view(-1, x.size(-1)), self.w)
            x = x.view(*size_out)
        else:
            raise NotImplementedError
        return x


class Attention(nn.Module):
    def __init__(self, nx, n_ctx, cfg, scale=False):
        super(Attention, self).__init__()
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % cfg.n_head == 0
        self.register_buffer('b', torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = cfg.n_head
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(n_state * 3, 1, nx)
        self.c_proj = Conv1D(n_state, 1, nx)
        self.attn_dropout = nn.Dropout(cfg.attn_pdrop)
        self.resid_dropout = nn.Dropout(cfg.resid_pdrop)

    def _attn(self, q, k, v):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        # w = w * self.b + -1e9 * (1 - self.b)  # TF implem method: mask_attn_weights
        # XD: self.b may be larger than w, so we need to crop it
        b = self.b[:, :, :w.size(-2), :w.size(-1)]
#         print('w: {}'.format(w.size()))
#         print('b: {}'.format(b.size()))
        w = w * b + -1e9 * (1 - b)

        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, x):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        a = self._attn(query, key, value)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)
        return a


class MLP(nn.Module):
    def __init__(self, n_state, cfg):  # in MLP: n_state=3072 (4 * n_embd)
        super(MLP, self).__init__()
        nx = cfg.n_embd
        self.c_fc = Conv1D(n_state, 1, nx)
        self.c_proj = Conv1D(nx, 1, n_state)
        self.act = ACT_FNS[cfg.afn]
        self.dropout = nn.Dropout(cfg.resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class Block(nn.Module):
    def __init__(self, n_ctx, cfg, scale=False):
        super(Block, self).__init__()
        nx = cfg.n_embd
        self.attn = Attention(nx, n_ctx, cfg, scale)
        self.ln_1 = LayerNorm(nx)
        self.mlp = MLP(4 * nx, cfg)
        self.ln_2 = LayerNorm(nx)

    def forward(self, x):
        a = self.attn(x)
        n = self.ln_1(x + a)
        m = self.mlp(n)
        h = self.ln_2(n + m)
        return h


class TransformerModel(nn.Module):
    """ Transformer model """

    def __init__(self, cfg, vocab=40990, n_ctx=512):
        super(TransformerModel, self).__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(vocab, cfg.n_embd)
        self.drop = nn.Dropout(cfg.embd_pdrop)
        block = Block(n_ctx, cfg, scale=True)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(cfg.n_layer)])

        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, x):
        # x[batch, n_sequence, n_tokens, seq_or_pos]
        x = x.view(-1, x.size(-2), x.size(-1)).contiguous()  #[batch, n_toknes, 0/1]
        e = self.drop(self.embed(x)) #[batch, n_toknes, 2, 768]

        # Add the position information to the input embeddings
        h = e.sum(dim=2) #[batch, n_tokens, 768]
        for block in self.h:
            h = block(h)
        return h  #[batch, n_toknes, 768]

class SharedTransformerModel(nn.Module):
    """ Transformer model """

    def __init__(self, cfg, vocab=40990, n_ctx=512):
        super(SharedTransformerModel, self).__init__()

        block = Block(n_ctx, cfg, scale=True)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(cfg.n_layer)])


    def forward(self, x):
        # x[batch, n_toknes, 768]
        
        # Add the position information to the input embeddings
        h = x
        for block in self.h:
            h = block(h)
        return h
    
class TransformerDecoder(nn.Module):
    """ Transformer with language model head only """
    def __init__(self, cfg, vocab=40990, n_ctx=512, return_probs=False):
        super(TransformerDecoder, self).__init__()
        self.transformer = SharedTransformerModel(cfg, vocab=vocab, n_ctx=n_ctx)

        self.decoder = nn.Linear(cfg.n_embd, vocab - n_ctx, bias=False)
        self.n_embd = cfg.n_embd
        self.return_probs = return_probs

    def forward(self, x):
        # x [batch, n_toknens, 768]
        h = self.transformer(x) #[batch, n_toknens, 768]

        lm_logits = self.decoder(h)

        return lm_logits, h


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


DEFAULT_CONFIG = dotdict({
    'n_embd': 480,
    'n_head': 12,
    'n_layer': 4,
    'embd_pdrop': 0.5,
    'attn_pdrop': 0.5,
    'resid_pdrop': 0.5,
    'afn': 'gelu',
    'clf_pdrop': 0.5})

DEFAULT_SHARED_CONFIG = dotdict({
    'n_embd': 480,
    'n_head': 12,
    'n_layer': 2,
    'embd_pdrop': 0.5,
    'attn_pdrop': 0.5,
    'resid_pdrop': 0.5,
    'afn': 'gelu',
    'clf_pdrop': 0.5})
