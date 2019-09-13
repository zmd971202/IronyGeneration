from collections import Counter
from torch.autograd import Variable
import torch
import numpy as np
import random
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

def add_noise(l):
    words = l.split()
    length = len(words)
    i = 0
    new_words = []
    while i < length:
        a = random.random()
        w = words[i]
        if a <= 0.1:
            i += 1
            continue
        elif a <= 0.2:
            new_words.append(w)
            new_words.append(w)
            i += 1
        elif a <= 0.3 and i != length - 1:
            new_words.append(words[i + 1])
            new_words.append(w)
            i += 2
        else:
            new_words.append(w)
            i += 1
    return ' '.join(new_words)

class MyDataset(Dataset):
    def __init__(self, file_path, tag, word2idx, noise=False, debug=False):
        seqs = open(file_path, "r", encoding="utf-8").readlines()
        
        seqs = list(map(lambda line: line.strip(), seqs))

        if tag == 1:
            labels = list(np.ones((len(seqs),)))
        else:
            labels = list(np.zeros((len(seqs),)))
            
        self.seqs = seqs
        self.labels = labels
        self.num_total_seqs = len(self.seqs)
        self.word2idx = word2idx
        self.noise = noise
        if debug:
            self.seqs = self.seqs[:100]
            self.labels = self.labels[:100]
            self.num_total_seqs = len(self.seqs)

    def __getitem__(self, index):
        seq = self.seqs[index]
        if self.noise:
            seq = add_noise(seq)
        label = self.labels[index]
        seq = self.words2ids(seq)
        return seq, label

    def __len__(self):
        return self.num_total_seqs

    def words2ids(self, sentence):
        tokens = sentence.lower().split()
        sequence = []
        sequence.append(self.word2idx['<sos>'])
        for token in tokens:
            if token in self.word2idx:
                sequence.append(self.word2idx[token])
            else:
                sequence.append(self.word2idx['<unk>'])
        sequence.append(self.word2idx['<eos>'])
        
        sequence = torch.LongTensor(sequence)
        return sequence

class DisDataset(Dataset):
    def __init__(self, data_path, label_path, word2idx, debug=False):
        seqs = open(data_path, "r", encoding="utf-8").readlines()
#         labels = [[1, 0] if int(l) == 0 else [0, 1] for l in open(label_path, "r", encoding="utf-8").read().split('\n')]
        labels = [int(l) for l in open(label_path, "r", encoding="utf-8").read().split('\n')]
        self.ls = [int(l) for l in open(label_path, "r", encoding="utf-8").read().split('\n')]
        seqs = list(map(lambda line: line.strip(), seqs))
        
        self.seqs = seqs
        self.labels = labels
        self.num_total_seqs = len(self.seqs)
        self.word2idx = word2idx
        if debug:
            self.seqs = self.seqs[:100]
            self.labels = self.labels[:100]
            self.num_total_seqs = len(self.seqs)

    def __getitem__(self, index):
        seq = self.seqs[index]
        label = self.labels[index]
        seq = self.words2ids(seq)
        return seq, label

    def __len__(self):
        return self.num_total_seqs

    def words2ids(self, sentence):
        tokens = sentence.lower().split()
        sequence = []
        sequence.append(self.word2idx['<sos>'])
        for token in tokens:
            if token in self.word2idx:
                sequence.append(self.word2idx[token])
            else:
                sequence.append(self.word2idx['<unk>'])
        sequence.append(self.word2idx['<eos>'])
        sequence = torch.LongTensor(sequence)
        return sequence

def make_vocab(hps):
    words = []
    lines = open(hps.dis_train_data_path, 'r').read().split('\n')
    lines += open(hps.dis_dev_data_path, 'r').read().split('\n')
    #lines = open(hps.whole_senti_train_data_path, 'r').read().split('\n')
    #lines += open(hps.whole_senti_dev_data_path, 'r').read().split('\n')
    lines += open(hps.irony_path, 'r').read().split('\n')
    lines += open(hps.non_path, 'r').read().split('\n')
    lines += open(hps.senti_train_data_path, 'r').read().split('\n')
    lines += open(hps.senti_dev_data_path, 'r').read().split('\n')
    lines += open(hps.non_senti_train_data_path, 'r').read().split('\n')
    lines += open(hps.non_senti_dev_data_path, 'r').read().split('\n')
    
    for l in lines:
        words += l.split()
    c = Counter(words)
    top_k_words = sorted(c.keys(), reverse=True, key=c.get)#[:hps.vocab_size - 4]
    words = ['<pad>', '<unk>', '<sos>', '<eos>'] + [w for w in top_k_words if c[w] > 2]
    print('vocab size {}'.format(len(words)))
    vocab = {w:i for i, w in enumerate(words)}
    idx2word = {i:w for i, w in enumerate(words)}
    
    open(hps.test_path + hps.vocab_path, 'w', encoding='utf-8').write('\n'.join([w + '\t' + str(vocab[w]) for w in vocab]))
    return vocab, idx2word

def make_weights_for_balanced_classes(samples, nclasses, PosOverNeg=1):                        
    count = [0] * nclasses                                                      
    for item in samples:                                                         
        count[item] += 1                                                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])
    
    weight_per_class[1] *= PosOverNeg 
    weight = [0] * len(samples)                                              
    for idx, val in enumerate(samples):                                          
        weight[idx] = weight_per_class[val]
    return weight

def collate_fn(data):
    def merge(sequences):
        lengths = torch.LongTensor([len(seq) for seq in sequences])
        padded_seq = torch.zeros(len(sequences), 40, dtype=torch.long)
        for i, seq in enumerate(sequences):
            end = min(40, lengths[i])
            
            padded_seq[i, :end] = seq[:end]

        return padded_seq
    
#     data.sort(key=lambda x: len(x[0]), reverse=True)

    seqs, labels = zip(*data)  # tuples
    seqs = merge(seqs)
    
    labels = torch.LongTensor(list(labels))
    return seqs, labels


def prepare_non_senti_data(hps, vocab):
    print('preparing non senti data...')
    dataset = DisDataset(hps.non_senti_train_data_path, hps.non_senti_train_label_path, vocab, debug=False)
    
    weights = make_weights_for_balanced_classes(dataset.ls, 2, PosOverNeg=1)
    sampler = WeightedRandomSampler(weights, len(weights))
    
    train_data_loader = DataLoader(dataset,\
                             batch_size=hps.non_senti_batch_size,\
                             shuffle=False,\
                             collate_fn=collate_fn, drop_last=False, sampler=sampler)
    
    dataset = DisDataset(hps.non_senti_dev_data_path, hps.non_senti_dev_label_path, vocab, debug=False)
    weights = make_weights_for_balanced_classes(dataset.ls, 2, PosOverNeg=1)
    sampler = WeightedRandomSampler(weights, len(weights))
    dev_data_loader = DataLoader(dataset,\
                             batch_size=hps.non_senti_batch_size,\
                             shuffle=False,\
                             collate_fn=collate_fn, drop_last=False, sampler=sampler)
    return train_data_loader, dev_data_loader

def prepare_senti_data(hps, vocab):
    print('preparing senti data...')
    dataset = DisDataset(hps.senti_train_data_path, hps.senti_train_label_path, vocab, debug=False)
    
    weights = make_weights_for_balanced_classes(dataset.ls, 2, PosOverNeg=1)
    sampler = WeightedRandomSampler(weights, len(weights))
    
    train_data_loader = DataLoader(dataset,\
                             batch_size=hps.senti_batch_size,\
                             shuffle=False,\
                             collate_fn=collate_fn, drop_last=False, sampler=sampler)
    
    dataset = DisDataset(hps.senti_dev_data_path, hps.senti_dev_label_path, vocab, debug=False)
    weights = make_weights_for_balanced_classes(dataset.ls, 2, PosOverNeg=1)
    sampler = WeightedRandomSampler(weights, len(weights))
    dev_data_loader = DataLoader(dataset,\
                             batch_size=hps.senti_batch_size,\
                             shuffle=False,\
                             collate_fn=collate_fn, drop_last=False, sampler=sampler)
    return train_data_loader, dev_data_loader

def prepare_discriminator_data(hps, vocab):
    print('preparing dis data...')
    train_dataset = DisDataset(hps.dis_train_data_path, hps.dis_train_label_path, vocab, debug=False)
    
    train_weights = make_weights_for_balanced_classes(train_dataset.ls, 2, PosOverNeg=1)
    train_sampler = WeightedRandomSampler(train_weights, len(train_weights))
    
    train_data_loader = DataLoader(train_dataset,\
                             batch_size=hps.dis_batch_size,\
                             shuffle=False,\
                             collate_fn=collate_fn, drop_last=False, sampler=train_sampler)
    
    dev_dataset = DisDataset(hps.dis_dev_data_path, hps.dis_dev_label_path, vocab, debug=False)
    dev_weights = make_weights_for_balanced_classes(dev_dataset.ls, 2, PosOverNeg=1)
    dev_sampler = WeightedRandomSampler(dev_weights, len(dev_weights))
    dev_data_loader = DataLoader(dev_dataset,\
                             batch_size=hps.dis_batch_size,\
                             shuffle=False,\
                             collate_fn=collate_fn, drop_last=False, sampler=dev_sampler)
    return train_data_loader, dev_data_loader

    
def prepare_seq2seq_data(hps,vocab, noise=False):
    print('preparing seq data...')
    irony_dataset = MyDataset(hps.irony_path, 1, vocab, noise, debug=False)
    irony_data_loader = DataLoader(irony_dataset,\
                             batch_size=hps.batch_size,\
                             shuffle=True,\
                             collate_fn=collate_fn, drop_last=True)
    
    non_dataset = MyDataset(hps.non_path, 0, vocab, noise, debug=False)
    non_data_loader = DataLoader(non_dataset,\
                             batch_size=hps.batch_size,\
                             shuffle=True,\
                             collate_fn=collate_fn, drop_last=True)
    return irony_data_loader, non_data_loader

def prepare_test_data(hps, vocab):
    print('preparing test data...')
    non_dataset = MyDataset(hps.non_test_path, 0, vocab, False, debug=False)
    non_data_loader = DataLoader(non_dataset,\
                             batch_size=hps.test_batch_size,\
                             shuffle=False,\
                             collate_fn=collate_fn, drop_last=False)
    
    irony_dataset = MyDataset(hps.irony_test_path, 1, vocab, False, debug=False)
    irony_data_loader = DataLoader(irony_dataset,\
                             batch_size=hps.test_batch_size,\
                             shuffle=False,\
                             collate_fn=collate_fn, drop_last=False)
    return irony_data_loader, non_data_loader