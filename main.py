import sys

import numpy as np
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from seq2seq import Seq2Seq
from cnn_discriminator import CNN_Text
import helpers
import random
from collections import namedtuple

#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

hps_dict = {
            'test_path': 'dumped/test1/',
    
            'vocab_size': 50004,
            'emb_dim': 128,
        
            #dis
            'dis_batch_size': 32,
            'dis_epochs': 5,
            'dis_dropout': 0.5,
            'dis_train_data_path':'data/train_all.txt', 
            'dis_train_label_path':'data/train_all_labels.txt',  
            'dis_dev_data_path':'data/tokens_sample.txt',  
            'dis_dev_label_path':'data/labels_sample.txt',  
            'dis_model_path':'model/dis.model',
    
            #senti_dis
            'senti_epochs': 5,
            'senti_batch_size': 32,
            'senti_dropout': 0.5,
            'senti_train_data_path': 'data/train_data', 
            'senti_train_label_path': 'data/train_label', 
            'senti_dev_data_path': 'data/dev_data', 
            'senti_dev_label_path': 'data/dev_label', 
            'senti_model_path':'model/senti_dis.model',
            
            #non_senti_dis
            'non_senti_epochs': 5,
            'non_senti_batch_size': 32,
            'non_senti_dropout': 0.5,
            'non_senti_train_data_path': 'data/train.csv',
            'non_senti_train_label_path':'data/train_label.csv',
            'non_senti_dev_data_path': 'data/dev.csv',
            'non_senti_dev_label_path': 'data/dev_label.csv',
            'non_senti_model_path':'model/non_senti_dis.model',
            
            #seq2seq
            'batch_size': 32,
            'max_len': 40,
            'seq_epochs': 200000,
            'irony_path': 'data/train_ironys.txt',
            'non_path': 'data/train_non_ironys.txt',
            'seq_model_path': 'model/e{}_seq.model',
    
            #whole
            'epochs': 200000,
            'checkpoint_every': 100,
            'lr': 0.00001,
            'whole_model_path': 'model/e{}_whole.model',
            'vocab_path': 'data/vocab',
    
            #predict
            'test_batch_size': 100,
            'non_test_path': 'data/test_non_ironys.txt',
            'irony_test_path': 'data/test_ironys.txt',
            'i2n_output_path': 'data/whole/e{}_i2n_output.txt',
            'i2n_preds_path': 'data/whole/e{}_i2n_preds.txt',
            'n2i_output_path': 'data/whole/e{}_n2i_output.txt',
            'n2i_preds_path': 'data/whole/e{}_n2i_preds.txt',
            'n2n_output_path': 'data/seq/e{}_n2n_output.txt',
            'n2n_preds_path': 'data/seq/e{}_n2n_preds.txt',
            'i2i_output_path': 'data/seq/e{}_i2i_output.txt',
            'i2i_preds_path': 'data/seq/e{}_i2i_preds.txt',
            'i2n_trans_path': 'data/seq/e{}_i2n_output.txt',
            'n2i_trans_path': 'data/seq/e{}_n2i_output.txt'
           }
hps = namedtuple("HParams", hps_dict.keys())(**hps_dict)

if os.path.exists(hps.test_path) == False:
    os.mkdir(hps.test_path)
    os.mkdir(hps.test_path + 'data')
    os.mkdir(hps.test_path + 'data/seq')
    os.mkdir(hps.test_path + 'data/whole')
    os.mkdir(hps.test_path + 'model')

    
def load_vocab(hps):
    lines = open(hps.test_path + hps.vocab_path, 'r').read().split('\n')
    words = [l.split('\t', 1)[0] for l in lines]
    vocab = {w:i for i, w in enumerate(words)}
    idx2word = {i:w for i, w in enumerate(words)}
    return vocab, idx2word

vocab, idx2word = load_vocab(hps)    
#vocab, idx2word = helpers.make_vocab(hps)

hps = hps._replace(vocab_size = len(vocab))

repeatable_words = ['i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
                   'my', 'your', 'his', 'its', 'our', 'their', 'is', 'was', 'are', 'were', 'be', 'been',
                    'to', 'of', 'from', 'in', 'on', 'for', '.', ',', "'", 's', 'a', 'an', 'and', 'the', 'that']
mask = [1 for i in range(hps.vocab_size)]
for w in repeatable_words:
    mask[vocab[w]] = 0

def save_output(irony_outputs, non_outputs,idx2word, irony_path, non_path, hps):
    ironys = []
    nons = []
    for io, no in zip(irony_outputs, non_outputs):
        ironys.append(' '.join([idx2word[idx] for idx in io]))
        nons.append(' '.join([idx2word[idx] for idx in no]))
    open(hps.test_path + irony_path, 'w', encoding='utf-8').write('\n'.join(ironys))
    open(hps.test_path + non_path, 'w', encoding='utf-8').write('\n'.join(nons))
    print('saved outputs')

def train_non_senti_discriminator(model, opt, epochs, hps):
    model.train()
    
    batch_size = hps.non_senti_batch_size
    checkpoint_every = hps.checkpoint_every
    
    train_data_loader, dev_data_loader = helpers.prepare_non_senti_data(hps, vocab)

    for epoch in range(epochs):
        total_loss = 0
        total_acc = 0
        for i, data in enumerate(train_data_loader):

            input, label = data
            input, label = input.cuda(), label.cuda()
            batch_size = input.size(0)
            
            
            opt.zero_grad()
            out = model(input)
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(out, label)
            loss.backward()
            opt.step()
            
            total_loss += loss.item()
            
            out = torch.argmax(out, 1)
            total_acc += np.sum([1 if o == l else 0 for o, l in zip(out, label)]) / batch_size
            
            if (i + 1) % checkpoint_every == 0:
                print('epoch {} batch {} avg_loss {} avg_acc {}'.
                      format(epoch, i, total_loss / checkpoint_every, total_acc / checkpoint_every))
                total_loss = 0
                total_acc = 0
    
    print('validing')
    model.eval()

    batch_num = 0
    valid_total_acc = 0
    for data in dev_data_loader:
        batch_num += 1
        input, label = data
        input, label = input.cuda(), label.cuda()
        batch_size = input.size(0)
        pred = model(input)
        
        pred = torch.argmax(pred, 1)
        valid_total_acc += np.sum([1 if p == l else 0 for p, l in zip(pred, label)]) / batch_size

    print('val_acc={}'.format(valid_total_acc / batch_num))


def train_senti_discriminator(model, opt, epochs, hps):
    model.train()
    
    batch_size = hps.senti_batch_size
    checkpoint_every = hps.checkpoint_every
    
    train_data_loader, dev_data_loader = helpers.prepare_senti_data(hps, vocab)

    for epoch in range(epochs):
        total_loss = 0
        total_acc = 0
        for i, data in enumerate(train_data_loader):

            input,  label = data
            input,  label = input.cuda(), label.cuda()
            batch_size = input.size(0)
            
            
            opt.zero_grad()
            out = model(input)
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(out, label)
            loss.backward()
            opt.step()
            
            total_loss += loss.item()
            
            out = torch.argmax(out, 1)
            total_acc += np.sum([1 if o.item() == l.item() else 0 for o, l in zip(out, label)]) / batch_size
            
            if (i + 1) % checkpoint_every == 0:
                print('epoch {} batch {} avg_loss {} avg_acc {}'.
                      format(epoch, i, total_loss / checkpoint_every, total_acc / checkpoint_every))
                total_loss = 0
                total_acc = 0
    
    print('validing')
    model.eval()

    batch_num = 0
    valid_total_acc = 0
    for data in dev_data_loader:
        batch_num += 1
        input, label = data
        input,  label = input.cuda(), label.cuda()
        batch_size = input.size(0)

        pred = model(input)

        pred = torch.argmax(pred, 1)

        valid_total_acc += np.sum([1 if p.item() == l.item() else 0 for p, l in zip(pred, label)]) / batch_size

    print('val_acc={}'.format(valid_total_acc / batch_num))
    
def train_irony_discriminator(model, opt, epochs, hps):
    model.train()
    
    batch_size = hps.dis_batch_size
    checkpoint_every = hps.checkpoint_every
    
    train_data_loader, dev_data_loader = helpers.prepare_discriminator_data(hps, vocab)

    for epoch in range(epochs):
        total_loss = 0
        total_acc = 0
        for i, data in enumerate(train_data_loader):
            
            model.train()
            
            input, label = data
            input, label = input.cuda(), label.cuda()
            batch_size = input.size(0)
            
            opt.zero_grad()
            out = model(input)
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(out, label)
            loss.backward()
            opt.step()
            
            total_loss += loss.item()
            
            out = torch.argmax(out, 1)
            total_acc += np.sum([1 if o.item() == l.item() else 0 for o, l in zip(out, label)]) / batch_size
            
            if (i + 1) % checkpoint_every == 0:
                print('epoch {} batch {} avg_loss {} avg_acc {}'.
                      format(epoch, i, total_loss / checkpoint_every, total_acc / checkpoint_every))
                total_loss = 0
                total_acc = 0
    
    print('validing')
    model.eval()

    batch_num = 0
    valid_total_acc = 0
    for data in dev_data_loader:
        batch_num += 1
        input,  label = data
        input,  label = input.cuda(),  label.cuda()
        batch_size = input.size(0)

        pred = model(input)

        pred = torch.argmax(pred, 1)

        valid_total_acc += np.sum([1 if p.item() == l.item() else 0 for p, l in zip(pred, label)]) / batch_size

    print('val_acc={}'.format(valid_total_acc / batch_num))

    
def train_seq2seq(model, opt, epochs, hps):
    model.train()
    
    batch_size = hps.batch_size
    checkpoint_every = hps.checkpoint_every
    
    irony_data_loader, non_data_loader = helpers.prepare_seq2seq_data(hps,vocab, noise=True)
    irony_data_loader1, non_data_loader1 = helpers.prepare_seq2seq_data(hps,vocab, noise=False)
    test_irony_data_loader, test_non_data_loader = helpers.prepare_test_data(hps,vocab)
    
    for epoch in range(epochs):
        model.train()
        
        # auto encoder
        total_loss = 0
        
        for i, data in enumerate(irony_data_loader):
            input, tag = data
            new_input = torch.LongTensor(hps.batch_size, 1, hps.max_len, 2).fill_(0)
            new_input[:, 0, :, 0] = input
            new_input[:, 0, :, 1] = torch.LongTensor(np.arange(hps.vocab_size, hps.vocab_size + hps.max_len))
            input, tag = new_input.cuda(), tag.cuda()

            opt.zero_grad()
            loss = model.batchNLLLoss(input, input, tag)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            
            if (i + 1) % checkpoint_every == 0:
                print('epoch {} irony ae batch {} avg_loss {}'.
                      format(epoch, i, total_loss / checkpoint_every))
                total_loss = 0
        
        total_loss = 0
        
        for i, data in enumerate(non_data_loader):
            input, tag = data
            new_input = torch.LongTensor(hps.batch_size, 1, hps.max_len, 2)
            new_input[:, 0, :, 0] = input
            new_input[:, 0, :, 1] = torch.LongTensor(np.arange(hps.vocab_size, hps.vocab_size + hps.max_len))
            input, tag = new_input.cuda(), tag.cuda()
            
            opt.zero_grad()
            loss = model.batchNLLLoss(input, input, tag)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            
            if (i + 1) % checkpoint_every == 0:
                print('epoch {} non ae batch {} avg_loss {}'.
                      format(epoch, i, total_loss / checkpoint_every))
                total_loss = 0
        
        # back translation
        total_loss = 0
        for i, data in enumerate(irony_data_loader1):
            input, tag = data
            new_input = torch.LongTensor(hps.batch_size, 1, hps.max_len, 2)
            new_input[:, 0, :, 0] = input
            new_input[:, 0, :, 1] = torch.LongTensor(np.arange(hps.vocab_size, hps.vocab_size + hps.max_len))
            input, tag = new_input.cuda(), tag.cuda()
            
            opt.zero_grad()
            loss = model.batchBT(input, input, tag)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            
            if (i + 1) % checkpoint_every == 0:
                print('epoch {} irony bt batch {} avg_loss {}'.
                      format(epoch, i, total_loss / checkpoint_every))
                total_loss = 0
        
        total_loss = 0
        
        for i, data in enumerate(non_data_loader1):
            input, tag = data
            new_input = torch.LongTensor(hps.batch_size, 1, hps.max_len, 2)
            new_input[:, 0, :, 0] = input
            new_input[:, 0, :, 1] = torch.LongTensor(np.arange(hps.vocab_size, hps.vocab_size + hps.max_len))
            input, tag = new_input.cuda(), tag.cuda()
            
            opt.zero_grad()
            loss = model.batchBT(input, input, tag)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            
            if (i + 1) % checkpoint_every == 0:
                print('epoch {} non bt batch {} avg_loss {}'.
                      format(epoch, i, total_loss / checkpoint_every))
                total_loss = 0
        
    
        if epoch % 3 == 2:
            print('testing')
            model.eval()
            with torch.no_grad():
                irony_outputs = []
                for i, data in enumerate(test_irony_data_loader):
                    input, tag = data
                    new_input = torch.LongTensor(hps.test_batch_size, 1, hps.max_len, 2)
                    new_input[:, 0, :, 0] = input
                    new_input[:, 0, :, 1] = torch.LongTensor(np.arange(hps.vocab_size, hps.vocab_size + hps.max_len))
                    input, tag = new_input.cuda(), tag.cuda()
                    
                    output = model.test_ae(input, tag)
                    irony_outputs += output.data.tolist()

                non_outputs = []
                for i, data in enumerate(test_non_data_loader):
                    input, tag = data
                    new_input = torch.LongTensor(hps.test_batch_size, 1, hps.max_len, 2)
                    new_input[:, 0, :, 0] = input
                    new_input[:, 0, :, 1] = torch.LongTensor(np.arange(hps.vocab_size, hps.vocab_size + hps.max_len))
                    input, tag = new_input.cuda(), tag.cuda()
                    
                    output = model.test_ae(input, tag)
                    non_outputs += output.data.tolist()
                save_output(irony_outputs, non_outputs, idx2word, hps.i2i_output_path.format(epoch), hps.n2n_output_path.format(epoch), hps)

                irony_outputs = []
                for i, data in enumerate(test_irony_data_loader):
                    input, tag = data
                    new_input = torch.LongTensor(hps.test_batch_size, 1, hps.max_len, 2)
                    new_input[:, 0, :, 0] = input
                    new_input[:, 0, :, 1] = torch.LongTensor(np.arange(hps.vocab_size, hps.vocab_size + hps.max_len))
                    input, tag = new_input.cuda(), tag.cuda()
                    
                    output = seq.test_tran(input, tag)
                    irony_outputs += output.data.tolist()

                non_outputs = []
                for i, data in enumerate(test_non_data_loader):
                    input, tag = data
                    new_input = torch.LongTensor(hps.test_batch_size, 1, hps.max_len, 2)
                    new_input[:, 0, :, 0] = input
                    new_input[:, 0, :, 1] = torch.LongTensor(np.arange(hps.vocab_size, hps.vocab_size + hps.max_len))
                    input, tag = new_input.cuda(), tag.cuda()
                    
                    output = seq.test_tran(input, tag)
                    non_outputs += output.data.tolist()
                save_output(irony_outputs, non_outputs, idx2word, hps.i2n_trans_path.format(epoch), hps.n2i_trans_path.format(epoch), hps)

                print('saving model...')
                torch.save(model.state_dict(), hps.test_path + hps.seq_model_path.format(epoch))
    

def train(seq, dis, senti, non_senti, opt, epochs, hps):
    seq.train()
    dis.eval()
    senti.eval()
    non_senti.eval()
    
    batch_size = hps.batch_size
    checkpoint_every = hps.checkpoint_every
    
    irony_data_loader, non_data_loader = helpers.prepare_seq2seq_data(hps,vocab, noise=False)
    test_irony_data_loader, test_non_data_loader = helpers.prepare_test_data(hps,vocab)
    
    p = 200
    
    for epoch in range(epochs):
        seq.train()
        total_loss = 0
        bt_loss = 0
        bt_cnt = 0

        irony_rw_sample = irony_rw_base = senti_rw_sample = senti_rw_base = 0
        for i, data in enumerate(irony_data_loader):
            
            
            input, tag = data
            new_input = torch.LongTensor(hps.batch_size, 1, hps.max_len, 2).fill_(0)
            new_input[:, 0, :, 0] = input
            new_input[:, 0, :, 1] = torch.LongTensor(np.arange(hps.vocab_size, hps.vocab_size + hps.max_len))
            input, tag = new_input.cuda(), tag.cuda()

            opt.zero_grad()
            loss, r1, r2, r3, r4 = seq.batchPGLoss(input, dis, senti, non_senti, idx2word, 0)

            irony_rw_sample += r1
            irony_rw_base += r2
            senti_rw_sample += r3
            senti_rw_base += r4
           
            
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(seq.parameters(), 5.0)
            opt.step()
            total_loss += loss.item()
            
            
            if i % p == 0:
                input, tag = data
                new_input = torch.LongTensor(hps.batch_size, 1, hps.max_len, 2).fill_(0)
                new_input[:, 0, :, 0] = input
                new_input[:, 0, :, 1] = torch.LongTensor(np.arange(hps.vocab_size, hps.vocab_size + hps.max_len))
                input, tag = new_input.cuda(), tag.cuda()

                opt.zero_grad()
                loss = seq.batchBT(input, input, tag)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(seq.parameters(), 5.0)
                opt.step()
                
                bt_loss += loss.item()
                bt_cnt += 1
                

        
            if (i + 1) % checkpoint_every == 0:
                print('epoch {} irony2non batch {} avg_loss {}'.
                      format(epoch, i, total_loss / checkpoint_every))
                print('irony reward sample {} base {}'.format(irony_rw_sample / checkpoint_every, irony_rw_base / checkpoint_every))
                print('senti reward sample {} base {}'.format(senti_rw_sample / checkpoint_every, senti_rw_base / checkpoint_every))

                if bt_cnt != 0:
                    print('bt loss {}'.format(bt_loss / bt_cnt))
                total_loss = 0
                bt_loss = 0
                bt_cnt = 0
                irony_rw_sample = irony_rw_base = senti_rw_sample = senti_rw_base = 0

                sys.stdout.flush()

        total_loss = 0
        bt_loss = 0
        bt_cnt = 0
        irony_rw_sample = irony_rw_base = senti_rw_sample = senti_rw_base = 0
        for i, data in enumerate(non_data_loader):
            
            tmp_loss = 0
            
            input, tag = data
            new_input = torch.LongTensor(hps.batch_size, 1, hps.max_len, 2).fill_(0)
            new_input[:, 0, :, 0] = input
            new_input[:, 0, :, 1] = torch.LongTensor(np.arange(hps.vocab_size, hps.vocab_size + hps.max_len))
            input, tag = new_input.cuda(), tag.cuda()

            opt.zero_grad()
            loss, r1, r2, r3, r4 = seq.batchPGLoss(input, dis, senti, non_senti, idx2word, 1)

            irony_rw_sample += r1
            irony_rw_base += r2
            senti_rw_sample += r3
            senti_rw_base += r4
            
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(seq.parameters(), 5.0)
            opt.step()
            total_loss += loss.item()
            
            
            if i % p == 0:
                input, tag = data
                new_input = torch.LongTensor(hps.batch_size, 1, hps.max_len, 2).fill_(0)
                new_input[:, 0, :, 0] = input
                new_input[:, 0, :, 1] = torch.LongTensor(np.arange(hps.vocab_size, hps.vocab_size + hps.max_len))
                input, tag = new_input.cuda(), tag.cuda()

                opt.zero_grad()
                loss = seq.batchBT(input, input, tag)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(seq.parameters(), 5.0)
                opt.step()
                
                bt_loss += loss.item()
                bt_cnt += 1


            if (i + 1) % checkpoint_every == 0:
                print('epoch {} non2irony batch {} avg_loss {}'.
                      format(epoch, i, total_loss / checkpoint_every))
                print('irony reward sample {} base {}'.format(irony_rw_sample / checkpoint_every, irony_rw_base / checkpoint_every))
                print('senti reward sample {} base {}'.format(senti_rw_sample / checkpoint_every, senti_rw_base / checkpoint_every))

                if bt_cnt != 0:
                    print('bt loss {}'.format(bt_loss / bt_cnt))
                total_loss = 0
                bt_loss = 0
                bt_cnt = 0
                irony_rw_sample = irony_rw_base = senti_rw_sample = senti_rw_base = 0

                sys.stdout.flush()
        

        
        if epoch % 1 == 0:
            print('testing')
            seq.eval()
            with torch.no_grad():
                irony_outputs = []
                for i, data in enumerate(test_irony_data_loader):
                    input, tag = data
                    new_input = torch.LongTensor(hps.test_batch_size, 1, hps.max_len, 2).fill_(0)
                    new_input[:, 0, :, 0] = input
                    new_input[:, 0, :, 1] = torch.LongTensor(np.arange(hps.vocab_size, hps.vocab_size + hps.max_len))
                    input, tag = new_input.cuda(), tag.cuda()
                    
                    output = seq.test_tran(input, tag)
                    irony_outputs += output.data.tolist()

                non_outputs = []
                for i, data in enumerate(test_non_data_loader):
                    input, tag = data
                    new_input = torch.LongTensor(hps.test_batch_size, 1, hps.max_len, 2).fill_(0)
                    new_input[:, 0, :, 0] = input
                    new_input[:, 0, :, 1] = torch.LongTensor(np.arange(hps.vocab_size, hps.vocab_size + hps.max_len))
                    input, tag = new_input.cuda(), tag.cuda()
                    
                    output = seq.test_tran(input, tag)
                    non_outputs += output.data.tolist()
                save_output(irony_outputs, non_outputs, idx2word, hps.i2n_output_path.format(epoch), hps.n2i_output_path.format(epoch), hps)
                print('saving model...')
                torch.save(seq.state_dict(), hps.test_path + hps.whole_model_path.format(epoch))


if __name__ == '__main__':
    seq = Seq2Seq(mask, hps).cuda()

    dis = CNN_Text(hps, hps.dis_dropout).cuda()

    senti = CNN_Text(hps, hps.senti_dropout).cuda()

    non_senti = CNN_Text(hps, hps.non_senti_dropout).cuda()

    # uncomment when RL training
#     seq.load_state_dict(torch.load(hps.test_path + hps.seq_model_path.format(5)))
    dis.load_state_dict(torch.load(hps.test_path + hps.dis_model_path))
    senti.load_state_dict(torch.load(hps.test_path + hps.senti_model_path))
    non_senti.load_state_dict(torch.load(hps.test_path + hps.non_senti_model_path))
    
    # pretain non senti dis
    # sentiment classifier for non-irony
#     print('pretraining non senti discriminator...')
#     non_senti_optimizer = optim.Adam(non_senti.parameters(), lr=hps.lr)
#     train_non_senti_discriminator(non_senti, non_senti_optimizer, hps.non_senti_epochs, hps)
#     print('saving model...')
#     torch.save(non_senti.state_dict(), hps.test_path + hps.non_senti_model_path)
    
    # pretrain senti dis
    # sentiment classifier for irony
#     print('pretraining senti discriminator...')
#     senti_optimizer = optim.Adam(senti.parameters(), lr=hps.lr)
#     train_senti_discriminator(senti, senti_optimizer, hps.senti_epochs, hps)
#     print('saving model...')
#     torch.save(senti.state_dict(), hps.test_path + hps.senti_model_path)

    # pretrain dis
    # irony classifier
#     print('pretraining irony discriminator...')
#     dis_optimizer = optim.Adam(dis.parameters(), lr=hps.lr)
#     train_irony_discriminator(dis, dis_optimizer, hps.dis_epochs, hps)
#     print('saving model...')
#     torch.save(dis.state_dict(), hps.test_path + hps.dis_model_path)
    
    # pretrain seq2seq
    # comment this part when RL training
    print('pretraining seq2seq...')
    seq_optimizer = optim.Adam(seq.parameters(), lr=hps.lr)
    train_seq2seq(seq, seq_optimizer, hps.seq_epochs, hps)
       
#     whole
    print('training...')
    optimizer = optim.Adam(seq.parameters(), lr=hps.lr)
    train(seq, dis, senti, non_senti, optimizer, hps.epochs, hps)
