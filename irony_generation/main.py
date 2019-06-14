import sys

import numpy as np
from itertools import count
import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from seq2seq import Seq2Seq
from irony_discriminator import IronyDiscriminator
from senti_discriminator import SentiDiscriminator
from non_senti_discriminator import NonSentiDiscriminator
from lstm_classfier import LSTMClassifier
import helpers
import random
from collections import namedtuple

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

hps_dict = {
            'test_path': 'dumped/test10/',
    
            #encoder
            'emb_dim': 128,#128,
            'encoder_hid_dim': 128,
            'encoder_n_layers': 2,
            'encoder_dropout': 0.5,
            
            #decoder
            'decoder_hid_dim': 128,
            'decoder_n_layers': 2,
            'decoder_dropout': 0.5,
            'vocab_size': 50004,
        
            #dis  settled 0.67
            'dis_batch_size': 32,
            'dis_hid_dim':128,#128
            'dis_dropout': 0.3,#0.5,
            'dis_n_layers': 1,
            'dis_epochs': 5,#5,
            'dis_train_data_path':'data/tokens_train.txt',
            'dis_train_label_path':'data/labels_train.txt',
            'dis_dev_data_path':'data/tokens_dev.txt',
            'dis_dev_label_path':'data/labels_dev.txt',
            'dis_model_path':'model/dis.model',
    
            #senti_dis settled 0.716
            'senti_epochs': 5,
            'senti_batch_size': 32,
            'senti_hid_dim': 128, #256,
            'senti_dropout': 0.5,
            'senti_n_layers': 1,
            'senti_train_data_path':'data/train_data',
            'senti_train_label_path':'data/train_label',
            'senti_dev_data_path':'data/dev_data',
            'senti_dev_label_path':'data/dev_label',
            'senti_model_path':'model/senti_dis.model',
            
            #non_senti_dis 0.73
            'non_senti_epochs': 5,
            'non_senti_batch_size': 128,
            'non_senti_hid_dim': 128, #256,
            'non_senti_dropout1': 0.4,
            'non_senti_dropout2': 0.5,
            'non_senti_n_layers': 1,
            'non_senti_train_data_path':'data/train.csv',
            'non_senti_train_label_path':'data/train_label.csv',
            'non_senti_dev_data_path':'data/dev.csv',
            'non_senti_dev_label_path':'data/dev_label.csv',
            'non_senti_model_path':'model/non_senti_dis.model',
            
            #seq2seq
            'batch_size': 64,
            'max_len': 40,
            'seq_epochs': 200000,
            'irony_path':'data/not_selected.txt',
            'non_path':'data/non_clash_selected.txt',
            'seq_model_path': 'model/e{}_seq.model',
            'alpha1': 0.00001,
            'alpha2': 0.00001,
            'alpha3': 1.0,
    
            #whole
            'epochs': 200000,
            'checkpoint_every': 100,
            'lr': 0.0001,
            'whole_model_path': 'model/e{}_whole.model',
            'vocab_path': 'data/vocab',
    
            #predict
            'test_batch_size': 1,
            'non_test_path':'data/non_dev.txt',
            'irony_test_path': 'data/not_dev.txt',
            'i2n_output_path': 'data/whole/e{}_i2n_output.txt',
            'i2n_preds_path': 'data/whole/e{}_i2n_preds.txt',
            'n2i_output_path': 'data/whole/e{}_n2i_output.txt',
            'n2i_preds_path': 'data/whole/e{}_n2i_preds.txt',
            'n2n_output_path': 'data/seq/e{}_n2n_output.txt',
            'n2n_preds_path': 'data/seq/e{}_n2n_preds.txt',
            'i2i_output_path': 'data/seq/e{}_i2i_output.txt',
            'i2i_preds_path': 'data/seq/e{}_i2i_preds.txt'
           }
hps = namedtuple("HParams", hps_dict.keys())(**hps_dict)

if os.path.exists(hps.test_path) == False:
    os.mkdir(hps.test_path)
    os.mkdir(hps.test_path + 'data')
    os.mkdir(hps.test_path + 'data/seq')
    os.mkdir(hps.test_path + 'data/whole')
    os.mkdir(hps.test_path + 'model')

vocab, idx2word = helpers.make_vocab(hps)

hps = hps._replace(vocab_size = len(vocab))

def save_output(irony_outputs, non_outputs,idx2word, irony_path, non_path, hps):
    ironys = []
    nons = []
    for io, no in zip(irony_outputs, non_outputs):
        ironys.append(' '.join([idx2word[idx] for idx in io]))
        nons.append(' '.join([idx2word[idx] for idx in no]))
    open(hps.test_path + irony_path, 'w', encoding='utf-8').write('\n'.join(ironys))
    open(hps.test_path + non_path, 'w', encoding='utf-8').write('\n'.join(nons))
    print('saved outputs')
    
def train_senti_discriminator(model, opt, epochs, hps):
    model.train()
    
    batch_size = hps.senti_batch_size
    checkpoint_every = hps.checkpoint_every
    
    train_data_loader, dev_data_loader = helpers.prepare_senti_data(hps, vocab)

    for epoch in range(epochs):
        total_loss = 0
        total_acc = 0
        for i, data in enumerate(train_data_loader):

            input, length, label = data
            input, length, label = input.cuda(), length.cuda(), label.cuda()
            
            
            opt.zero_grad()
            state = model.init_hidden(input.size(0))
            out = model(input, length, state)
            loss_fn = nn.BCELoss()
            loss = loss_fn(out, label)
            loss.backward()
            opt.step()
            
            total_loss += loss.item()
            
            out = torch.argmax(out, 1).tolist()
            label = torch.argmax(label, 1).tolist() 
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
        input, length, label = data
        input, length, label = input.cuda(), length.cuda(), label.cuda()
        state = model.init_hidden(input.size(0))
        pred = model(input, length, state)
        pred = torch.argmax(pred, 1).tolist()
        label = torch.argmax(label, 1).tolist() 
        valid_total_acc += np.sum([1 if p == l else 0 for p, l in zip(pred, label)]) / batch_size
        #print(pred)
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

            input, length, label = data
            input, length, label = input.cuda(), length.cuda(), label.cuda()
            
            
            opt.zero_grad()
            state = model.init_hidden(input.size(0))
            out = model(input, length, state)
            loss_fn = nn.BCELoss()
            loss = loss_fn(out, label)
            loss.backward()
            opt.step()
            
            total_loss += loss.item()
            
            out = torch.argmax(out, 1).tolist()
            label = torch.argmax(label, 1).tolist() 
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
    labels = []
    for data in dev_data_loader:
        batch_num += 1
        input, length, label = data
        input, length, label = input.cuda(), length.cuda(), label.cuda()
        state = model.init_hidden(input.size(0))
        pred = model(input, length, state)
        
        pred = torch.argmax(pred, 1).tolist()
        label = torch.argmax(label, 1).tolist() 
        labels += label
        valid_total_acc += np.sum([1 if p == l else 0 for p, l in zip(pred, label)]) / batch_size
        #print(pred)
    print('val_acc={}'.format(valid_total_acc / batch_num))
    #print(labels)
    
def train_seq2seq(model, opt, epochs, hps):
    model.train()
    
    batch_size = hps.batch_size
    checkpoint_every = hps.checkpoint_every
    
    irony_data_loader, non_data_loader = helpers.prepare_seq2seq_data(hps,vocab)
    test_irony_data_loader, test_non_data_loader = helpers.prepare_test_data(hps,vocab)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for i, data in enumerate(irony_data_loader):
            input, length, tag = data
            input, length, tag = input.cuda(), length.cuda(), tag.cuda()

            opt.zero_grad()
            _, loss = model.batchNLLLoss(input, input, length, tag, 0)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            
            if (i + 1) % checkpoint_every == 0:
                print('epoch {} irony batch {} avg_loss {}'.
                      format(epoch, i, total_loss / checkpoint_every))
                total_loss = 0
        
        total_loss = 0
        
        for i, data in enumerate(non_data_loader):
            input, length, tag = data
            input, length, tag = input.cuda(), length.cuda(), tag.cuda()
            
            opt.zero_grad()
            _, loss = model.batchNLLLoss(input, input, length, tag, 0)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            
            if (i + 1) % checkpoint_every == 0:
                print('epoch {} non batch {} avg_loss {}'.
                      format(epoch, i, total_loss / checkpoint_every))
                total_loss = 0
        
        if epoch % 3 == 2:
            print('testing')
            model.eval()
            irony_outputs = []
            for i, data in enumerate(test_irony_data_loader):
                input, length, tag = data
                input, length, tag = input.cuda(), length.cuda(), tag.cuda()
                output = model.test_ae(input, length, tag)
                irony_outputs.append(output.data.tolist())
            
            non_outputs = []
            for i, data in enumerate(test_non_data_loader):
                input, length, tag = data
                input, length, tag = input.cuda(), length.cuda(), tag.cuda()
                output = model.test_ae(input, length, tag)
                non_outputs.append(output.data.tolist())
            save_output(irony_outputs, non_outputs, idx2word, hps.i2i_output_path.format(epoch), hps.n2n_output_path.format(epoch), hps)
            print('saving model...')
            torch.save(model.state_dict(), hps.test_path + hps.seq_model_path.format(epoch))
        
    
    print('validing')
    model.eval()
    dev_irony_data_loader, dev_non_data_loader = helpers.prepare_dev_data(hps,vocab)

    irony_total_loss = 0
    irony_batch_num = 0
    for i, data in enumerate(dev_irony_data_loader):
        irony_batch_num += 1
        input, length, tag = data
        input, length, tag = input.cuda(), length.cuda(), tag.cuda()

        
        _, loss = model.batchNLLLoss(input, input, length, tag, 0)
        irony_total_loss += loss.item()
    
    non_total_loss = 0
    non_batch_num = 0
    non_outputs = []
    for i, data in enumerate(dev_non_data_loader):
        non_batch_num += 1
        input, length, tag = data
        input, length, tag = input.cuda(), length.cuda(), tag.cuda()

        
        _, loss = model.batchNLLLoss(input, input, length, tag, 0)
        non_total_loss += loss.item()
    
    print('irony_loss {} non_loss {}'.format(irony_total_loss / irony_batch_num, 
                                             non_total_loss / non_batch_num))
    
    #print('testing')
    #test_irony_data_loader, test_non_data_loader = helpers.prepare_test_data(hps,vocab)
    #irony_outputs = []
    #for i, data in enumerate(test_irony_data_loader):
    #    input, length, tag = data
    #    input, length, tag = input.cuda(), length.cuda(), tag.cuda()
    #    output = model.test_ae(input, length, tag)
    #    irony_outputs.append(output.data.tolist())
        
    #non_outputs = []
    #for i, data in enumerate(test_non_data_loader):
    #    input, length, tag = data
    #    input, length, tag = input.cuda(), length.cuda(), tag.cuda()
    #    output = model.test_ae(input, length, tag)
    #    non_outputs.append(output.data.tolist())
    #save_output(irony_outputs, non_outputs, idx2word, hps.i2i_output_path, hps.n2n_output_path, hps)
    
def train(seq, dis, senti, non_senti, opt, epochs, hps):
    seq.train()
    dis.eval()
    senti.eval()
    non_senti.eval()
    
    batch_size = hps.batch_size
    checkpoint_every = hps.checkpoint_every
    
    irony_data_loader, non_data_loader = helpers.prepare_seq2seq_data(hps,vocab)
    test_irony_data_loader, test_non_data_loader = helpers.prepare_test_data(hps,vocab)
    
    for epoch in range(epochs):
        if epoch % 5 == 4:
            seq.train()
            total_loss = 0
            for i, data in enumerate(irony_data_loader):
                input, length, tag = data
                input, length, tag = input.cuda(), length.cuda(), tag.cuda()
                opt.zero_grad()
                loss = seq.batchPGLoss(input, length, dis, senti, non_senti, 0)
                loss.backward()
                opt.step()
                total_loss += loss.item()

                if (i + 1) % checkpoint_every == 0:
                    print('epoch {} irony2non batch {} avg_loss {}'.
                          format(epoch, i, total_loss / checkpoint_every))
                    total_loss = 0

            sys.stdout.flush()

            total_loss = 0
            for i, data in enumerate(non_data_loader):
                input, length, tag = data
                input, length, tag = input.cuda(), length.cuda(), tag.cuda()
                opt.zero_grad()
                loss = seq.batchPGLoss(input, length, dis, senti, non_senti, 1)
                loss.backward()
                opt.step()
                total_loss += loss.item()

                if (i + 1) % checkpoint_every == 0:
                    print('epoch {} non2irony batch {} avg_loss {}'.
                          format(epoch, i, total_loss / checkpoint_every))
                    total_loss = 0

            sys.stdout.flush()
        else:
            seq.train()
            total_loss = 0

            for i, data in enumerate(irony_data_loader):
                input, length, tag = data
                input, length, tag = input.cuda(), length.cuda(), tag.cuda()

                opt.zero_grad()
                _, loss = seq.batchNLLLoss(input, input, length, tag, 0)
                loss.backward()
                opt.step()
                total_loss += loss.item()

                if (i + 1) % checkpoint_every == 0:
                    print('epoch {} irony batch {} avg_loss {}'.
                          format(epoch, i, total_loss / checkpoint_every))
                    total_loss = 0
            
            sys.stdout.flush()
            
            total_loss = 0

            for i, data in enumerate(non_data_loader):
                input, length, tag = data
                input, length, tag = input.cuda(), length.cuda(), tag.cuda()

                opt.zero_grad()
                _, loss = seq.batchNLLLoss(input, input, length, tag, 0)
                loss.backward()
                opt.step()
                total_loss += loss.item()

                if (i + 1) % checkpoint_every == 0:
                    print('epoch {} non batch {} avg_loss {}'.
                          format(epoch, i, total_loss / checkpoint_every))
                    total_loss = 0
                    
            sys.stdout.flush()
        
        if epoch % 5 == 0:
            print('testing')
            seq.eval()
            irony_outputs = []
            for i, data in enumerate(test_irony_data_loader):
                input, length, tag = data
                input, length, tag = input.cuda(), length.cuda(), tag.cuda()
                output = seq.test_tran(input, length, tag)
                irony_outputs.append(output.data.tolist())
            
            non_outputs = []
            for i, data in enumerate(test_non_data_loader):
                input, length, tag = data
                input, length, tag = input.cuda(), length.cuda(), tag.cuda()
                output = seq.test_tran(input, length, tag)
                non_outputs.append(output.data.tolist())
            save_output(irony_outputs, non_outputs, idx2word, hps.i2n_output_path.format(epoch), hps.n2i_output_path.format(epoch), hps)
            print('saving model...')
            torch.save(seq.state_dict(), hps.test_path + hps.whole_model_path.format(epoch))

if __name__ == '__main__':
    seq = Seq2Seq(hps).cuda()
    dis = IronyDiscriminator(hps).cuda()
    senti = SentiDiscriminator(hps).cuda()
    non_senti = NonSentiDiscriminator(hps).cuda()
    
    #seq.load_state_dict(torch.load(hps.test_path + hps.seq_model_path))
    dis.load_state_dict(torch.load(hps.test_path + hps.dis_model_path))
    senti.load_state_dict(torch.load(hps.test_path + hps.senti_model_path))
    non_senti.load_state_dict(torch.load(hps.test_path + hps.non_senti_model_path))
    
#     lstm = LSTMClassifier(hps).cuda()
#     dis_optimizer = optim.Adagrad(dis.parameters())
#     train_irony_discriminator(lstm, dis_optimizer, hps.dis_epochs, hps)

    # pretain non senti dis
#     print('pretraining non senti discriminator...')
#     non_senti_optimizer = optim.Adam(non_senti.parameters(), lr=hps.lr)
#     preds, labels = train_non_senti_discriminator(non_senti, non_senti_optimizer, hps.non_senti_epochs, hps)
    
    # pretrain senti dis
    #print('pretraining senti discriminator...')
    #senti_optimizer = optim.Adagrad(senti.parameters())
    #train_senti_discriminator(senti, senti_optimizer, hps.senti_epochs, hps)
    #print('saving model...')
    #torch.save(senti.state_dict(), hps.test_path + hps.senti_model_path)

    # pretrain dis
    #print('pretraining irony discriminator...')
    #dis_optimizer = optim.Adagrad(dis.parameters())
    #train_irony_discriminator(dis, dis_optimizer, hps.dis_epochs, hps)
    #print('saving model...')
    #torch.save(dis.state_dict(), hps.test_path + hps.dis_model_path)
    
    # pretrain seq2seq
    #print('pretraining seq2seq...')
    #seq_optimizer = optim.Adam(seq.parameters(), lr=hps.lr)
    #train_seq2seq(seq, seq_optimizer, hps.seq_epochs, hps)
       
    # whole
    print('training...')
    optimizer = optim.Adam(seq.parameters(), lr=hps.lr)
    train(seq, dis, senti, non_senti, optimizer, hps.epochs, hps)