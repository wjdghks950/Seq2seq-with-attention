import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from preprocess import DataProcess

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LEN = 20
SOS_token = '<\s>'
EOS_token = '<\e>'

class TrainModel():
    def __init__(self, learning_rate=1e-3, batch_size=1, teacher_forcing=0.5, epoch=5000, max_seq_len=MAX_LEN):
        self.opt = {}
        self.opt['lr'] = learning_rate
        self.opt['batch_size'] = batch_size
        self.opt['max_seq_len'] = max_seq_len
        self.opt['teacher_forcing'] = teacher_forcing
        self.opt['epoch'] = epoch

    def train(self, input, target, encoder, decoder, encoder_optim, decoder_optim, criterion):
        enc_optimizer = encoder_optim
        dec_optimizer = decoder_optim
        enc_optimizer.zero_grad()
        dec_optimizer.zero_grad()
        
        pair = (input, target)
        input_len = input.size(0)
        print('Input length: ', input_len)
        target_len = target.size(0)
        enc_output_tensor = torch.zeros(self.opt['max_seq_len'], encoder.hidden_size, device=device)
        enc_hidden = encoder.initHidden()

        for word_idx in range(input_len):
            enc_output, enc_hidden = encoder(input[word_idx], enc_hidden)
            enc_output_tensor[word_idx] = enc_output[0,0]

        dec_input = torch.tensor([[SOS_token]], device=device)
        dec_hidden = enc_hidden

        teacher_forcing_flag = True if random.random() < teacher_forcing else False
        loss = 0

        for ind in range(target_len):
            dec_output, dec_hidden, dec_attn = decoder(dec_input, dec_hidden, enc_output_tensor)
            if teacher_forcing_flag:
                dec_input = target[ind]
            else:
                topv, topi = dec_output.topk(1)
                dec_input = topi.squeeze().detach() #detach from history of outputs as input

            loss += criterion(dec_output, target[ind])
            if decoder_input.item() == EOS_token:
                break
        # Compute gradient for every parameter
        loss.backward()
        # Update the parameters using accumulated gradients
        enc_optimizer.step()
        dec_optimizer.step()

        return loss.item() / target_len
    
    def trainIters(self, lang1, lang2, encoder, decoder):
        enc_optimizer = optim.Adam(encoder.parameters, lr=self.opt['lr'])
        dec_optimizer = optim.Adam(decoder.optimizer, lr=self.opt['lr'])

        d = DataProcess()
        data = d.preprocess()