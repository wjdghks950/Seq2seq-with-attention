import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

MAX_LEN = 20

class EncoderBRNN(nn.Module):
    # A bidirectional rnn based encoder
    def __init__(self, input_size, hidden_size, emb_size, batch_size=1, num_layers=2, bidir=True):
        super(EncoderBRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.embedding_dim = emb_size
        self.num_layers = num_layers
        self.bidir = bidir
        self.embedding_layer = nn.Embedding(self.input_size, self.embedding_dim)
        self.enc_layer = nn.LSTM(self.embedding_dim, self.hidden_size, num_layers=self.num_layers, bidirectional=self.bidir)

    def forward(self, input, hidden):
        embed = self.embedding_layer(input).view(1, 1, -1)
        output, hidden = self.enc_layer(embed, hidden)
        return output, hidden

    def initHidden(self, device):
        if self.bidir:
            num_stacks = self.num_layers * 2
        else:
            num_stacks = self.num_layers
        return torch.zeros(num_stacks, self.batch_size, self.hidden_size, device=device) #TODO:Dimensionality error

class DecoderRNN(nn.Module):
    # A rnn decoder using seq2seq attention mechanism
    def __init__(self, output_size, hidden_size, emb_size, dropout_p=0.1, max_seq_len=MAX_LEN):
        super(DecoderRNN, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.max_seq_len = max_seq_len
        self.embedding_dim = emb_size
        self.embedding_layer = nn.Embedding(self.output_size, self.embedding_dim)
        '''
         Attentional module takes in the concatenation of following two:
         i) Final hidden state of Encoder(hidden_size)
         ii) Embedded decoder input (embedding_dim)
        '''
        self.attn_layer = nn.Linear((self.hidden_size + self.embedding_dim), self.max_seq_len)
        self.attn_to_dec = nn.Linear((self.hidden_size + self.embedding_dim), self.hidden_size)
        self.dec_layer = nn.LSTM(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, enc_output):
        embed = self.embedding_layer(input).view(1, 1, -1) # dim(1, batch_size) -> dim(1, batch_size, embedding_dim)
        # Attention weight calculation
        print('Decoder input size:', embed.size())
        print('Hidden state size: ', hidden.size())
        attn_input = torch.cat((embed[0], hidden[0]), 1)
        print('Attention input [Embedded decoder input, hidden state] size: ', attn_input.size())
        attn_weight = F.softmax(self.attn_layer(attn_input), dim=1)
        attn_output = torch.bmm(attn_weight.unsqueeze(0), enc_output.unsqueeze(0)) # applying attention weight to context vector
        # Making decoder rnn input
        dec_input = torch.cat((embed[0], attn_output), 1) # concatenate embedded prev dec output and attn_output as dec input
        dec_input = F.relu(self.attn_to_dec(dec_input)) # fully-connected + relu
        dec_input = dec_input.unsqueeze(0)

        output, hidden = self.dec_layer(dec_input, hidden) # output dim(1, batch_size, dec_hidden_size)
        output = F.log_softmax(self.out(output[0]), dim=1)

        return output, hidden, attn_weight