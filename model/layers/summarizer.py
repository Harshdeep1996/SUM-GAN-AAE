# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from layers.lstmcell import StackedLSTMCell

class sLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        """Scoring LSTM"""
        ## Takes in a set of features for the frames and return
        ## the scores for each of the frames
        super().__init__()

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, bidirectional=True)
        ## Getting a scalar score out for the hidden size and pass 
        ## through sigmoid to get the probability
        self.out = nn.Sequential(
            nn.Linear(hidden_size * 2, 1),  # bidirection => scalar
            nn.Sigmoid())

    def forward(self, features, init_hidden=None):
        """
        Args:
            features: [seq_len, 1, 500] (compressed pool5 features)
        Return:
            scores: [seq_len, 1]
        """
        ## Flatten the parameters to put them in a contiguous block
        self.lstm.flatten_parameters()
        # [seq_len, 1, hidden_size * 2]
        features, (h_n, c_n) = self.lstm(features)
        # [seq_len, 1]
        scores = self.out(features.squeeze(1))
        return scores


class eLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        """Encoder LSTM"""
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.mu = nn.Linear(hidden_size, hidden_size)
        self.log_var = nn.Linear(hidden_size, hidden_size)

    def forward(self, frame_features):
        """
        Args:
            frame_features: [seq_len, 1, hidden_size]
        Return:
            output: [seq_len, 1, hidden_size]
            last hidden:
                h_last [num_layers=2, 1, hidden_size]
                c_last [num_layers=2, 1, hidden_size]
        """
        self.lstm.flatten_parameters()
        output, (h_last, c_last) = self.lstm(frame_features)

        return output, (h_last, c_last)


class dLSTM(nn.Module):
    def __init__(self, input_size=2048, hidden_size=2048, num_layers=2):
        """Decoder LSTM"""
        super().__init__()

        ## TODO: should this be num_layers, input_size, hidden_size
        self.lstm_cell = StackedLSTMCell(
            num_layers, 2 * input_size, hidden_size)
        self.out = nn.Linear(hidden_size, input_size)

    def forward(self, seq_len, encoder_output, init_hidden):
        """
        Args:
            seq_len: (int)
            encoder_output: [seq_len, 1, hidden_size]
            init_hidden:
                h [num_layers=2, 1, hidden_size]
                c [num_layers=2, 1, hidden_size]
        Return:
            out_features: [seq_len, 1, hidden_size]
        """
        batch_size = init_hidden[0].size(1)
        hidden_size = init_hidden[0].size(2)

        input_step = Variable(torch.zeros(batch_size, hidden_size)).cuda()
        h, c = init_hidden  # (h_0, c_0): last state of eLSTM

        out_features = []
        for i in range(seq_len):
            # last_h: [1, hidden_size] (h from last layer)
            # last_c: [1, hidden_size] (c from last layer)
            # h: [num_layers=2, 1, hidden_size] (h from all layers)
            # c: [num_layers=2, 1, hidden_size] (c from all layers)
            (last_h, last_c), (h, c) = self.lstm_cell(input_step, (h, c))
            input_step = self.out(last_h)
            out_features.append(last_h)
        # list of seq_len '[1, hidden_size]-sized Variables'
        return out_features


class VAE(nn.Module):  
    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.e_lstm = eLSTM(input_size, hidden_size, num_layers)
        self.d_lstm = dLSTM(input_size, hidden_size, num_layers)
        self.softplus = nn.Softplus()

    def reparametrize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = Variable(torch.randn(std.size())).cuda()
        sample = mu + (eps * std)
        return sample.unsqueeze(1)

    def forward(self, features):
        """
        Args:
            features: [seq_len, 1, hidden_size]
        Return:
            decoded_features: [seq_len, 1, hidden_size]
        """
        seq_len = features.size(0)

        # encoder_output: [seq_len, 1, hidden_size]
        # h and c: [num_layers, 1, hidden_size]
        encoder_output, (h, c) = self.e_lstm(features)
        h = h.squeeze(1)
        h_mu = self.e_lstm.mu(h)
        h_log_variance = torch.log(self.softplus(self.e_lstm.log_var(h)))
        h = self.reparametrize(h_mu, h_log_variance)

        # [seq_len, 1, hidden_size]
        ## Get reparametrized hidden state, memory state 'c' remains the same
        decoded_features = self.d_lstm(seq_len, encoder_output, init_hidden=(h, c))
        decoded_features.reverse() ## reverse the sequence
        decoded_features = torch.stack(decoded_features)
        return h_mu, h_log_variance, decoded_features


class Summarizer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.s_lstm = sLSTM(input_size, hidden_size, num_layers)
        self.auto_enc = VAE(input_size, hidden_size, num_layers)

    def forward(self, image_features, uniform=False):
        """
        Args:
            image_features: [seq_len, 1, hidden_size]
        Return:
            scores: [seq_len, 1]
            decoded_features: [seq_len, 1, hidden_size]
        """

        # Apply weights
        # [seq_len, 1]
        scores = weighted_features = None
        if not uniform:
            scores = self.s_lstm(image_features)
        else:
            scores = torch.Tensor(image_features.size(0)).uniform_(0, 1).cuda()

        ## Multiplying by columns since you are weighted the features
        ## for each sequence
        weighted_features = images_features * scores.view(-1,1,1)
        h_mu, h_log_var, decoded_features = self.auto_enc(weighted_features)
        return scores, h_mu, h_log_var, decoded_features


if __name__ == '__main__':
    pass
