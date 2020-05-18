# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import numpy as np

class Classifier(nn.Module):
    # define all the layers used in model
    def __init__(self, pretrained_embedding, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()
        # embedding layer
        # issue: https://github.com/chenxijun1029/DeepFM_with_PyTorch/issues/1
        _, embedding_dim = pretrained_embedding.shape
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(pretrained_embedding))
        #self.embedding.weight.data.uniform_(-.5, .5)
        self.embedding.weight.requires_grad = True
        # lstm layer
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=bidirectional,
                            dropout=dropout,
                            batch_first=True)

        # Fully connected layer
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.act = nn.LeakyReLU(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim , output_dim)

    def forward(self, text):
        # text = [batch size,sent_length]
        seq_lens = [np.max(np.nonzero(np.array(sent))) for sent in text]
        embedded = self.embedding(text)
        # embedded = [batch size, sent_len, emb_dim]

        # packed sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, seq_lens, batch_first=True, enforce_sorted=False)

        # hidden = [batch size, num layers * num directions,hid dim]
        packed_output, (hidden, _) = self.lstm(packed_embedded)

        # concat the final forward and backward hidden state
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)

        # hidden = [batch size, hidden dim * num directions]
        outputs1 = self.fc1(hidden)
        act1 = self.act(outputs1)
        logits = self.fc2(act1)

        return logits