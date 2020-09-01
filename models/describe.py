#!/usr/bin/env python
# https://github.com/spro/char-rnn.pytorch

import sys
sys.path.append('..')

import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import os

from tqdm import tqdm

from helpers import *
from .model import *
from . import model

sys.modules['model'] = model

import math

def describe(model, inp):
    """
    Generate and record hidden states for a given model on a given input.

    Args:
        model (nn.Module): Model to get hidden states for
        inp (batch_size x seq_len LongTensor): Input to run it on

    Returns:
        Array of hidden states, not including the initial zero state vector.
        For an LSTM, this will be an array of tuples (cell, hidden), where each
        of cell and hidden are (num_layers x batch_size x hidden_size).

        Note that this means it is transposed in the following way:
        to get the hidden state of batch (i), character (j),
        at hidden layer (l) you need to get result[j][l][i].
    """
    batch_size, seq_len = inp.size()

    with torch.no_grad():
        hidden = model.init_hidden(batch_size)

        hidden_record = []

        if inp.is_cuda:
            if model.model == 'gru':
                hidden = hidden.cuda()
            else:
                hidden = tuple(x.cuda() for x in hidden)

        for c in range(seq_len):
            output, hidden = model(inp[:,c], hidden)
            hidden_record.append(hidden)

        return hidden_record

def run_description(model, dataset, cuda = True):
    """
    Generate descriptions for a given model file on a given dataset file.

    Args:
        model (str): `.pt` location of the trained model
        dataset (str): `setences` location of the data to test on
        cuda (bool, optional, default True): Use cuda

    Returns:
        Array of batches.
        Each batch is a tuple (indices, positions); indices map the positions
        to lines in the dataset, while each position is a tuple (output, hidden),
        each of which is a tensor (num_layers x batch_size x hidden_state).

        This means the dataset is very weirdly transposed.

        To get the hidden state associated with line (i), position (j), layer (l),
        you need to:
            - find the x, y such that results[x][0][y] = i
            - then take results[x][1][j][y]
    """
    decoder = torch.load(model)

    criterion = nn.CrossEntropyLoss()

    batch_indices, batches = load_dataset(dataset, decoder.n_characters)

    if cuda:
        decoder.cuda()

    result_records = []

    for indices, (inp, target) in tqdm(zip(batch_indices, batches)):
        individual_record = describe(decoder, inp)
        result_records.append((indices, individual_record))

    return result_records
