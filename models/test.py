# https://github.com/spro/char-rnn.pytorch
import sys
sys.path.append('..')

import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import os

import numpy as np

from tqdm import tqdm

from helpers import *
from .model import *
from . import model

sys.modules['model'] = model

import random_walks
import json
import math

def test(model, inp, target):
    """
    Test a single model on a single batch.

    Args:
        model (nn.Module): model to test
        inp (batch_size x seq_len LongTensor): input
        target (batch_size x seq_len LongTensor): expected output

    Returns:
        (scalar) Total summed cross-entropy loss for this batch
    """
    criterion = nn.CrossEntropyLoss()

    batch_size, seq_len = inp.size()

    with torch.no_grad():
        hidden = model.init_hidden(batch_size)
        if inp.is_cuda:
            if model.model == 'gru':
                hidden = hidden.cuda()
            else:
                hidden = tuple(x.cuda() for x in hidden)

        loss = 0

        for c in range(seq_len):
            output, hidden = model(inp[:,c], hidden)
            loss += criterion(output.view(batch_size, -1), target[:,c])

        return loss.data * batch_size

def run_test(model, generator, dataset, min_length = 0, max_length = 128, cuda = True):
    """
    Run a test for a given model against a given dataset.

    Args:
        model (str): `.pt` filename for the model to test
        generator (str): `.json` filename for the underlying graph that generated the dataset
        dataset (str): `sentences` file for the test set
        min_length (int, optional, default 0): minimum length to test against; will ignore test sentences shorter than this
        max_length (int, optional, default 128): maximum length to test against; will ignore test sentences longer than this
        cuda (bool, optional, default True): Use cuda

    Returns:
        (loss, perfect_loss), where `loss` is the average cross-entropy loss per character,
        and `perfect_loss` is the average cross-entropy loss per character of a perfect model
        that knows about the underlying generator.
    """
    decoder = torch.load(model)

    _, batches = load_dataset(dataset, decoder.n_characters, filter_function = lambda x: min_length < len(x) <= max_length)

    with open(generator) as f:
        true_graph = random_walks.load_from_serialized(
            json.load(f)
        )
        perfect_decoder = random_walks.PerfectPredictor(true_graph)

    if cuda:
        decoder.cuda()
        perfect_decoder.cuda()

    loss_avg = 0
    perfect_loss_avg = 0
    total_samples = 0
    for inp, target in tqdm(batches):
        batch_size, seq_len = inp.size()

        loss = test(decoder, inp, target)
        loss_avg += loss

        perfect_loss = test(perfect_decoder, inp, target)
        perfect_loss_avg += perfect_loss

        total_samples += batch_size * seq_len

    return loss_avg.cpu().numpy().tolist(), \
            perfect_loss_avg.cpu().numpy().tolist(), total_samples
