import sys
sys.path.append('..')

import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import os

from torch.nn import functional as F

import numpy as np

from tqdm import tqdm

from helpers import *
from .model import *
from .gaussians import *

import random_walks
import json
import math

epsilon = 1e-4

def predict(decoder, extractor, distributions, inp):
    """
    Get model predictions and inferred hidden states by both logistic
    and Gaussian methods for a given model.

    Args:
        decoder (nn.Module): model to run
        extractor (nn.Module): logistic regression extractor model
        distributions (DistributionsRecord): Gaussian distributions for hidden states
        inp (batch_size x seq_len LongTensor): input to run on

    Returns:
        predictions, extractions, gaussians, where:
            predictions is size batch_size x seq_len x alphabet_size
            extractions is an array of length seq_len with elements of size batch_size x states
            gaussians is an array of length seq_len with elements of size batch_size x states x 2
    """
    criterion = nn.CrossEntropyLoss()

    batch_size, seq_len = inp.size()

    with torch.no_grad():
        hidden = decoder.init_hidden(batch_size)
        if inp.is_cuda:
            if decoder.model == 'gru':
                hidden = hidden.cuda()
            else:
                hidden = tuple(x.cuda() for x in hidden)

        loss = 0

        outputs = []
        extractions = []
        gaussians = []
        for c in range(seq_len):
            output, hidden = decoder(inp[:,c], hidden)

            # These will each be batch_size x 100
            hidden_concat = torch.cat([
                hidden[0][0], hidden[0][1],
                hidden[1][0], hidden[1][1]
            ], dim=1)

            if extractor is not None:
                states = F.softmax(extractor(hidden_concat), dim=1).cpu() # Output is batch_size x states
                extractions.append(states)

            if distributions is not None:
                gaussian = distributions.evaluate(hidden_concat) # Output is batch_size x states x 2
                gaussians.append(gaussian)

            final_output = F.softmax(output.view(batch_size, -1), dim=1).cpu() # Output is batch_size x chars
            outputs.append(final_output)

        return outputs, extractions, gaussians

def run_predictions(model, extractor_file, distributions_file, generator, dataset, min_length = 0, max_length = 128, cuda = True):
    """
    Get model predictions and inferred hidden states by both logistic
    and Gaussian methods for a given model, given file names.

    Args:
        model (str): `.pt` file for model to run
        extractor_file (str): `.pt` file for logistic regression extractor model
        distributions_file (str): `.json` file for Gaussian distributions for hidden states
        generator (str): `.json` file for underlying WalkGraph
        dataset (str): `sentences` file to run predictions on
        min_length (int, optional, default 0): minimum length to run predictions for
        max_length (int, optional, default 128): maximum length to run predictions for
        cuda (int, optional, default True): Use cuda

    Returns:
        Dictionary mapping sentence indices (line indices into the original dataset file)
        to dictionaries {"predictions", "extractions", "gaussian_extractions"}, with each of those
        mapped to ordinary Python arrays of size seq_len x num_states (x 2, in the case of gaussian extractions).
    """
    decoder = torch.load(model)

    if os.path.exists(extractor_file):
        extractor = torch.load(extractor_file)
    else:
        extractor = None

    print('Loading batches for min length %d, max length %d' % (min_length, max_length))
    batch_indices, batches = load_dataset(dataset, decoder.n_characters, filter_function = lambda x: min_length < len(x) <= max_length)

    print('total of %d batches' % (len(batches),))

    with open(generator) as f:
        true_graph = random_walks.load_from_serialized(
            json.load(f)
        )
        perfect_decoder = random_walks.PerfectPredictor(true_graph)

    if os.path.exists(distributions_file):
        with open(distributions_file) as f:
            params = json.load(f)

            distributions = DistributionsRecord(params, true_graph)
    else:
        distributions = None

    if cuda:
        decoder.cuda()
        perfect_decoder.cuda()

    loss_avg = 0
    perfect_loss_avg = 0
    total_samples = 0

    batch_outputs = []
    for indices, (inp, target) in tqdm(zip(batch_indices, batches)):
        batch_size, seq_len = inp.size()

        outputs, extractions, gaussians = predict(decoder, extractor, distributions, inp)
        batch_outputs.append((indices, outputs, extractions, gaussians))

    information = {}
    for indices, outputs, states, gaussians in batch_outputs:
        # Sentence index attached to listified data
        for i, index in enumerate(indices):
            predictions = [
                pred[i].cpu().numpy().tolist()
                for pred in outputs
            ]

            extractions = [
                state[i].cpu().numpy().tolist()
                for state in states
            ]

            gaussian_extractions = [
                gaussian[i].cpu().numpy().tolist()
                for gaussian in gaussians
            ]

            information[index] = {
                'predictions': predictions,
                'extractions': extractions,
                'gaussian_extractions': gaussian_extractions
            }

    return information
