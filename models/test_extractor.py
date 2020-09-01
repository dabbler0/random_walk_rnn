import sys
sys.path.append('..')

import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import os
import numpy as np

import json

from tqdm import tqdm

from helpers import *
from .model import *
from .logistic_regression import *
from . import logistic_regression
sys.modules['logistic_regression'] = logistic_regression

import random_walks

def test_extractor(description, generator, sentences, extractor_model):
    """
    Train an extractor for the original states given a description file
    (as generated by `describe.py`), a WalkGraph `generator`, and the
    original dataset that the descirption file was generated on.

    Args:
        description (array of batches of hidden states): output of `describe.py`
        generator (WalkGraph): the walk graph to track hidden states on
        sentences (array of tuples of ints): the original dataset that the description is for

    Returns:

    """
    np.random.seed(128)

    def index_to_state_array(index):
        sentence = sentences[index]

        sentence = tuple(decoding_dict[c] for c in sentence)

        states = tuple(generator.reconstruct_hidden_states(sentence))

        return states

    def index_to_comparison_state_array(index):
        sentence = sentences[index]

        sentence = tuple(decoding_dict[c] for c in sentence)

        states = tuple(comparison.reconstruct_hidden_states(sentence))

        return states

    total_example_pool = []

    for batch in tqdm(description):
        indices, positions = batch

        labels = [
            index_to_state_array(index)
            for index in indices
        ]

        for j, position in enumerate(positions):
            cell, hidden = position

            sample = torch.cat([cell[0], cell[1],
                    hidden[0], hidden[1]], dim=1)
            # sample is now a batch_sizex400 tensor
            # we need to attach labels.

            label = torch.LongTensor(
                [labels[i][j] for i in range(sample.size(0))]
            )

            if sample.is_cuda:
                label = label.cuda()

            # One batch.
            total_example_pool.append((label, sample))

    correct = 0
    total = 0
    total_loss = 0

    with torch.no_grad():
        for labels, sample in total_example_pool:
            pred = extractor_model(sample)
            maxs, indices = torch.max(pred, dim=1)
            correct += (indices == labels).sum()
            total += indices.size(0)

    return float(correct) / total

def run_extraction_test(description_file, generator_file, dataset_file):
    """
    Train an pair of extractors for a given description file;
    one that extracts the actual states, and one control one which extracts
    the states of a random graph.

    Writes the results to the same directory as the description file.

    Args:
        description_file (str): `.pt` location of the description file
        generator_file (str): `.json` location of the WalkGraph serialization
        dataset_file (str): `sentences` location of the sentences that the description file is for
    """
    description = torch.load(description_file)

    # Now description will be an array of tuples
    # (indices, record)
    # where indices is an array of indices into the dataset,
    # and record is an array of length seq_len of batched hidden states.

    # Open up the generator
    with open(generator_file) as f:
        generator = random_walks.load_from_serialized(json.load(f))

    # Now, we would like to read the real generator states
    # off the indices. Let's do that.

    file, file_len = read_file(dataset_file)

    sentences = file.split('\n')

    model_location = os.path.split(description_file)[0]

    if os.path.exists(os.path.join(model_location, 'comparison-graph.json')):
        with open(os.path.join(model_location, 'comparison-graph.json')) as f:
            comparison = random_walks.load_from_serialized(json.load(f))
    else:
        comparison = None

    # Test
    true_accuracy = test_extractor(description, generator, sentences,
        torch.load(os.path.join(model_location, 'extractor-model.pt')))

    if comparison is not None and os.path.exists(os.path.join(model_location, 'extractor-comparison.pt')):
        control_accuracy = test_extractor(description, comparison, sentences,
            torch.load(os.path.join(model_location, 'extractor-comparison.pt')))
    else:
        control_accuracy = -1

    return (true_accuracy, control_accuracy)
