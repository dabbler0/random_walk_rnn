import sys
sys.path.append('..')

import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import numpy as np

import json

from tqdm import tqdm

from helpers import *

import random_walks

def estimate_gaussians(description, generator, sentences):
    """
    Estimate parameters for multivariate gaussians for distributions
    of what the RNN hidden states are for each ground truth state
    in the random walk.

    Args:
        description (array of batches of hidden states): output of `describe.py`
        generator (WalkGraph): ground truth graph
        sentences (array of tuples of ints): dataset used to generate description

    Returns:
        Dicitionary mapping states to Gaussian parameters (mean, covariance matrix).
    """
    def index_to_state_array(index):
        sentence = sentences[index]

        sentence = tuple(decoding_dict[c] for c in sentence)

        states = tuple(generator.reconstruct_hidden_states(sentence))

        return states

    # TODO: create a random 0-drop-prob placebo generator
    # to make sure that we are better at predicting the real
    # one than the fake one.

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


    buckets = {
            state: [] for state in range(generator.states)
        }

    for labels, sample in total_example_pool:
        # Here sample is a batch_sizex400 tensor
        # labels is a batch_size tensor of longs

        for i, batch in enumerate(sample):
            buckets[labels[i].cpu().numpy().tolist()].append(batch.cpu())

    # Reconcatenate
    buckets = {
        state: torch.stack(buckets[state]) for state in range(generator.states)
    }

    def covariance(M):
        mu = torch.mean(M, dim=0, keepdim=True)
        M -= mu

        n_samples = M.size(0)

        # (dim x sample) * (sample x dim) => dim x dim
        return torch.mm(
            torch.transpose(M, 0, 1),
            M / (n_samples - 1)
        )

    # Now we've separated into buckets, take means and covariances
    return {
        state: (
            torch.mean(buckets[state], dim=0).cpu().numpy().tolist(),
            covariance(buckets[state]).cpu().numpy().tolist()
        ) for state in range(generator.states)
    }

def run_gaussian_estimate(description_file, generator_file, sentences_file, output_file):
    """
    Estimate parameters for multivariate gaussians for distributions
    of what the RNN hidden states are for each ground truth state
    in the random walk, given files containing data.

    Args:
        description_file (str): `.pt` description file generated by `describe.py`
        generator_file (str): `.json` serialized ground truth WalkGraph
        sentences_file (str): `sentences` file of dataset used to generate description
        output_file (str): where to write the output json file
    """
    # Open up the description
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

    file, file_len = read_file(sentences_file)

    sentences = file.split('\n')

    params = estimate_gaussians(description, generator, sentences)

    with open(output_file, 'w') as f:
        json.dump(params, f)
