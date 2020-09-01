#!/usr/bin/env python
# https://github.com/spro/char-rnn.pytorch

import sys
sys.path.append('..')

import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.autograd import Variable
import argparse
import os
import numpy as np
import math

import json

from tqdm import tqdm

from helpers import *

import random_walks

# Parse command line arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('description', type=str)
argparser.add_argument('generator', type=str)
argparser.add_argument('dataset', type=str)
argparser.add_argument('params', type=str)
argparser.add_argument('--output', type=str, default="")
argparser.add_argument('--batch_size', type=int, default=100)
argparser.add_argument('--cuda', action='store_true')
args = argparser.parse_args()

distributions = {}
cparams = {}

epsilon = 1e-4

max_states = 0

with open(args.params) as f:
    params = json.load(f)
    max_states = len(params)

    # Move to GPU
    for key in params:
        mean, cov = params[key]

        mean = torch.Tensor(mean).cuda()

        # Get effective dimension
        degs = torch.matrix_rank(torch.Tensor(cov))

        # Add independent perturbation for smoothing purposes
        cov = torch.Tensor(cov).cuda() + torch.eye(len(cov)).cuda() * epsilon

        # Take inverse and store in params
        inv = torch.inverse(cov)

        cparams[key] = mean, inv, degs

        distributions[key] = MultivariateNormal(mean, cov)

# MLE (no Bayes rule) for the mixed gaussian model.
def classify(batch):
    results = torch.stack([
        distributions[str(key)].log_prob(batch)
        for key in range(max_states)
    ])

    maxs, indices = torch.max(results, dim=0)

    return indices

# Mahalanobis distance
def mahalanobis(state_param, batch):
    mean, inv, degs = state_param

    # Diffs has dims batch x n
    diff = batch - mean

    batch_size = diff.shape[0]

    return torch.bmm(
        torch.bmm(diff.unsqueeze(1),
            inv.unsqueeze(0).expand(batch_size, -1, -1)),
        diff.unsqueeze(2)
    ) / degs

with open(args.generator) as f:
    generator = random_walks.load_from_serialized(json.load(f))

file, file_len = read_file(args.dataset)

sentences = file.split('\n')
encoding_string = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
decoding_dict = {c: i for i, c in enumerate(encoding_string)}

def index_to_state_array(index):
    sentence = sentences[index]

    sentence = tuple(decoding_dict[c] for c in sentence)

    states = tuple(generator.reconstruct_hidden_states(sentence))

    return states

description = torch.load(args.description)

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

        label = torch.LongTensor(
            [labels[i][j] for i in range(sample.size(0))]
        )

        if sample.is_cuda:
            label = label.cuda()

        total_example_pool.append((label, sample))

correct = 0
incorrect = 0

inside_mahalanobis = 0
for labels, sample in tqdm(total_example_pool):
    results = classify(sample)

    for i, batch in enumerate(sample):
        inside_mahalanobis += mahalanobis(cparams[str(labels[i].cpu().numpy().tolist())], batch.unsqueeze(0))

    correct += (labels == results).sum()
    incorrect += (labels != results).sum()

print('Average inside mahalanobis', inside_mahalanobis / (correct + incorrect))
print('%d CORRECT, %d INCORRECT -- %f ACCURACY' % (correct, incorrect, float(correct) / float(correct + incorrect)))
