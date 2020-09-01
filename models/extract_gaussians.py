import torch
from torch.distributions.multivariate_normal import MultivariateNormal

import argparse
import os
import numpy as np
import math

import json

from tqdm import tqdm

from helpers import *
from model import *
from generate import *

import random_walks

epsilon = 1e-4

class MVGDistribution:
    def __init__(self, mean, cov):
        self.mean = torch.Tensor(mean).cuda()

        cov = torch.Tensor(cov).cuda()

        # For estimating degrees of freedom
        self.rank = torch.matrix_rank(cov)

        self.cov = cov + torch.eye(cov.shape[0]).cuda() * epsilon

        self.inv = torch.inverse(self.cov)

        self.distribution = MultivariateNormal(self.mean, self.cov)

    def mahalanobis(self, hidden_concat):
        diff = hidden_concat - self.mean
        batch_size = diff.shape[0]

        return torch.bmm(
                torch.bmm(diff.unsqueeze(1),
                    self.inv.unsqueeze(0).expand(batch_size, -1, -1)
                ),
                diff.unsqueeze(2)
        ) / self.rank

class DistributionsRecord:
    def __init__(self, params, graph):
        self.graph = graph
        self.distributions = []
        for state in range(graph.states):
            mean, cov = params[str(state)]
            self.distributions.append(MVGDistribution(mean, cov))

    def probs(self, hidden_concat):
        return torch.stack([
            dist.distribution.log_prob(hidden_concat).squeeze()
            for dist in self.distributions
        ], dim=1)

    def mahalanobis(self, hidden_concat):
        return torch.stack([
            dist.mahalanobis(hidden_concat).squeeze()
            for dist in self.distributions
        ], dim=1)

    def evaluate(self, hidden_concat):
        return torch.stack([
            self.probs(hidden_concat),
            self.mahalanobis(hidden_concat)
        ], dim=2)

def predict(decoder, distributions, extractor, inp, target):
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

            states = F.softmax(extractor(hidden_concat), dim=1).cpu() # Output is batch_size x states
            gaussian = distributions.evaluate(hidden_concat) # Output is batch_size x states x 2
            extractions.append(states)
            gaussians.append(gaussian)
            final_output = F.softmax(output.view(batch_size, -1), dim=1).cpu() # Output is batch_size x chars
            outputs.append(final_output), gaussians

        return outputs, extractions

argparser = argparse.ArgumentParser()
argparser.add_argument('description', type=str)
argparser.add_argument('generator', type=str)
argparser.add_argument('params', type=str)
argparser.add_argument('--output', type=str, default='')
args = argparser.parse_args()

with open(args.generator) as f:
    graph = random_walks.load_from_serialized(json.load(f))

with open(args.params) as f:
    params = json.load(f)

    dists = DistributionsRecord(params, graph)

description = torch.load(args.description)

results = {}

def tostr(index):
    return str(index)

for batch in tqdm(description):
    indices, positions = batch

    for index in indices:
        results[tostr(index)] = []

    for j, position in enumerate(positions):

        cell, hidden = position

        sample = torch.cat([cell[0], cell[1],
            hidden[0], hidden[1]], dim=1)

        # batch x states x 2
        evaluation = dists.evaluate(sample).cpu().numpy().tolist()

        for i, row in enumerate(evaluation):
            results[tostr(indices[i])].append(row)

    with open(args.output, 'w') as f:
        json.dump(results, f)
