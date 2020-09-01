import sys
sys.path.append('..')

import torch
from torch.distributions.multivariate_normal import MultivariateNormal

import argparse
import os
import numpy as np
import math

import json

from tqdm import tqdm

from helpers import *
from .model import *
from .gaussians import *

import random_walks

def run_prediction(description_file, generator_file, params_file, output):
    """
    Create a dump file of Gaussian log-probs and Mahalanobis distances for each
    position in a description file. Writes a dict mapping (sentence index => array of size seq_len x states x 2).

    Args:
        description_file (str): `.pt` description file output of `describe.py`
        generator_file (str): `.json` underlying WalkGraph
        params_file (str): `.json` Gaussian estimated parameters output of `estimate_gaussians.py`
        output (str): output file
    """
    with open(generator_file) as f:
        graph = random_walks.load_from_serialized(json.load(f))

    with open(params_file) as f:
        params = json.load(f)

        dists = DistributionsRecord(params, graph)

    description = torch.load(description_file)

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

        with open(output, 'w') as f:
            json.dump(results, f)
