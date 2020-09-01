#!/usr/bin/env python

import sys
sys.path.append('..')

import json
import torch

from models import estimate_gaussians

for states in [2, 3, 4, 5, 8, 16]:
    for alphabet_size in [2, 3, 4, 5, 8, 16, 32]:
        for random_seed in [0]:
            for train_length in [128]:
                print('Running %d, %d, %d, %d' % (states, alphabet_size, random_seed, train_length))

                estimate_gaussians.run_gaussian_estimate(
                    '/raid/lingo/abau/random-walks/lstm-%d-%d-%d-%d/description.pt' %
                        (states, alphabet_size, random_seed, train_length),
                    '/raid/lingo/abau/random-walks/dataset-%d-%d-%d/graph.json' % (states, alphabet_size, random_seed),
                    '/raid/lingo/abau/random-walks/testset-%d-%d-%d/sentences' % (states, alphabet_size, random_seed),
                    '/raid/lingo/abau/random-walks/lstm-%d-%d-%d-%d/gaussians.json' %
                        (states, alphabet_size, random_seed, train_length),
                )
