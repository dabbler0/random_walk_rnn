#!/usr/bin/env python

import sys
sys.path.append('..')

import json
import torch

from models import test_extractor
from models.logistic_regression import *

for states in [2, 3, 4, 5, 8, 16]:
    for alphabet_size in [2, 3, 4, 5, 8, 16, 32]:
        for random_seed in [0]:
            for train_length in [2, 4, 8, 16, 32, 64, 128]:
                print('Running %d, %d, %d, %d' % (states, alphabet_size, random_seed, train_length))

                print(test_extractor.run_extraction_test(
                    '/raid/lingo/abau/random-walks/lstm-%d-%d-%d-%d/description.pt' %
                        (states, alphabet_size, random_seed, train_length),
                    '/raid/lingo/abau/random-walks/dataset-%d-%d-%d/graph.json' % (states, alphabet_size, random_seed),
                    '/raid/lingo/abau/random-walks/testset-%d-%d-%d/sentences' % (states, alphabet_size, random_seed)
                ))
