#!/usr/bin/env python
import sys
sys.path.append('..')

import os
import numpy as np
import math

import json

from tqdm import tqdm

from helpers import *

for S in [2, 3, 4, 5, 8, 16]:
    for A in [2, 3, 4, 5, 8, 16, 32]:
        extractions = '/raid/lingo/abau/random-walks/lstm-%d-%d-0-128/gaussian-extractions.json' % (S, A)

        if not os.path.exists(extractions):
            continue

        with open(extractions) as f:
            data = json.load(f)

        avg_best = 0
        total = 0
        for key in data:
            mahalanobis = [x[1] for x in data[key][-2]]
            best = min(mahalanobis)
            avg_best += best
            total += 1

        print(S, A, avg_best / total)
