#!/usr/bin/env python

import sys
sys.path.append('..')

import json
import torch

from models import predict

for i in [2, 3, 4, 5, 8, 16]:
    for j in [2, 3, 4, 5, 8, 16, 32]:
        sample_filename = '/raid/lingo/abau/random-walks/testset-%d-%d-0/small_sample_extended' % (i, j)
        generator_filename = '/raid/lingo/abau/random-walks/dataset-%d-%d-0/graph.json' % (i, j)

        for l in [2, 4, 8, 16, 32, 64, 128]:
            print('Running', i, j, l)
            model_filename = '/raid/lingo/abau/random-walks/lstm-%d-%d-0-%d/epoch-0.pt' % (i, j, l)
            extractor_filename = '/raid/lingo/abau/random-walks/lstm-%d-%d-0-%d/extractor-model.pt' % (i, j, l)

            param_filename = '/raid/lingo/abau/random-walks/lstm-%d-%d-0-%d/gaussians.json'

            results = predict.run_predictions(
                model_filename,
                extractor_filename,
                param_filename,
                generator_filename,
                sample_filename
            )

            output_filename = '/raid/lingo/abau/random-walks/lstm-%d-%d-0-%d/extended-annotated.json' % (i, j, l)

            with open(output_filename, 'w') as f:
                json.dump(results, f)
