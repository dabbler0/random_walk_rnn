#!/usr/bin/env/python

import random
import random_walks
import json
import os

encoding_string = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

for i in [2, 3, 4, 5, 8, 16]:
    for j in [2, 3, 4, 5, 8, 16, 32]:
        # Get dataset
        dataset_name = '/raid/lingo/abau/random-walks/testset-%d-%d-0' % (i, j)
        origin_name = '/raid/lingo/abau/random-walks/dataset-%d-%d-0' % (i, j)
        small_sample = os.path.join(dataset_name, 'small_sample')
        graph_file = os.path.join(origin_name, 'graph.json')

        with open(graph_file) as f:
            graph =  random_walks.load_from_serialized(json.load(f))

        alphabet_size = graph.alphabet_size


        with open(small_sample) as f:
            sentences = [x for x in f.read().split('\n') if len(x) > 0]

        new_sentences = []

        for sentence in sentences:
            new_sentences.extend([
                sentence + encoding_string[x] + 'a' for x in range(alphabet_size)
            ])

        dest_name = '/raid/lingo/abau/random-walks/testset-%d-%d-0/small_sample_extended' % (i, j)

        with open(dest_name, 'w') as f:
            f.write('\n'.join(new_sentences))
