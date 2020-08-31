#!/usr/bin/env python
import sys
sys.path.append('..')

import models.test as test
import json

test_results = {}

for states in [2, 3, 4, 5, 8, 16]:
    for alphabet_size in [2, 3, 4, 5, 8, 16, 32]:
        for random_seed in [0]:
            for train_length in [2, 4, 8, 16, 32, 64, 128]:
                for test_length in [2, 4, 8, 16, 32, 64, 128]:

                    print('Running %d, %d, %d, %d, %d' % (states, alphabet_size, random_seed, train_length, test_length))

                    test_results[str((states, alphabet_size, random_seed, train_length, test_length))] = \
                            test.run_test(
                                '/raid/lingo/abau/random-walks/lstm-%d-%d-%d-%d/epoch-0.pt' % (states, alphabet_size, random_seed, train_length),
                                '/raid/lingo/abau/random-walks/dataset-%d-%d-%d/graph.json' % (states, alphabet_size, random_seed),
                                '/raid/lingo/abau/random-walks/testset-%d-%d-%d/sentences' % (states, alphabet_size, random_seed),

                                min_length = test_length // 2,
                                max_length = test_length
                            )


                    with open('test-results.json', 'w') as f:
                        json.dump(test_results, f)
