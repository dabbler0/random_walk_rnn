#!/usr/bin/env python
import sys
sys.path.append('..')

from datasets import create_transition_dataset

for num_states in [2, 3, 4, 5, 8, 16]:
    for alphabet_size in [2, 3, 4, 5, 8, 16, 32]:
        for original_seed in [0]:
            create_transition_dataset.generate_transition_dataset(
                    '/raid/lingo/abau/random-walks/dataset-%d-%d-%d' %
                    (num_states, alphabet_size, original_seed), # Existing dataset

                    '/raid/lingo/abau/random-walks/lstm-%d-%d-%d-128' %
                    (num_states, alphabet_size, original_seed), # New location

                    seed = 3)
