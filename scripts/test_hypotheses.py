#!/usr/bin/env python

import sys
sys.path.append('..')

from random_walks import *
import numpy as np
import os
import json
import math
from tqdm import tqdm

in_edges_record = {'rank': 0, 'count': 0}
in_edges_char_record = {'rank': 0, 'count': 0}
appearance_record = {'rank': 0, 'count': 0}

def evaluate(
        data_location,
        model_location,
        num_sentences = 500,
        length = 32,
        seed = 0):

    np.random.seed(seed)

    print('Generating to %s' % model_location)

    with open(os.path.join(data_location, 'graph.json'), 'r') as f:
        graph = load_from_serialized(json.load(f))

    with open(os.path.join(data_location, 'state-record.json'), 'r') as f:
        training_record = json.load(f)

    with open(os.path.join(model_location, 'ghost-edges.json'), 'r') as f:
        ghost_edges = json.load(f)

    for state in ghost_edges:
        state = int(state)
        for char in ghost_edges[str(state)]:
            char = int(char)

            # What's the maximum out?
            predicted_state = max(ghost_edges[str(state)][str(char)],
                    key = lambda k: ghost_edges[str(state)][str(char)][k])

            predicted_state = int(predicted_state)

            if ghost_edges[str(state)][str(char)][str(predicted_state)] < 0.8:
                print('(%d, %d) too uncertain, skipping' % (state, char))
                continue

            # For each given fake edge
            if graph.transitions[state][char] == predicted_state:
                print('FAKE EDGE %d-%d-%d' % (state, char, predicted_state))

                measure('in-edges', predicted_state, graph.states, in_edges_record,
                        lambda x: count_in_edges(graph, x)
                )

                measure('in-char-edges', predicted_state, graph.states, in_edges_char_record,
                        lambda x: count_char_edges(graph, char, x)
                )

                measure('appearance-edges', predicted_state, graph.states, appearance_record,
                        lambda x: training_set_ocurrence(graph, training_record, x)
                )

def measure(message, s, n, r, f):
    print(message, f(s), rank(s, n, f), average(n, f))
    r['rank'] += rank(s, n, f)
    r['count'] += 1

def rank(s, n, f):
    return list(sorted(list(range(n)), key = lambda x: -f(x))).index(s) / n

def average(n, f):
    return sum(f(x) for x in range(n)) / n

def count_in_edges(graph, state):
    return len(
        [ (istate, char)
            for istate in range(graph.states)
            for char in range(graph.alphabet_size)
            if graph.transitions[istate][char] == state
        ]
    )

def count_char_edges(graph, char, state):
    return len(
        [
            istate
            for istate in range(graph.states)
            if graph.transitions[istate][char] == state
        ]
    )

def training_set_ocurrence(graph, record, state):
    return record[str(state)] / sum(
            record[str(state)] for state in range(graph.states))

for num_states in [2, 3, 4, 5, 8, 16]:
    for alphabet_size in [2, 3, 4, 5, 8, 16, 32]:
        for original_seed in [0]:
            evaluate(
                    '/raid/lingo/abau/random-walks/dataset-%d-%d-%d' %
                    (num_states, alphabet_size, original_seed), # Graph location

                    '/raid/lingo/abau/random-walks/lstm-%d-%d-%d-128' %
                    (num_states, alphabet_size, original_seed) # Model location
                )

print('IN EDGES:')
print(in_edges_record['rank'] / in_edges_record['count'])
print('IN CHAR EDGES:')
print(in_edges_char_record['rank'] / in_edges_char_record['count'])
print('APPEARANCE:')
print(appearance_record['rank'] / appearance_record['count'])
