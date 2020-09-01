#!/usr/bin/env python
import sys
sys.path.append('..')

import random_walks
import json
from helpers import *

for S in [2, 3, 4, 5, 8, 16, 32]:
    for A in [2, 3, 4, 5, 8, 16, 32]:
        graph_file = '/raid/lingo/abau/random-walks/dataset-%d-%d-0/graph.json' % (S, A)
        sample_file = '/raid/lingo/abau/random-walks/testset-%d-%d-0/small_sample_extended' % (S, A)
        annotations_file = '/raid/lingo/abau/random-walks/lstm-%d-%d-0-%d/extended-annotated.json' % (S, A, 128)

        with open(graph_file) as f:
            graph = random_walks.load_from_serialized(json.load(f))

        with open(sample_file) as f:
            samples = [x for x in f.read().split('\n') if len(x) > 0]

        with open(annotations_file) as f:
            annotations = json.load(f)

        ghost_edges = {
            state: {
                c: {
                    nstate: 0
                    for nstate in range(graph.states)
                    }
                for c in range(graph.alphabet_size)
                }
            for state in range(graph.states)
        }

        ghost_edges_weighted = {
            state: {
                c: {
                    nstate: 0
                    for nstate in range(graph.states)
                    }
                for c in range(graph.alphabet_size)
                }
            for state in range(graph.states)
        }

        certainty = {
            state: {
                c: 0
                for c in range(graph.alphabet_size)
            }
            for state in range(graph.states)
        }

        totals = {
            state: {
                c: 0
                for c in range(graph.alphabet_size)
            }
            for state in range(graph.states)
        }

        for i, sample in enumerate(samples):
            if str(i) not in annotations:
                continue

            sample = [decoding_dict[c] for c in sample]

            states = [s for s in graph.reconstruct_hidden_states(sample)]

            pstates = annotations[str(i)]['extractions'][-1]

            char = sample[-2]
            state = states[-3]

            pstate = max(range(graph.states), key = lambda i: pstates[i])

            # Take the last pstate prediction
            ghost_edges[state][char][pstate] += 1

            for ostate in range(graph.states):
                ghost_edges_weighted[state][char][ostate] += pstates[ostate]

            certainty[state][char] += pstates[pstate]

            totals[state][char] += 1

        for istate in ghost_edges:
            for char in ghost_edges[istate]:
                certainty[istate][char] /= totals[istate][char]
                for state in ghost_edges[istate][char]:
                    ghost_edges[istate][char][state] /= totals[istate][char]
                    ghost_edges_weighted[istate][char][state] /= totals[istate][char]

        print(S, A)
        print(certainty)

        result_file = '/raid/lingo/abau/random-walks/lstm-%d-%d-0-%d/ghost-edges-info.json' % (S, A, 128)

        with open(result_file, 'w') as f:
            json.dump((certainty, ghost_edges, ghost_edges_weighted), f)
