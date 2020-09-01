from random_walks import *
import numpy as np
import os
import json
import math
from tqdm import tqdm

def generate_transition_dataset(
        data_location,
        model_location,
        num_sentences = 500,
        length = 32,
        seed = 0):
    """
    Generate a transition dataset; that is a dataset which, for a given
    transition s1 -a-> s2, always uses that transition (length) characters
    before the end of each sentence, even if that transition isn't in the actual graph.

    Writes the output name `transition_sentences_test` in the model location directory.

    Args:
        data_location (str): directory for the original data containing the underlying WalkGraph, etc. (dataset-X-X-X)
        model_location (str): directory containing the model (lstm-X-X-X-X) and associated ghost edge data
        num_sentences (int, optional, default 500): number of sentences to generate
        length (int, optional, default 32): how many more emissions to make after the given transition
        seed (int, optional, default 0): numpy seed for reproducibility
    """

    np.random.seed(seed)

    print('Generating to %s' % model_location)

    with open(os.path.join(data_location, 'graph.json'), 'r') as f:
        graph = load_from_serialized(json.load(f))

    with open(os.path.join(model_location, 'ghost-edges.json'), 'r') as f:
        ghost_edges = json.load(f)

    for state in range(graph.states):
        for char in range(graph.alphabet_size):
            for predicted_state in range(graph.states):
                # We could use the actual ghost edges found, but here we just
                # generate transition files for every possible transition.
                '''
                predicted_state = max(ghost_edges[str(state)][str(char)],
                        key = lambda k: ghost_edges[str(state)][str(char)][k])

                predicted_state = int(predicted_state)

                if ghost_edges[str(state)][str(char)][str(predicted_state)] < 0.8:
                    print('(%d, %d) too uncertain, skipping' % (state, char))
                    continue
                '''

                with open(os.path.join(model_location,
                    'transition_sentences_test-%d-%d-%d' % (state, char, predicted_state)), 'w') as f:

                    for _ in tqdm(range(num_sentences)):
                        pre_sentence = graph.generate_between_states(
                            np.random,
                            min_length = 8, max_length = 128, # TODO parameters?
                            start_state = 0, end_state = state)
                        post_sentence = tuple(graph.generate(
                            length, np.random, start_state = predicted_state))

                        sentence = pre_sentence + (char,) + post_sentence

                        f.write(''.join(
                            'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'[i] for i in sentence
                        ) + '\n')

