import sys
sys.path.append('..')

from random_walks import *
import numpy as np
import os
import json
from tqdm import tqdm

def generate_dataset(
        location,
        states = 2,
        symbols = 2,
        num_sentences = 100000,
        min_length = 2,
        max_length = 128,
        name = 'sentences',
        seed = 0):
    """
    Generate a random graph and a dataset from that graph, to a given location. The location should be a directory.
    It will generate `graph.json` for retrieving the underlying generating graph
    as well as `sentences` (or specified name), containing samples of random walks through the graph.

    Args:
        location (str): output location; should be a directory
        states (int): number of states to make in the graph
        symbols (int): number of alphabet symbols to make in the graph
        num_sentences (int): number of sentences to generate
        min_length (int): minimum length of a sentence (sentence lengths uniformly sampled in [min_length, max_length))
        max_length (int): maximum length of a sentence (sentence lengths uniformly sampled in [min_length, max_length))
        seed (int): seed to feed to numpy.random, for reproducibility
        name (str, optional, default 'sentences'): name of the file in the directory to generate new sentences to
    """
    np.random.seed(seed)

    if not os.path.exists(location):
        os.makedirs(location)

    graph = create_random_connected_walk_graph(states, symbols, np.random)

    with open(os.path.join(location, 'graph.json'), 'w') as f:
        print(graph.serialize())
        json.dump(graph.serialize(), f)

    graph.visualize(os.path.join(location, 'graph-image'))

    with open(os.path.join(location, 'sentences'), 'w') as f:
        for _ in tqdm(range(num_sentences)):
            length = np.random.randint(min_length, max_length)

            sentence = graph.generate_of_length(length, np.random)

            f.write(''.join(
                'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'[i] for i in sentence
            ) + '\n')

def generate_dataset_from_model(
        existing_location,
        new_location,
        num_sentences = 100000,
        min_length = 2,
        max_length = 128,
        name = 'sentences'
        seed = 0):
    """
    Load an existing WalkGraph and generate a dataset from that graph, to a given location. The location should be a directory.

    Args:
        existing_location (str): location of the existing graph; should be a directory containing `graph.json`
        new_location (str): output location; should be a directory
        num_sentences (int): number of sentences to generate
        min_length (int): minimum length of a sentence (sentence lengths uniformly sampled in [min_length, max_length))
        max_length (int): maximum length of a sentence (sentence lengths uniformly sampled in [min_length, max_length))
        seed (int): seed to feed to numpy.random, for reproducibility
        name (str, optional, default 'sentences'): name of the file in the directory to generate new sentences to
    """

    np.random.seed(seed)

    if not os.path.exists(new_location):
        os.makedirs(new_location)

    with open(os.path.join(existing_location, 'graph.json'), 'r') as f:
        graph = load_from_serialized(json.load(f))

    with open(os.path.join(new_location, 'sentences'), 'w') as f:
        for _ in tqdm(range(num_sentences)):
            length = np.random.randint(min_length, max_length)

            sentence = graph.generate_of_length(length, np.random)

            f.write(''.join(
                'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'[i] for i in sentence
            ) + '\n')

