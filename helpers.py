# https://github.com/spro/char-rnn.pytorch

import unidecode
import string
import random
import time
import math
import torch
from torch.autograd import Variable
from tqdm import tqdm

# Reading and un-unicode-encoding data

all_characters = string.printable
n_characters = len(all_characters)

encoding_string = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
decoding_dict = {encoding_string[i]: i for i in range(len(encoding_string))}

def read_file(filename):
    """
    Read a file.

    Args:
        filename (str): The filename

    Returns:
        tupel (str, int) containing the contents and length of the file.
    """
    file = unidecode.unidecode(open(filename).read())
    return file, len(file)

def char_tensor(string):
    """
    Turn an alphabet-encoded string into a Torch
    tensor of ints. In files, we encode strings using Latin alphabet
    characters, so what the model reads as [0, 0, 1, 3], we store as
    "aabd".

    Args:
        string (str): the encoded string

    Returns:
        Torch tensor corresponding to the given encoded string.
    """
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        try:
            tensor[c] = decoding_dict[string[c]]
        except:
            continue
    return tensor

def time_since(since):
    """
    Format elapsed time.

    Args:
        since (time.time()): former output of time.time()

    Returns:
        Human-readble elapsed time since `since`.
    """
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def load_dataset(dataset, cuda = True, filter_function = lambda x: True, batch_size = 100):
    """
    Load a dataset generated using one of the tools in `datasets/`. The file should contain alphabet-encoded strings
    separated by newlines, like:
    ```
    aaac
    aba
    abbb
    ```

    Args:
        dataset (str): Filename of the dataset to load
        cuda (bool, optional, default True): Transfer the tensors to GPU?
        filter_function (function (str => bool), optional, default lambda x: True): Filter function
            to only load certain sentences of the dataset, e.g. if you want to only load sentences
            of certain lengths, etc.

    Returns:
        (batch_indices, batches), where:
            - batches is an array of tuples (input, target), where
              input and target are both (batch_size x seq_len) LongTensors
            - batch_indices is an array of 1d LongTensors of size (batch_size),
              mapping individual samples to the line number in the original dataset.
    """
    file, file_len = read_file(dataset)

    sentences = file.split('\n')
    sentences_of_length = {}

    for i, sentence in enumerate(sentences):
        if len(sentence) == 0:
            continue
        if len(sentence) not in sentences_of_length:
            sentences_of_length[len(sentence)] = []
        sentences_of_length[len(sentence)].append((i, sentence))

    def create_batch(sentences):
        chunk_len = len(sentences[0]) - 1
        batch_size = len(sentences)

        inp = torch.LongTensor(batch_size, chunk_len)
        target = torch.LongTensor(batch_size, chunk_len)

        for bi in range(batch_size):
            chunk = sentences[bi]

            inp[bi] = char_tensor(chunk[:-1])
            target[bi] = char_tensor(chunk[1:])

        inp = Variable(inp)
        target = Variable(target)

        if cuda:
            inp = inp.cuda()
            target = target.cuda()

        return inp, target

    def create_batches(f = lambda x: True):
        batches = []
        batch_indices = []

        for length in tqdm(sentences_of_length):

            sentences_here = [(i, sentence) for i, sentence in sentences_of_length[length]
                    if f(sentence)]

            # Divide these up into batches of size at most args.batch_size
            n_sentences = len(sentences_here)
            n_batches = math.ceil(n_sentences / batch_size)

            for bi in range(n_batches):
                inds, sens = zip(*
                        sentences_here[bi * batch_size : (bi + 1) * batch_size])

                batches.append(create_batch(sens))
                batch_indices.append(inds)

        return batch_indices, batches

    return create_batches(filter_function)
