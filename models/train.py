#/usr/bin/env python
# Based on: https://github.com/spro/char-rnn.pytorch

import sys
sys.path.append('..')

import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import os

from tqdm import tqdm

from helpers import *
from model import *

import random_walks
import json

# Parse command line arguments
argparser = argparse.ArgumentParser(
        description="""
        Train an LSTM or GRU on a given dataset generated using the tools in `datasets/`.
        Currently not general; expects to see files in `/raid/lingo/abau/random-walks/`.
        """)
argparser.add_argument('identifier', type=str,
        help="Dataset identifier states-symbols-seed; looks for datasets in /raid/lingo/abau/random-walks/")
argparser.add_argument('--model', type=str, default="gru",
        help="Either 'lstm' or 'gru'; which model to train")
argparser.add_argument('--output', type=str, default="",
        help="Output directory.")
argparser.add_argument('--max_epochs', type=int, default=50000,
        help="""Maximum number of individual examples to train for. Usually trains less than
        this due to early stopping on validation""")
argparser.add_argument('--max_length', type=int, default=128,
        help="Maximum length to filter training sentences; will only train on sentences up to this length.")
argparser.add_argument('--hidden_size', type=int, default=100,
        help="Size of the LSTM/GRU hidden layers")
argparser.add_argument('--n_layers', type=int, default=2,
        help="Number of hidden layers")
argparser.add_argument('--learning_rate', type=float, default=0.01,
        help="Learning rate")
argparser.add_argument('--batch_size', type=int, default=100,
        help="Batch size")
argparser.add_argument('--cuda', action='store_true',
        help="Use CUDA")
args = argparser.parse_args()

def train(inp, target):
    batch_size, seq_len = inp.size()

    hidden = decoder.init_hidden(batch_size)

    if args.cuda:
        if args.model == 'gru':
            hidden = hidden.cuda()
        else:
            hidden = tuple(x.cuda() for x in hidden)

    decoder.zero_grad()
    loss = 0

    for c in range(seq_len):
        output, hidden = decoder(inp[:,c], hidden)
        loss += criterion(output.view(batch_size, -1), target[:,c])

    loss.backward()
    decoder_optimizer.step()

    # Multiply by batch size for proper global averaging.
    return loss.data * batch_size

def save(epoch):
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    save_filename = os.path.join(args.output, 'epoch-%d.pt' % epoch)
    torch.save(decoder, save_filename)

    print('Saved as %s' % save_filename)

def test(decoder, inp, target):
    criterion = nn.CrossEntropyLoss()

    batch_size, seq_len = inp.size()

    with torch.no_grad():
        hidden = decoder.init_hidden(batch_size)
        if inp.is_cuda:
            if decoder.model == 'gru':
                hidden = hidden.cuda()
            else:
                hidden = tuple(x.cuda() for x in hidden)

        loss = 0

        for c in range(seq_len):
            output, hidden = decoder(inp[:,c], hidden)
            loss += criterion(output.view(batch_size, -1), target[:,c])

        return loss.data * batch_size


train_filename = '/raid/lingo/abau/random-walks/dataset-%s/sentences' % (args.identifier,)
valid_filename = '/raid/lingo/abau/random-walks/validation-%s/sentences' % (args.identifier,)
graph_filename = '/raid/lingo/abau/random-walks/dataset-%s/graph.json' % (args.identifier,)

with open(graph_filename) as f:
    true_graph = random_walks.load_from_serialized(
        json.load(f)
    )
    perfect_decoder = random_walks.PerfectPredictor(true_graph)

_, train_data = load_dataset(train_filename, true_graph.alphabet_size,
        filter_function = lambda x: len(x) <= args.max_length)
_, valid_data = load_dataset(valid_filename, true_graph.alphabet_size,
        filter_function = lambda x: len(x) <= args.max_length)

decoder = CharRNN(
    true_graph.alphabet_size,
    args.hidden_size,
    true_graph.alphabet_size,
    model=args.model,
    n_layers=args.n_layers,
)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.learning_rate)
criterion = nn.CrossEntropyLoss()

if args.cuda:
    decoder.cuda()

start = time.time()
all_losses = []

print("Training for at most %d epochs..." % args.max_epochs)

epoch = 0
full_epoch = 0

valid_history = []

# Establish perfect loss
perfect_avg = 0
valid_samples = 0
for inp, target in tqdm(valid_data):
    batch_size, seq_len = inp.size()

    perfect_loss = test(perfect_decoder, inp, target)
    perfect_avg += perfect_loss

    valid_samples += batch_size * seq_len

perfect_avg /= valid_samples

print("Aiming for valid loss %.4f" % perfect_avg)

best_valid_loss = 9999

STOP_THRESH = 5e-3
STALL_THRESH = 1e-4

while epoch < args.max_epochs:
    full_epoch += 1

    loss_avg = 0
    total_samples = 0

    for inp, target in tqdm(train_data):
        batch_size, seq_len = inp.size()

        loss = train(inp, target)

        loss_avg += loss
        total_samples += batch_size * seq_len

        epoch += 1

    with torch.no_grad():
        valid_avg = 0
        valid_samples = 0
        for inp, target in tqdm(valid_data):
            batch_size, seq_len = inp.size()

            loss = test(decoder, inp, target)
            valid_avg += loss

            valid_samples += batch_size * seq_len

        valid_history.append(valid_avg / valid_samples)

    print('[%s (%d %d%%) TRAIN %.4f VALID %.4f PERFECT %.4f]' % (time_since(start), epoch, epoch / args.max_epochs * 100, loss_avg / total_samples, valid_avg / valid_samples, perfect_avg))

    save(full_epoch)

    if valid_avg / valid_samples < best_valid_loss:
        best_valid_oss = valid_avg / valid_samples
        save(0) # "Current best"

    # If we have been within epsilon of perfect for 2 epochs, stop
    if len(valid_history) >= 2 and valid_history[-1] - perfect_avg < STOP_THRESH and valid_history[-2] - perfect_avg < STOP_THRESH:
        print('STOPPING BECAUSE PERFECT')
        break

    # Or if we have not improved in 5 epochs
    if len(valid_history) >= 5 and all(valid_history[-5] - valid_history[-x] < STALL_THRESH for x in range(1, 5)):
        print('STOPPING BECAUSE STALLED')
        break
