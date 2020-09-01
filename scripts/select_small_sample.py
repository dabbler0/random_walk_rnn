#!/usr/bin/env python

# 8 sentences of each length
import random
import os

for i in [2, 3, 4, 5, 8, 16]:
    for j in [2, 3, 4, 5, 8, 16, 32]:
        # Get dataset
        dataset_name = '/raid/lingo/abau/random-walks/testset-%d-%d-0' % (i, j)

        sentences_of_length = {length: [] for length in range(2, 128)}

        with open(os.path.join(dataset_name, 'sentences')) as f:
            sentences = [x for x in f.read().split('\n') if len(x) > 0]

            for sentence in sentences:
                sentences_of_length[len(sentence)].append(sentence)


        chosen_sentences = []

        for length in range(2, 128):
            chosen_sentences.extend(
                    random.choices(sentences_of_length[length], k=8)
            )

        with open(os.path.join(dataset_name, 'small_sample'), 'w') as f:
            f.write('\n'.join(chosen_sentences))
