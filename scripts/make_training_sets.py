import sys
sys.path.append('..')

from datasets.generate_dataset import generate_dataset

print('TRAINING SETS')

for num_states in [2, 3, 4, 5, 8, 16:
    for alphabet_size in [2, 3, 4, 5, 8, 16, 32]:
        for random_seed in [0, 1, 2]:
            print('%d-%d-%d' % (num_states, alphabet_size, random_seed))
            generate_dataset('/raid/lingo/abau/random-walks/dataset-%d-%d-%d' %
                    (num_states, alphabet_size, random_seed),
                    num_sentences = 100000,
                    states = num_states,
                    symbols = alphabet_size,
                    seed = random_seed)

print('TEST SETS')

for num_states in [2, 3, 4, 5, 8, 16]:
    for alphabet_size in [2, 3, 4, 5, 8, 16, 32]:
        for original_seed in [0, 1, 2]:
            print('%d-%d-%d' % (num_states, alphabet_size, random_seed))
            generate_dataset_from_model(
                    '/raid/lingo/abau/random-walks/dataset-%d-%d-%d' %
                    (num_states, alphabet_size, original_seed), # Existing dataset

                    '/raid/lingo/abau/random-walks/testset-%d-%d-%d' %
                    (num_states, alphabet_size, original_seed), # New location

                    num_sentences = 10000,
                    seed = 3)

print('VALIDATION SETS')

for num_states in [2, 3, 4, 5, 8, 16, 32]:
    for alphabet_size in [2, 3, 4, 5, 8, 16, 32]:
        for original_seed in [0, 1, 2]:
            print('%d-%d-%d' % (num_states, alphabet_size, random_seed))
            generate_dataset_from_model(
                    '/raid/lingo/abau/random-walks/dataset-%d-%d-%d' %
                    (num_states, alphabet_size, original_seed), # Existing dataset

                    '/raid/lingo/abau/random-walks/validation-%d-%d-%d' %
                    (num_states, alphabet_size, original_seed), # New location

                    num_sentences = 100000,
                    seed = 4)
