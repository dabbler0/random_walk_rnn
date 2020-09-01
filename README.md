Random Walk RNN Experiments
===========================

This is a collection of code for running experiments training RNNs on datasets generated by random walks through graphs, to see what RNNs extrapolate
about out-of-domain data.

## Creating graphs, data, and models

To run the experiments, you'll first need to create the graphs and datasets. General-purpose code for this is in `datasets/create_dataset.py`, with an example in `scripts/make_training_sets.py`.

On `dgx2`, the relevant datasets are in files:
  - `/raid/lingo/abau/random-walks/dataset-{STATES}-{ALPHABET_SIZE}-{RANDOM_SEED}` for states in {2, 3, 4, 5, 8, 16}, alphabet size in {2, 3, 4, 5, 8, 16, 32}, and random seed in {0, 1, 2}
  - `/raid/lingo/abau/random-walks/validation-{STATES}-{ALPHABET-SIZE}-{RANDOM_SEED}` for the same
  - `/raid/lingo/abau/random-walks/testset-{STATES}-{ALPHABET-SIZE}-{RANDOM_SEED}` for the same

You'll then need to train some models. We use a training script modified from [https://github.com/spro/char-rnn.pytorch](https://github.com/spro/char-rnn.pytorch); run `models/train.py` and follow its help. For an example see `scripts/train.sh`.

On `dgx2`, the relevant models are in files:
  - `/raid/lingo/abau/random-walks/lstm-{STATES}-{ALPHABET_SIZE}-{RANDOM_SEED}-{MAX_TRAIN_LENGTH}` for all the datasets given above and max train length in {2, 4, 8, 16, 32, 64, 128}.

## Testing the models on various lengths of test data

General-purpose code for testing models in `models/test.py`; for a usage example see `scripts/run_tests.py`.

On `dgx2`, the resulting test numbers are in
  - `/raid/lingo/abau/random-walks/test-results.json`.

## Extracting graph states from RNN hidden states (TODO cleanup)

We can extract graph states from RNN hidden states in two different ways; through logistic regression or through a mixture of multivariate Gaussians.

To do this, we first need to dump the RNN hidden states at test time. General-purpose code for this is in `models/describe.py`; for a usage example see `models/run_descriptions.py`.

On `dgx2`, the relevant dumps are in:
  - `/raid/lingo/abau/random-walks/lstm-{STATES}-{ALPHABET_SIZE}-{RANDOM_SEED}-{MAX_TRAIN_LENGTH}/description.pt` for all above given models.

General-purpose code for training logistic regression models is in `models/extract.py`; for a usage example see `scripts/run_extractions.py`.

On `dgx2`, the relevant logistic regression models are in:
  - `/raid/lingo/abau/random-walks/lstm-{STATES}-{ALPHABET_SIZE}-{RANDOM_SEED}-{MAX_TRAIN_LENGTH}/extractor-model.pt` for all the above given models.

One can then test the extraction with `models/test_extractor.py` or `scripts/run_extractor_tests.py`. To make sure that the logistic regression is learning something legitimate, we create a fake graph where all emissions are always allowed, with the same number of states, and make sure that it's easier to learn the states of the true graph than the fake one.

General-purpose code for training the mixture of Gaussians is in `models/estimate_gaussians.py`; for usage example see `scripts/run_gaussian_estimates.py`.

On `dgx2`, the relevant Gaussian parameters are in:
  - `/raid/lingo/abau/random-walks/lstm-{STATES}-{ALPHABET_SIZE}-{RANDOM_SEED}-128/gaussians.json` for each dataset. We only have Gaussian parameters for the length-128 trained models.

## Visualizing individual predictions (TODO cleanup)

Using our trained extractor models, we can show what the model thinks its hidden state is for individual predictions. To generate this data, we use the code in `models/predict.py`; usage example in `scripts/predict.py`. This generates JSON data for use by a web interface for visualization, and includes predictions alongside predicted states.

For visualization purposes we use a small subsample of the test set. This is generaetd using `scripts/pick_small_sample.py`. The small samples used on `dgx2` are:
  - `/raid/lingo/abau/random-walks/testset-{STATES}-{ALPHABET_SIZE}-{RANDOM_SEED}/small_sample`

On `dgx2`, the generated predictions (missing Gaussians, because these were generated before I wrote that code) are in files:
  - `/raid/lingo/abau/random-walks/lstm-{STATES}-{ALPHABET_SIZE}-{RANDOM_SEED}-128/sample-annotated.json`

There's too much data to visualize it all at once, so we can visualize individual predictions interactively instead, using `scripts/server.py` and `scripts/visualize.html`. Run `server.py` and browse `http://localhost:8080`.

## Measuring ghost edges for out-of-domain emissions (TODO cleanup)

This last experiment, in which we test the models on out-of-domain data, has several steps. First, we need to create a model where we give every possible out-of-domain character in a small number of contexts. To do this, we take the small sample from before and add characters to the end of each sentence. Code for this is in `scripts/make_extended_sets.py`.

On `dgx2`, the extended datasets are in:
  - `/raid/lingo/abau/random-walks/testset-{STATES}-{ALPHABET_SIZE}-{RANDOM_SEED}/small_sample_extended`

We then run extractions and predictions using `models/predict.py` on this extended dataset to see what states the model thinks it goes to.

On `dgx2`, these predictions are written to:
  - `/raid/lingo/abau/random-walks/lstm-{STATES}-{ALPHABET_SIZE}-{RANDOM_SEED}-128/extended-annotated.json`. It's only done for length-128 trained models.

We collect this data together using `scripts/measure_ghost_edges.py` to see where the model thinks it goes for various out-of-domain emissions.

On `dgx2`, results are written to:
  - `/raid/lingo/abau/random-walks/lstm-{STATES}-{ALPHABET_SIZE}-{RANDOM_SEED}-128/ghost-edges.json`. It's only done for length-128 trained models.

We can test various hypotheses for what the ghost edges are likely to be; script for this is at `scripts/test_hypotheses.py`

## Experiments with what happens after a ghost transition

To see how much a ghost transition is like a real transition, we can create datasets that use a fake transition like a real one. Scripts for creating these datasets are in `datasets/create_transition_dataset.py` and `scripts/make_transition_sets.py`.

On `dgx2`, the generated data is in:
  - `/raid/lingo/abau/random-walks/lstm-{STATES}-{ALPHABET_SIZE}-{RANDOM_SEED}-128/transition-{STATE_1}-{EMISSION}-{STATE_2}.json`. It's only done for length-128 trained models.

We can then use a special test script to test the performance of these models only after the actual finished transition. Code for this is in `models/test_transitions.py` and `scripts/run_transition_tests.py`.

We can also measure how close they are by Mahalanobis distance to the closest Gaussian cluster of real states. Code for this is in `scripts/test_mahalanobis.py`.
