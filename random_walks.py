import random
import numpy as np
import scipy.linalg
import torch
from torch import nn
from torch.autograd import Variable

from graphviz import Digraph
import math
from helpers import encoding_string

class WalkGraph:
    """
    A randomly-constructed graph for generating datasets based on
    random walks through it. Edges are labelled with alphabet characters,
    and a string is generated by reading off edge labels during a random walk
    through the graph.

    Args:
        states (int): Number of vertices in the graph
        alphabet_size (int): Number of possible alphabet characters to emit
        transitions (states x alphabet_size matrix): Transition table, where
                transitions[s1][a] = s2 means an edge goes from s1 to s2 with label a.
                If there is no out-edge from s1 with label a, then transitions[s1][a] = -1.

    Attributes:
        states (int): Number of vertices in the graph (vertices will have ids in [0, states))
        alphabet_size (int): Number of possible alphabet characters to emit
        transitions (states x alphabet_size matrix): Transition table, where
                transitions[s1][a] = s2 means an edge goes from s1 to s2 with label a.
                If there is no out-edge from s1 with label a, then transitions[s1][a] = -1.
    """
    def __init__(self, states, alphabet_size, transitions):
        self.alphabet_size = alphabet_size
        self.states = states
        self.transitions = transitions

        # Convert graph to Markov transition matrix for entropy computation
        self.out_edges = [sum((0 if x == -1 else 1) for x in transitions[state])
            for state in range(states)]

        self.markov_transition_matrix = np.array([
            [
                sum(
                    ((1 / self.out_edges[from_state])
                     if transitions[from_state][symbol] == to_state else
                     0)
                    for symbol in range(alphabet_size)
                )
                for from_state in range(states)
            ]
            for to_state in range(states)
        ])

        self.emission_matrix = np.array([
            [
                (0
                 if transitions[state][symbol] == -1 else
                 (1 / self.out_edges[state]))
                for state in range(states)
            ]
            for symbol in range(alphabet_size)
        ])

    def visualize(self, filename):
        """
        Render a graphviz visualization of this graph as a png image.

        Args:
            filename (str): Filename to write the image to.
        """
        g = Digraph(format = 'png')
        for state in range(self.states):
            for i, out in enumerate(self.transitions[state]):
                if out != -1:
                    g.edge(str(state), str(out), label=encoding_string[i])
        g.render(filename)

    def serialize(self):
        """
        Get a JSON-serializable dictionary representing this graph.

        Returns:
            A dictionary that can be serialized to a string with json.dumps()
        """
        return {
            'states': int(self.states),
            'alphabet_size': int(self.alphabet_size),
            'transitions': [[int(x) for x in row] for row in self.transitions]
        }

    def reachable_states(self, state = 0):
        """
        Get the set of states reachable from a given state. Mostly for internal use,
        e.g. for testing if a graph is strongly-connected.

        Args:
            state (int): Starting state id to measure reachability from.

        Returns:
            The set of state ids that can be reached from the given state.
        """
        frontier = {state}
        total = {-1}
        while len(frontier) > 0:
            total |= frontier
            frontier = set().union(*(
                set(self.transitions[i])
                for i in frontier)) - total
        return total - {-1}

    def reconstruct_hidden_states(self, string):
        """
        Given a string that was generated by a walk through this graph,
        reconstruct what the walk actually was.

        Args:
            string (tuple of ints): tuple of alphabet symbols that were emitted

        Returns:
            A generator giving each state that was visited in the walk, *not*
            including the starting state 0.
        """
        state = 0
        for c in string:
            state = self.transitions[state][c]
            yield state

    def uses_transition(self, string, transition):
        """
        Determine whether a given string uses a given pair of state transitions
        (e.g. s1 -a-> s2 -b-> s3). For generating datasets with specific parts
        of a graph effectively removed.

        Args:
            string (tuple of ints): tuple of alphabet symbols that were emitted.
            transition (tuple of ints): tuple (s1, a, s2, b, s3) representing which states
                    and edges are being tested. This tuple means (s1 -a-> s2 -b-> s3) is being
                    tested.

        Returns:
            True if the transition pair was used, False otherwise.
        """
        state = 0

        history = [state]

        for c in string:
            history.append(c)
            state = self.transitions[state][c]
            history.append(state)

            if len(history) >= 5 and all(i == j for i, j in zip(history[-5:], transition)):
                return True

        return False

    def connected(self):
        """
        Is the entire graph reachable from the start state 0?

        Returns:
            True if the entire graph is reachable from the start state,
            False otherwise.
        """
        return len(self.reachable_states(0)) == self.states

    def strongly_connected(self):
        """
        Is the graph strongly connected?

        Returns:
            True if the graph is strongly connected, False otherwise.
        """
        return all((len(self.reachable_states(i)) == self.states) for i in range(self.states))

    def match(self, string):
        """
        Could a string have been generated by a walk through this graph?

        Args:
            string (tuple of ints): tuple of the alphabet symbols that were emitted

        Returns:
            True if this string could have been generated by a walk through this graph,
            False otherwise.
        """
        state = 0
        for symbol in string:
            if transitions[state][symbol] == -1:
                return False
            state = transitions[state][symbol]
        return True

    def generate_between_states(self, random_state, start_state = 0,
            end_state = 0, min_length = 2, max_length = 128):
        """
        Generate a string using a random walk that starts and ends at given fixed states,
        with length within given bounds. This is generated by performing a random walk starting
        at the starting state and continuing until we happen to reach the end state while our length
        is between the length bounds, retrying if we fail to reach the end state inside those bounds.
        Note: this will hang and crash if the given parameters are not possible
        (e.g. requiring a length of 0 but giving two different start/end states)!

        Args:
            random_state: seeded instance of numpy.random, for reproducibility
            start_state (int): state the walk must start at
            end_state (int): state the walk must end at
            min_length (int): minimum length of the walk
            max_length (int): maximum length of the walk

        Returns:
            tuple of ints containing the emitted alphabet symbols of the generated walk
        """
        state = start_state
        result = []

        for i in range(max_length):
            choices = [i for i in range(self.alphabet_size) if self.transitions[state][i] != -1]

            choice = random_state.choice(choices)
            result.append(choice)

            state = self.transitions[state][choice]

            if i > min_length and state == end_state:
                return tuple(result)

        return self.generate_between(random_state,
                start_state = start_state, end_state = end_state,
                min_length = min_length, max_length = max_length)

    def generate_toward(self, length, random_state, end_state = 0):
        """
        Generate a string of a given length that *ends* at a given fixed state.
        The start state will vary, and the strings are generated according to
        no guaranteed distribution. Kept in case it is useful; I don't recommend
        using this for anything, since the generating distribution is probably
        different from the forward-generating distribution. Instead, do rejection
        sampling with generate().

        Args:
            length (int): the length of walk to generate
            random_state: seeded instance of numpy.random, for reproducibility
            end_state (int): the state to end on

        Returns:
            tuple of ints containing emitted alphabet symbols of the generated walk
        """
        state = end_state

        result = []

        for i in range(length):
            # Search for something that could LEAD TO the given state
            in_edges = [
                (nstate, char)
                for nstate in range(self.states)
                for char in range(self.alphabet_size)
                if self.transitions[nstate][char] == state
            ]

            state, char = in_edges[random_state.randint(len(in_edges))]
            result.append(char)

        return tuple(reversed(result))

    def generate(self, length, random_state, forbidden = None, start_state = 0):
        """
        Generate a string of a given length via random walk. Optionally,
        forbid a given pair of transitions s1 -a-> s2 -b-> s3.

        Args:
            length (int): the length of walk to generate
            random_state: seeded instance of numpy.random, for reproducibility
            forbidden (tuple of ints, optional): tuple (s1, a, s2, b, s3); if provided, forbids
                the transition s1 -a-> s2 -b-> s3 from occurring in the random walk
            start_state (int, default 0): state to start at; 0 is the canonical start state

        Returns:
            Generator yielding each emitted alphabet symbol in the walk.
        """
        state = start_state
        state_history = [start_state]
        for i in range(length):
            choices = [i for i in range(self.alphabet_size) if self.transitions[state][i] != -1]

            if forbidden is not None and len(state_history) >= 3 and all(i == j for (i, j) in zip(state_history[-3:], forbidden[:3])):
                choices = [i for i in choices if i != forbidden[3]]

            choice = random_state.choice(choices)
            yield choice
            state = self.transitions[state][choice]

            state_history.append(choice)
            state_history.append(state)

    def generate_negative_of_length(self, length, random_state):
        """
        Generate an example of a string that cannot be generated by a random walk
        through this graph, through rejection sampling over a uniform distribution.

        Args:
            length (int): the length of string to generate
            random_state: seeded instance of numpy.random, for reproducibility

        Returns:
            Tuple of ints representing a sequence of alphabet symbols that could not be
            generated.
        """
        string = tuple(random_state.choice(self.alphabet_size)
                for _ in range(length))
        while self.match(string):
            string = tuple(random_state.choice(self.alphabet_size)
                    for _ in range(length))
        return string

    def get_entropy_at_length(self, length):
        """
        Compute the entropy of the distribution over strings described by
        generate(length).

        Args:
            length (int): length of string to measure the entropy of the distribution over

        Returns:
            Entropy of the distribution over strings of that length described by random
            walks through this graph.
        """
        state_probs = np.array([1 if x == 0 else 0 for x in range(self.states)])
        total_entropy = 0

        for _ in range(length):
            emit, state_probs = self.transition(state_probs)
            total_entropy += entropy(emit)

        return total_entropy

    def transition(self, state_probs):
        """
        Perform a Markov state transition given some existing state probabilities.

        Args:
            state_probs (numpy 1d vector): current state probabilities

        Returns:
            numpy 1d vector of new state probabilites after one transition.
        """
        return (np.dot(self.emission_matrix, state_probs),
                np.dot(self.markov_transition_matrix, state_probs))

    # Some largely useless functions, now that I think about it:
    def get_steady_state(self):
        """
        Get the Markov steady state for random walks through this graph.

        Returns:
            numpy 1d vector of the Markov steady state.
        """
        # Get steady state
        space = scipy.linalg.null_space(self.markov_transition_matrix - np.eye(self.states))

        steady_state = space[:, 0].flatten()
        return steady_state / sum(steady_state)

    def get_limit_entropy(self):
        """
        Get the entropy of one emission from a random walk through
        this graph in the limit as the walk gets infinitely long.

        Returns:
            The limit entropy of the graph.
        """
        return entropy(self.transition(self.get_steady_state())[0])

    def __repr__(self):
        return str((self.states, self.alphabet_size, self.transitions))

LARGE_NUMBER = 1e6 # "Certain" log probability to the perfect predictor
SMALL_NUMBER = -1e6 # "Certainly not" log probability to the perfect predictor

class PerfectPredictor(nn.Module):
    """
    A perfect language model for a given random walk dataset, built on the knowledge
    of the underlying random walk graph. Built to mimic a GRU for the purposes of
    testing.

    Args:
        graph (WalkGraph): The underlying random walk graph
        starting_state (int, optional, default 0): The state strings start at in the dataset

    Attributes:
        graph (WalkGraph): The underlying random walk graph
        starting_state (int, optional, default 0): The state strings start at in the dataset
    """
    def __init__(self, graph, starting_state = 0):
        super(PerfectPredictor, self).__init__()

        self.starting_state = starting_state

        self.graph = graph

        # We call ourselves a "GRU" because our hidden states
        # are not tuples.
        self.model = 'gru'
        self.n_characters = self.graph.alphabet_size

        self.predictions = torch.Tensor([
            [LARGE_NUMBER if cell >= 0 else SMALL_NUMBER for cell in row]
            for row in self.graph.transitions
        ])
        self.transitions = torch.Tensor(self.graph.transitions)

    def init_hidden(self, batch_size):
        return Variable(torch.full((batch_size,), self.starting_state, dtype=torch.long))

    def forward(self, input, hidden):
        batch_size = input.size(0)

        result = Variable(torch.Tensor(batch_size, self.n_characters))
        new_hidden = Variable(torch.zeros(batch_size, dtype=torch.long))

        # Input and hidden will both be 1d of longs
        for i, (c, s) in enumerate(zip(input, hidden)):
            new_hidden[i] = self.transitions[s][c]
            result[i] = self.predictions[new_hidden[i]]

        if input.is_cuda:
            result = result.cuda()
            new_hidden = new_hidden.cuda()

        return result, new_hidden

def entropy(probs):
    """
    Helper function; entropy of a discrete distribution.

    Args:
        probs (numpy 1d vector): probabilities of the discrete distribution

    Returns:
        Entropy of the given discrete distribution.
    """
    return -sum(x * np.log(x) / np.log(2) for x in probs)

def load_from_serialized(serialized):
    """
    The inverse of WalkGraph.serialize().

    Returns:
        WalkGraph, such that load_from_serialized(g.serialize()) is the same graph as g.
    """
    return WalkGraph(
        serialized['states'],
        serialized['alphabet_size'],
        serialized['transitions']
    )

def create_random_walk_graph(states, alphabet_size, random_state, drop_prob = 0.5):
    """
    Generate a random WalkGraph.

    Args:
        states (int): Number of vertices in the graph
        alphabet_size (int): Alphabet size for edge labels in the graph
        random_state: seeded instance of numpy.random, for reproducibility
        drop_prob (float, optional, default 0.5): probability that any given vertex
            will *lack* an out-edge with any given emission label. Higher drop_prob
            values mean sparser generated graphs.

    Returns:
        Generated WalkGraph.
    """
    choices = list(range(states))
    transitions = [
        [
            -1 if random_state.random() < drop_prob else random_state.choice(choices)
            for symbol in range(alphabet_size)
        ]
        for state in range(states)
    ]

    return WalkGraph(states, alphabet_size, transitions)

def create_random_connected_walk_graph(states, alphabet_size, random_state, drop_prob = 0.5):
    """
    Create a random *strongly connected* WalkGraph by rejection sampling.

    Args:
        states (int): Number of vertices in the graph
        alphabet_size (int): Alphabet size for edge labels in the graph
        random_state: seeded instance of numpy.random, for reproducibility
        drop_prob (float, optional, default 0.5): probability that any given vertex
            will *lack* an out-edge with any given emission label. Higher drop_prob
            values mean sparser generated graphs.

    Returns:
        Generated strongly connected WalkGraph.
    """
    x = create_random_walk_graph(states, alphabet_size, random_state, drop_prob)
    while not x.strongly_connected():
        x = create_random_walk_graph(states, alphabet_size, random_state)
        print('not strongly connected, retrying')
    return x
