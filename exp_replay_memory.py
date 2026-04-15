import numpy as np
import random


class ReplayMemory(object):
    """
    Experience Replay Buffer for Deep Q-Networks (DQN)

    Stores transitions of the form:
        (s_t, a_t, r_{t+1}, s_{t+1}, done)

    Mathematical role:
        We sample i.i.d. transitions from a dataset D instead of using sequential data.

    This approximates:
        (s, a, r, s') ~ D

    Why this matters:
        - Breaks temporal correlations
        - Stabilizes stochastic gradient descent
        - Improves data efficiency (re-use past experiences)
    """

    def __init__(self, capacity):
        # Maximum number of transitions stored
        self.max_capacity = capacity

        # Internal storage (circular buffer)
        self.transitions = []

        # Pointer for next insertion (implements FIFO overwrite)
        self.next_transition_index = 0

    def length(self):
        """Returns current number of stored transitions."""
        return len(self.transitions)

    def store(self, state, action, reward, next_state, done):
        """
        Stores a transition in the replay buffer.

        Transition:
            (s_t, a_t, r_{t+1}, s_{t+1}, done)

        If buffer is full:
            Overwrites oldest transition (circular buffer)
        """

        transition = (state, action, reward, next_state, done)

        if self.next_transition_index >= self.length():
            # Buffer not full → append
            self.transitions.append(transition)
        else:
            # Buffer full → overwrite oldest transition
            self.transitions[self.next_transition_index] = transition

        # Move pointer cyclically
        self.next_transition_index = (self.next_transition_index + 1) % self.max_capacity

    def sample_minibatch(self, batch_size):
        """
        Samples a random minibatch of transitions.

        This approximates drawing i.i.d. samples:
            (s, a, r, s') ~ D

        Returns:
            states, actions, rewards, next_states, dones
        """

        # Uniform random sampling (without replacement preferred)
        batch = random.sample(self.transitions, batch_size)

        # Unzip batch into components
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )