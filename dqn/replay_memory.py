import numpy as np


class ReplayMemory(object):
    def __init__(self, initial_size, max_size):
        self._initial_size = initial_size
        self._max_size = max_size

        self.reset()

    def add(self, dataset, mask):
        for i in range(len(dataset)):
            self._states[self._idx] = dataset[i][0]
            self._actions[self._idx] = dataset[i][1]
            self._rewards[self._idx] = dataset[i][2]
            self._next_states[self._idx] = dataset[i][3]
            self._absorbing[self._idx] = dataset[i][4]
            self._last[self._idx] = dataset[i][5]
            self._mask[self._idx] = mask[i]

            self._idx += 1
            if self._idx == self._max_size:
                self._full = True
                self._idx = 0

    def get(self, n_samples):
        if self._current_sample_idx + n_samples >= len(self._sample_idxs):
            self._sample_idxs = np.random.choice(self.size, self.size,
                                                 replace=False)
            self._current_sample_idx = 0

        start = self._current_sample_idx
        stop = start + n_samples

        self._current_sample_idx = stop

        return np.stack([np.array(self._states[i]) for i in self._sample_idxs[start:stop]]),\
            np.array([self._actions[i] for i in self._sample_idxs[start:stop]]),\
            np.array([self._rewards[i] for i in self._sample_idxs[start:stop]]),\
            np.stack([np.array(self._next_states[i]) for i in self._sample_idxs[start:stop]]),\
            np.array([self._absorbing[i] for i in self._sample_idxs[start:stop]]),\
            np.array([self._last[i] for i in self._sample_idxs[start:stop]]),\
            np.array([self._mask[i] for i in self._sample_idxs[start:stop]])

    def reset(self):
        self._idx = 0
        self._full = False
        self._states = [None for _ in range(self._max_size)]
        self._actions = [None for _ in range(self._max_size)]
        self._rewards = [None for _ in range(self._max_size)]
        self._next_states = [None for _ in range(self._max_size)]
        self._absorbing = [None for _ in range(self._max_size)]
        self._last = [None for _ in range(self._max_size)]
        self._mask = [None for _ in range(self._max_size)]

        self._sample_idxs = np.random.choice(self._initial_size,
                                             self._initial_size,
                                             replace=False)
        self._current_sample_idx = 0

    @property
    def initialized(self):
        """
        Returns:
            Whether the replay memory has reached the number of elements that
            allows it to be used.

        """
        return self.size >= self._initial_size

    @property
    def size(self):
        """
        Returns:
            The number of elements contained in the replay memory.

        """
        return self._idx if not self._full else self._max_size
