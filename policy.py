import numpy as np

from mushroom.policy.td_policy import TDPolicy


class BootPolicy(TDPolicy):
    def __init__(self, n_approximators):
        super(BootPolicy, self).__init__()

        self._n_approximators = n_approximators

    def draw_action(self, state):
        n_actions = self._approximator.n_actions

        start = self._idx * n_actions
        stop = self._idx * n_actions + n_actions

        q = self._approximator.predict(state)[start:stop]
        max_a = np.argwhere(q == np.max(q)).ravel()

        if len(max_a) > 1:
            max_a = np.array([np.random.choice(max_a)])

        return max_a

    def set_apprx(self, idx):
        self._idx = idx
