import numpy as np

from mushroom.policy.td_policy import TDPolicy


class BootPolicy(TDPolicy):
    def __init__(self, n_approximators):
        super(BootPolicy, self).__init__()

        self._n_approximators = n_approximators
        self._evaluation = False

    def draw_action(self, state):
        if not self._evaluation:
            idx = np.random.randint(self._n_approximators)
            q = self._approximator.predict(state, idx=idx)

            max_a = np.argwhere(q == np.max(q)).ravel()
            if len(max_a) > 1:
                max_a = np.array([np.random.choice(max_a)])
        else:
            q = np.array(self._approximator.predict(state)).squeeze()

            max_as, count = np.unique(np.argmax(q, axis=1), return_counts=True)
            max_a = np.array([max_as[np.random.choice(
                np.argwhere(count == np.max(count)).ravel())]])

        return max_a

    def set_eval(self, eval):
        self._evaluation = eval
