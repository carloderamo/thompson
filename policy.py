import numpy as np

from mushroom.policy.td_policy import TDPolicy
from mushroom.utils.parameters import Parameter


class BootPolicy(TDPolicy):
    def __init__(self, n_approximators, epsilon=None):
        if epsilon is None:
            epsilon = Parameter(0.)

        super(BootPolicy, self).__init__()

        self._n_approximators = n_approximators
        self._epsilon = epsilon
        self._evaluation = False
        self._idx = None

    def draw_action(self, state):
        if not np.random.uniform() < self._epsilon(state):
            if not self._evaluation:
                q = self._approximator.predict(state, idx=self._idx)

                max_a = np.argwhere(q == np.max(q)).ravel()
                if len(max_a) > 1:
                    max_a = np.array([np.random.choice(max_a)])
            else:
                q = np.array(self._approximator.predict(state)).squeeze()

                max_as, count = np.unique(np.argmax(q, axis=1),
                                          return_counts=True)
                max_a = np.array([max_as[np.random.choice(
                    np.argwhere(count == np.max(count)).ravel())]])

            return max_a

        return np.array([np.random.choice(self._approximator.n_actions)])

    def set_epsilon(self, epsilon):
        self._epsilon = epsilon

    def set_eval(self, eval):
        self._evaluation = eval

    def set_idx(self, idx):
        self._idx = idx


class WeightedPolicy(TDPolicy):
    def __init__(self, n_approximators):
        super(WeightedPolicy, self).__init__()

        self._n_approximators = n_approximators
        self._evaluation = False
        self._precision = 1000

    def draw_action(self, state):
        all_q = list()
        if not self._evaluation:
            for i in range(self._n_approximators):
                all_q.append(self._approximator.predict(state, idx=i))

            mean_q = np.mean(all_q, 0)
            sigma_q = np.std(all_q, 0)

            samples = np.random.normal(np.repeat([mean_q], self._precision, 0),
                                       np.repeat([sigma_q], self._precision, 0))
            max_idx = np.argmax(samples, axis=1)
            max_idx, max_count = np.unique(max_idx, return_counts=True)
            count = np.zeros(mean_q.size)
            count[max_idx] = max_count

            w = count / self._precision

            return np.array([np.random.choice(np.arange(len(all_q[0])), p=w)])
        else:
            q = np.array(self._approximator.predict(state)).squeeze()

            max_as, count = np.unique(np.argmax(q, axis=1),
                                      return_counts=True)
            return np.array([max_as[np.random.choice(
                np.argwhere(count == np.max(count)).ravel())]])

    def set_eval(self, eval):
        self._evaluation = eval

    def set_idx(self, idx):
        pass
