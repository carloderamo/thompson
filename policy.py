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
            if self._evaluation:
                if isinstance(self._approximator.model, list):
                    q_list = list()
                    for q in self._approximator.model:
                        q_list.append(q.predict(state))
                else:
                    q_list = self._approximator.predict(state).squeeze()

                max_as, count = np.unique(np.argmax(q_list, axis=1),
                                          return_counts=True)
                max_a = np.array([max_as[np.random.choice(
                    np.argwhere(count == np.max(count)).ravel())]])

                return max_a
            else:
                q = self._approximator.predict(state, idx=self._idx)

                max_a = np.argwhere(q == np.max(q)).ravel()
                if len(max_a) > 1:
                    max_a = np.array([np.random.choice(max_a)])

                return max_a
        else:
            return np.array([np.random.choice(self._approximator.n_actions)])

    def set_epsilon(self, epsilon):
        self._epsilon = epsilon

    def set_eval(self, eval):
        self._evaluation = eval

    def set_idx(self, idx):
        self._idx = idx


class WeightedPolicy(TDPolicy):
    def __init__(self, n_approximators, epsilon=None):
        if epsilon is None:
            epsilon = Parameter(0.)

        super(WeightedPolicy, self).__init__()

        self._n_approximators = n_approximators
        self._epsilon = epsilon
        self._evaluation = False

    def draw_action(self, state):
        if not np.random.uniform() < self._epsilon(state):
            if self._evaluation:
                if isinstance(self._approximator.model, list):
                    q_list = list()
                    for q in self._approximator.model:
                        q_list.append(q.predict(state))
                else:
                    q_list = self._approximator.predict(state).squeeze()

                max_as, count = np.unique(np.argmax(q_list, axis=1),
                                          return_counts=True)
                max_a = np.array([max_as[np.random.choice(
                    np.argwhere(count == np.max(count)).ravel())]])

                return max_a
            else:
                if isinstance(self._approximator.model, list):
                    q_list = list()
                    for i in range(self._n_approximators):
                        q_list.append(self._approximator.predict(state, idx=i))
                else:
                    q_list = self._approximator.predict(state).squeeze()

                mean_q = np.mean(q_list, 0)
                sigma_q = np.std(q_list, 0, ddof=1)

                samples = np.random.normal(mean_q, sigma_q)

                max_a = np.array([np.argmax(samples)])

                return max_a
        else:
            return np.array([np.random.choice(
                self._approximator.n_actions)])

    def set_epsilon(self, epsilon):
        self._epsilon = epsilon

    def set_eval(self, eval):
        self._evaluation = eval

    def set_idx(self, idx):
        pass
