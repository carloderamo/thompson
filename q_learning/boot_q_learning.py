from copy import deepcopy

import numpy as np

from mushroom.algorithms.value import TD
from mushroom.utils.table import EnsembleTable


class Bootstrapped(TD):
    def __init__(self, policy, mdp_info, learning_rate, n_approximators=10,
                 mu=0., sigma=1., p=2 / 3., weighted=False):
        self._n_approximators = n_approximators
        self._mu = mu
        self._sigma = sigma
        self._p = p
        self._weighted = weighted
        self._mask = np.random.binomial(1, self._p, self._n_approximators)
        self.Q = EnsembleTable(self._n_approximators, mdp_info.size)
        for i in range(len(self.Q.model)):
            self.Q.model[i].table = np.random.randn(
                *self.Q[i].shape) * self._sigma + self._mu

        super(Bootstrapped, self).__init__(self.Q, policy, mdp_info,
                                           learning_rate)

        self.alpha = [deepcopy(self.alpha)] * n_approximators

    def draw_action(self, state):
        self.policy.set_idx(np.random.randint(self._n_approximators))

        return super(Bootstrapped, self).draw_action(state)

    def _update(self, state, action, reward, next_state, absorbing):
        raise NotImplementedError


class BootstrappedQLearning(Bootstrapped):
    def _update(self, state, action, reward, next_state, absorbing):
        q_current = np.array([x[state, action] for x in self.Q.model])

        for i in np.argwhere(self._mask).ravel():
            if self._weighted:
                idx = np.random.randint(self._n_approximators)
            else:
                idx = i

            q_next = np.max(self.Q[idx][next_state]) if not absorbing else 0.
            self.Q.model[i][
                state, action] = q_current[i] + self.alpha[i](state, action) * (
                reward + self.mdp_info.gamma * q_next - q_current[i])

        self._mask = np.random.binomial(1, self._p, self._n_approximators)


class BootstrappedDoubleQLearning(Bootstrapped):
    def __init__(self, policy, mdp_info, learning_rate, n_approximators=10,
                 mu=0., sigma=1., p=1., weighted=False):
        super(BootstrappedDoubleQLearning, self).__init__(
            policy, mdp_info, learning_rate, n_approximators, mu, sigma, p,
            weighted
        )

        self.Qs = [EnsembleTable(n_approximators, mdp_info.size),
                   EnsembleTable(n_approximators, mdp_info.size)]

        for i in range(len(self.Qs[0])):
            self.Qs[0][i].table = np.random.randn(
                *self.Qs[0][i].shape) * self._sigma + self._mu

        for i in range(len(self.Qs[1])):
            self.Qs[1][i].table = self.Qs[0][i].table.copy()
            self.Q[i].table = self.Qs[0][i].table.copy()

        self.alpha = [deepcopy(self.alpha), deepcopy(self.alpha)]

    def _update(self, state, action, reward, next_state, absorbing):
        if np.random.uniform() < .5:
            i_q = 0
        else:
            i_q = 1

        q_current = np.array([x[state, action] for x in self.Qs[i_q]])
        if not absorbing:
            for i in np.argwhere(self._mask).ravel():
                if self._weighted:
                    idx = np.random.randint(self._n_approximators)
                else:
                    idx = i

                q_ss = self.Qs[i_q].predict(next_state, idx=idx)
                max_q = np.max(q_ss)
                a_n = np.array(
                    [np.random.choice(np.argwhere(q_ss == max_q).ravel())])
                q_next = self.Qs[1-i_q].predict(next_state, a_n, idx=idx)
                self.Qs[i_q][i][state, action] = q_current[i] + self.alpha[i_q][i](
                    state, action) * (
                        reward + self.mdp_info.gamma * q_next - q_current[i])
                self._update_Q(state, action, idx=i)
        else:
            for i in np.argwhere(self._mask).ravel():
                self.Qs[i_q][i][state, action] = q_current[i] + self.alpha[i_q][i](
                    state, action) * (reward - q_current[i])
                self._update_Q(state, action, idx=i)

        self._mask = np.random.binomial(1, self._p, self._n_approximators)

    def _update_Q(self, state, action, idx):
        self.Q[idx][state, action] = np.mean(
            [q[idx][state, action] for q in self.Qs])
