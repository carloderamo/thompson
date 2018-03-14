import numpy as np

from mushroom.algorithms.value import TD
from mushroom.utils.table import EnsembleTable


class Bootstrapped(TD):
    def __init__(self, policy, mdp_info, learning_rate, n_approximators=10,
                 mu=0., sigma=1., p=2 / 3.):
        self._n_approximators = n_approximators
        self._mu = mu
        self._sigma = sigma
        self._p = p
        self._mask = np.random.binomial(1, self._p, self._n_approximators)
        self.Q = EnsembleTable(self._n_approximators, mdp_info.size)
        for i in xrange(len(self.Q.model)):
            self.Q.model[i].table = np.random.randn(
                *self.Q[i].shape) * self._sigma + self._mu

        super(Bootstrapped, self).__init__(self.Q, policy, mdp_info,
                                           learning_rate)

    def draw_action(self, state):
        self.policy.set_idx(np.random.randint(self._n_approximators))

        return super(Bootstrapped, self).draw_action(state)

    def _update(self, state, action, reward, next_state, absorbing):
        raise NotImplementedError


class BootstrappedQLearning(Bootstrapped):
    def _update(self, state, action, reward, next_state, absorbing):
        q_current = np.array([x[state, action] for x in self.Q.model])

        for i in np.argwhere(self._mask).ravel():
            q_next = np.max(self.Q[i][next_state]) if not absorbing else 0.
            self.Q.model[i][
                state, action] = q_current[i] + self.alpha(state, action) * (
                reward + self.mdp_info.gamma * q_next - q_current[i])

        self._mask = np.random.binomial(1, self._p, self._n_approximators)
