from copy import deepcopy

import numpy as np

from mushroom.algorithms.agent import Agent
from mushroom.approximators.regressor import Ensemble, Regressor
from mushroom.utils.replay_memory import Buffer, ReplayMemory


class DQN(Agent):
    def __init__(self, approximator, policy, mdp_info, params):
        alg_params = params['algorithm_params']
        self._batch_size = alg_params.get('batch_size')
        self._clip_reward = alg_params.get('clip_reward', True)
        self._n_approximators = alg_params.get('n_approximators', 1)
        self._train_frequency = alg_params.get('train_frequency')
        self._target_update_frequency = alg_params.get(
            'target_update_frequency')
        self._max_no_op_actions = alg_params.get('max_no_op_actions', 0)
        self._no_op_action_value = alg_params.get('no_op_action_value', 0)

        self._replay_memory = ReplayMemory(
            mdp_info,
            alg_params.get('initial_replay_size'),
            alg_params.get('max_replay_size'),
            alg_params.get('history_length', 1)
        )
        self._buffer = Buffer(size=alg_params.get('history_length', 1))

        self._n_updates = 0
        self._episode_steps = 0
        self._no_op_actions = None

        apprx_params_train = deepcopy(params['approximator_params'])
        apprx_params_train['name'] = 'train'
        apprx_params_target = deepcopy(params['approximator_params'])
        apprx_params_target['name'] = 'target'
        self.approximator = Regressor(approximator, **apprx_params_train)
        self.target_approximator = Regressor(approximator,
                                             **apprx_params_target)
        policy.set_q(self.approximator)

        self.target_approximator.model.set_weights(
            self.approximator.model.get_weights())

        super(DQN, self).__init__(policy, mdp_info, params)

    def fit(self, dataset):
        self._replay_memory.add(dataset)
        if self._replay_memory.initialized:
            state, action, reward, next_state, absorbing, _ =\
                self._replay_memory.get(self._batch_size)

            if self._clip_reward:
                reward = np.clip(reward, -1, 1)

            q_next = self._next_q(next_state, absorbing)
            q = reward.reshape(-1, 1) + self.mdp_info.gamma * q_next

            self.approximator.fit(state, action, q, **self.params['fit_params'])

            self._n_updates += 1

            if self._n_updates % self._target_update_frequency == 0:
                self._update_target()

    def _update_target(self):
        """
        Update the target network.

        """
        self.target_approximator.model.set_weights(
            self.approximator.model.get_weights())

    def _next_q(self, next_state, absorbing):
        """
        Args:
            next_state (np.ndarray): the states where next action has to be
                evaluated;
            absorbing (np.ndarray): the absorbing flag for the states in
                `next_state`.

        Returns:
            Maximum action-value for each state in `next_state`.

        """
        q = self.target_approximator.predict(next_state)
        for i in xrange(q.shape[0]):
            if absorbing[i]:
                q[i] *= 1. - absorbing[i]

        return np.max(q, axis=2)

    def draw_action(self, state):
        self._buffer.add(state)

        if self._episode_steps < self._no_op_actions:
            action = np.array([self._no_op_action_value])
        else:
            extended_state = self._buffer.get()

            action = super(DQN, self).draw_action(extended_state)

        self._episode_steps += 1

        return action

    def episode_start(self):
        self._no_op_actions = np.random.randint(
            self._buffer.size, self._max_no_op_actions + 1)
        self._episode_steps = 0

        self.policy.set_apprx(np.random.randint(self._n_approximators))


class DoubleDQN(DQN):
    """
    Double DQN algorithm.
    "Deep Reinforcement Learning with Double Q-Learning".
    Hasselt H. V. et al.. 2016.

    """
    def _next_q(self, next_state, absorbing):
        q = self.approximator.predict(next_state)
        for i in xrange(q.shape[0]):
            if absorbing[i]:
                q[i] *= 1. - absorbing[i]

        max_a = np.argmax(q, axis=2)

        return self.target_approximator.predict(next_state, max_a)


class WeightedDQN(DQN):
    """
    ...

    """
    def _next_q(self, next_state, absorbing):
        q = self.approximator.predict(next_state)
        if np.any(absorbing):
            q *= 1 - absorbing.reshape(-1, 1)

        q = np.reshape(q, [self._batch_size, self._n_approximators,
                       self.mdp_info.action_space.n])

        W = np.zeros((next_state.shape[0], self._n_approximators))

        pass

        return W
