import sys

import numpy as np
from joblib import Parallel, delayed

from mushroom.core import Core
from mushroom.environments.generators.taxi import generate_taxi
from mushroom.utils.callbacks import CollectDataset
from mushroom.utils.dataset import parse_dataset
from mushroom.utils.parameters import ExponentialDecayParameter, Parameter

from boot_q_learning import BootstrappedDoubleQLearning
sys.path.append('..')
from policy import BootPolicy, WeightedPolicy


def experiment(n_approximators, policy):
    np.random.seed()

    # MDP
    mdp = generate_taxi('../grid.txt')

    # Policy
    # epsilon = ExponentialDecayParameter(value=1., decay_exp=.5,
    #                                     size=mdp.info.observation_space.size)
    epsilon = Parameter(0.)
    pi = policy(n_approximators, epsilon=epsilon)

    # Agent
    # learning_rate = ExponentialDecayParameter(value=1., decay_exp=.3,
    #                                           size=mdp.info.size)
    learning_rate = Parameter(.15)
    algorithm_params = dict(learning_rate=learning_rate, sigma=2.)
    agent = BootstrappedDoubleQLearning(pi, mdp.info, **algorithm_params)

    # Algorithm
    collect_dataset = CollectDataset()
    callbacks = [collect_dataset]
    core = Core(agent, mdp, callbacks)

    # Train
    n_steps = 6e5
    core.learn(n_steps=n_steps, n_steps_per_fit=1, quiet=True)

    dataset = collect_dataset.get()
    _, _, reward, _, _, _ = parse_dataset(dataset)
    pi.set_eval(True)
    dataset = core.evaluate(n_steps=1000, quiet=True)
    reward_test = [r[2] for r in dataset]

    return reward, reward_test


if __name__ == '__main__':
    n_experiment = 10
    n_approximators = 10

    policy_name = {BootPolicy: 'Boot', WeightedPolicy: 'Weighted'}
    for p in [BootPolicy, WeightedPolicy]:
        out = Parallel(n_jobs=-1)(delayed(experiment)(
            n_approximators, p) for _ in range(n_experiment))

        r = [x[0] for x in out]
        r_test = [x[1] for x in out]
        np.save('r_%s.npy' % policy_name[p], r)
        np.save('r_test_%s.npy' % policy_name[p], r_test)
