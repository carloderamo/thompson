import sys

import numpy as np
from joblib import Parallel, delayed

from mushroom.core import Core
from mushroom.environments.generators.taxi import generate_taxi
from mushroom.utils.callbacks import CollectDataset
from mushroom.utils.dataset import compute_J, parse_dataset
from mushroom.utils.parameters import Parameter

from boot_q_learning import BootstrappedQLearning
sys.path.append('..')
from policy import BootPolicy, WeightedPolicy


def experiment(n_approximators, policy):
    np.random.seed()

    # MDP
    mdp = generate_taxi('../grid.txt')

    # Policy
    pi = policy(n_approximators)

    # Agent
    learning_rate = Parameter(value=.15)
    algorithm_params = dict(learning_rate=learning_rate)
    agent = BootstrappedQLearning(pi, mdp.info, **algorithm_params)

    # Algorithm
    collect_dataset = CollectDataset()
    callbacks = [collect_dataset]
    core = Core(agent, mdp, callbacks)

    # Train
    n_steps = 3e5
    core.learn(n_steps=n_steps, n_steps_per_fit=1, quiet=True)

    dataset = collect_dataset.get()
    _, _, reward, _, _, _ = parse_dataset(dataset)

    return reward


if __name__ == '__main__':
    n_experiment = 4
    n_approximators = 10

    policy_name = {BootPolicy: 'Boot', WeightedPolicy: 'Weighted'}
    for p in [BootPolicy, WeightedPolicy]:
        out = Parallel(n_jobs=-1)(delayed(experiment)(
            n_approximators, p) for _ in range(n_experiment))

        np.save('r_%s.npy' % policy_name[p], out)
