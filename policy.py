import numpy as np

from mushroom.policy.td_policy import TDPolicy


class BootPolicy(TDPolicy):
    def __init__(self):
        super(BootPolicy, self).__init__()

        self._evaluation = False

    def draw_action(self, state):
        q = self._approximator.predict(state)
        if not self._evaluation:
            q = q[self._idx]

            max_a = np.argwhere(q == np.max(q)).ravel()
            if len(max_a) > 1:
                max_a = np.array([np.random.choice(max_a)])
        else:
            max_as, count = np.unique(np.argmax(q, axis=1), return_counts=True)
            max_a = np.array([max_as[np.random.choice(
                np.argwhere(count == np.max(count)).ravel())]])

        return max_a

    def set_apprx(self, idx):
        self._idx = idx

    def set_eval(self, eval):
        self._evaluation = eval
