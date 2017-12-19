import numpy as np

from mushroom.utils.preprocessor import Preprocessor


class OneHot(Preprocessor):
    """
    This class implements the function to scale the input data by a given
    coefficient.

    """
    def __init__(self, n):
        self._n = n

    def _compute(self, x):
        oh = np.zeros((len(x), self._n))
        oh[np.arange(len(x)), x.ravel().astype(np.int)] = 1.

        return x
