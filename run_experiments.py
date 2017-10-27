import sys
import numpy as np
from mushroom import *
from atari_dqn import *

game = []

algorithms =['dqn', 'ddqn', 'wdqn', 'adqn']

for a in algorithms:
    if(a == 'wdqn' or a =='adqn'):
        sys.argv[1:] = ['--debug', '--algorithm', a, '--n-approximators' ,'10']
    else:
        sys.argv[1:] = ['--debug', '--algorithm', a]


    experiment()

