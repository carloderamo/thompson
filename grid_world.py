from copy import deepcopy

import numpy as np
import pygame

from mushroom.environments import Environment, MDPInfo
from mushroom.utils import spaces


class AbstractGridWorld(Environment):
    def __init__(self, mdp_info, height, width, start, goal):
        assert not np.array_equal(start, goal)

        assert goal[0] < height and goal[1] < width,\
            'Goal position not suitable for the grid world dimension.'

        self._state = None
        self._height = height
        self._width = width
        self._start = start
        self._goal = goal

        super(AbstractGridWorld, self).__init__(mdp_info)

    def reset(self, state=None):
        if state is None:
            state = self.convert_to_int(self._start, self._width)

        self._state = state

        return self._state

    def step(self, action):
        state = self.convert_to_grid(self._state, self._width)

        new_state, reward, absorbing, info = self._step(state, action)
        self._state = self.convert_to_int(new_state, self._width)

        return self._state, reward, absorbing, info

    def _step(self, state, action):
        raise NotImplementedError('AbstractGridWorld is an abstract class.')

    @staticmethod
    def convert_to_grid(state, width):
        return np.array([state[0] / width, state[0] % width])

    @staticmethod
    def convert_to_int(state, width):
        return np.array([state[0] * width + state[1]])


class AbstractGridWorldPixel(AbstractGridWorld):
    def reset(self, state=None):
        if state is None:
            self._state = self.convert_to_pixel(self._initial_grid,
                                                self.window_size[1],
                                                self.window_size[0])
        else:
            self._state = deepcopy(state)

        return self._state

    def step(self, action):
        self._grid = self.convert_to_grid(self._state, *self._grid.shape)

        state = np.argwhere(self._grid == self._symbols['S']).ravel()

        new_state, reward, absorbing, info = self._step(state, action)

        if info['success']:
            self._grid[tuple(state)] = self._symbols['.']
            self._grid[tuple(new_state)] = self._symbols['S']

        self._state = self.convert_to_pixel(self._grid,
                                            self.window_size[1],
                                            self.window_size[0])

        return self._state, reward, absorbing, info

    def _step(self, state, action):
        raise NotImplementedError(
            'AbstractGridWorldPixel is an abstract class.')

    def convert_to_grid(self, state, height, width):
        h = state.shape[0] / height
        w = state.shape[1] / width

        return state[::h, ::w]

    @staticmethod
    def convert_to_pixel(state, window_height, window_width):
        h = window_height / state.shape[0]
        w = window_width / state.shape[1]

        return np.repeat(np.repeat(state, h, axis=0), w, axis=1)


class GridWorldPixelGenerator(AbstractGridWorldPixel):
    def __init__(self, grid_map_file, height_window=84, width_window=84):
        self.__name__ = 'GridWorldPixelGenerator'

        self.window_size = (width_window, height_window)

        self._symbols = {'.': 0, 'S': 100, '*': 127, '#': 25,
                         'G': 255}

        self._grid, start, goal = self._generate(grid_map_file)
        self._grid = self._grid.astype(np.uint8)
        self._initial_grid = deepcopy(self._grid)
        height = self._grid.shape[0]
        width = self._grid.shape[1]

        assert height_window % height == 0 and width_window % width == 0

        # MDP properties
        observation_space = spaces.Box(
            low=0., high=255., shape=(self.window_size[1], self.window_size[0]))
        action_space = spaces.Discrete(4)
        horizon = 100
        gamma = .95
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        super(GridWorldPixelGenerator, self).__init__(mdp_info, height, width,
                                                      start, goal)

    def _step(self, state, action):
        new_state = np.array(state)
        if np.array_equal(new_state, self._goal):
            reward = 10
            absorbing = True
            success = True
        else:
            if action == 0:
                new_state[0] -= 1
            elif action == 1:
                new_state[0] += 1
            elif action == 2:
                new_state[1] -= 1
            elif action == 3:
                new_state[1] += 1

            c = self._grid[new_state[0]][new_state[1]]
            if c == self._symbols['*']:
                reward = 0 #-10
                absorbing = True
                success = True
            elif c in [self._symbols['.'], self._symbols['S']]:
                reward = 0 #np.random.choice([-12, 10])
                absorbing = False
                success = True
            elif c == self._symbols['G']:
                reward = 0 #np.random.choice([-12, 10])
                absorbing = False
                success = True
            elif c == self._symbols['#']:
                reward = 0 #np.random.choice([-12, 10])
                absorbing = False
                new_state = np.array(state)
                success = False

        return new_state, reward, absorbing, {'success': success}

    def _generate(self, grid_map):
        grid = list()
        with open(grid_map, 'r') as f:
            m = f.read()

            assert 'S' in m and 'G' in m

            row = list()
            row_idx = 0
            col_idx = 0
            for c in m:
                if c in ['#', '.', 'S', 'G', '*']:
                    row.append(self._symbols[c])
                    if c == 'S':
                        start = (row_idx, col_idx)
                    elif c == 'G':
                        goal = (row_idx, col_idx)
                    col_idx += 1
                elif c == '\n':
                    grid.append(row)
                    row = list()
                    row_idx += 1
                    col_idx = 0
                else:
                    raise ValueError('Unknown marker.')

        grid = np.array(grid)

        assert not np.array_equal(start, goal)

        return grid, start, goal

    def render(self, mode='human', close=False):
        if not hasattr(self, 'display'):
            pygame.init()
            self.display = pygame.display.set_mode(self.window_size)

        surf = pygame.surfarray.make_surface(self._state.T)

        self.display.blit(surf, (0, 0))
        pygame.display.update()

    def set_episode_end(self, ends_at_life):
        pass

