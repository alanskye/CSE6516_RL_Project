import gym
import numpy as np
import pickle
from tqdm import tqdm
from pathlib import Path
from enum import IntEnum

ALPHA = 0.2
GAMMA = 0.99
TRAIN_COUNT = 10000
TRAIN_STEP_DEBUG = TRAIN_COUNT // 10
TEST_COUNT = 1000
EPS = 0.1
MOVE_LIMIT = 300

class dir(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

movement = [(-1, 0), (1, 0), (0, -1), (0, 1)]


class MazeEnv(gym.Env):
    '''
        state: (up, down, left, right), 4^5
    '''
    def __init__(self, maze_file: Path):
        super(MazeEnv, self).__init__()
        with open(maze_file) as mf:
            self.lines = mf.readlines()
            for line in self.lines:
                print(line.rstrip('\n'))
        
        self.action_space = gym.spaces.Discrete(4)

        self.N = len(self.lines) // 2
        self.position = (0, 0)
        self.visit_count = np.zeros(shape=(self.N, self.N), dtype=int)
        self.visit_count[0][0] = 1
        self.passable = np.zeros(shape=(self.N, self.N, 4), dtype=bool)
                
        for (x, y) in np.ndindex((self.N, self.N)):
            mf_x = x * 2 + 1
            mf_y = y * 2 + 1

            if self.lines[mf_x - 1][mf_y] == ' ':
                self.passable[x][y][dir.UP] = True

            if self.lines[mf_x + 1][mf_y] == ' ':
                self.passable[x][y][dir.DOWN] = True

            if self.lines[mf_x][mf_y - 1] == ' ':
                self.passable[x][y][dir.LEFT] = True

            if self.lines[mf_x][mf_y + 1] == ' ':
                self.passable[x][y][dir.RIGHT] = True
        

    def _get_state(self) -> tuple:
        state_vector = [0] * 4 # up down left right
        for i in range(4):
            next_position = (self.position[0] + movement[i][0], self.position[1] + movement[i][1])
            if 0 <= next_position[0] < self.N and 0 <= next_position[1] < self.N:                
                if not self.passable[next_position][i]: # wall
                    state_vector[i] = 3
                else:
                    state_vector[i] = min(self.visit_count[next_position], 2)
            else:
                state_vector[i] = 3 # wall

        return tuple(state_vector)


    def step(self, action: dir):
        '''
            returns: next_state, reward, done
        '''
        done = False
        next_position = (self.position[0] + movement[action][0], self.position[1] + movement[action][1])

        if (0 <= next_position[0] < self.N) and (0 <= next_position[1] < self.N) and self.passable[self.position][action] == True:
            self.position = next_position
            self.visit_count[self.position] += 1
            if self.position == (self.N - 1, self.N - 1):
                done = True
                reward = 10
            elif self.visit_count[self.position[0]][self.position[1]] >= 2:
                reward = -1
            else:
                reward = 0

        else: # bump!
            reward = -1
            
        state = self._get_state()
        return state, reward, done


    def reset(self) -> tuple:
        self.position = (0, 0)
        self.visit_count = np.zeros(shape=(self.N, self.N), dtype=int)
        self.visit_count[0][0] = 1
        state = self._get_state()
        return state



class Agent:
    def __init__(self, env):
        self.env = env
        self.episode_reward = 0.0
        self.q_val = np.zeros(shape=(4, 4, 4, 4, 4), dtype=float)
        self.average_reward_list = []

    def train(self):
        state = self.env.reset()
        for t in range(MOVE_LIMIT):
            if np.random.rand() < EPS:
                action = self.env.action_space.sample() # exploration
            else:
                action = np.argmax(self.q_val[state]) # exploitation
            next_state, reward, done = self.env.step(action)
            q_next_max = np.max(self.q_val[next_state])
            self.q_val[state][action] = (1 - ALPHA) * self.q_val[state][action] \
                + ALPHA * (reward + GAMMA * q_next_max)
            
            if done:
                return reward
            else:
                state = next_state

        return 0.0


    def test(self):
        state = self.env.reset()
        for t in range(MOVE_LIMIT):
            action = np.argmax(self.q_val[state])
            next_state, reward, done = self.env.step(action)
            if done:
                return reward
            else:
                state = next_state

        return 0.0




env = MazeEnv('6x6_1.maz')
agent = Agent(env)
reward_total = 0.0

for train_step in tqdm(range(TRAIN_COUNT)):
    reward_total += agent.train()
    if (train_step + 1) % TRAIN_STEP_DEBUG == 0:
        average_reward = reward_total / (train_step + 1)
        print(f'{average_reward=:.2f}')

for test_step in tqdm(range(TEST_COUNT)):
    reward_total += agent.test()
    
