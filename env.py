import numpy as np
import gym
from gym import spaces
import random
from enum import Enum
import torch
#from utils import Actions4
#from utils import Actions3





class Actions4(Enum):
    Relax = 1
    EmotionalStress = 2
    PhysicalStress = 3
    CognitiveStress = 4

class Actions3(Enum):
    Relax = 1
    EmotionalStress = 2
    PhysicalStress = 3





class PhysioEnv4Classes(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, data, window_size, episode_max_len=300,rew_threshold=0.237, is_train=True, sogg='Sogetto0', classe='Relax0'):
        
        self._is_train = is_train  # a flag to specify if the env is for training or test
        self.data = data 
        

        # episode
        self._episode_max_len = episode_max_len
        self._time = None  # number of ticks since the beginning of the episode. an episode starts with a random tick in the in a random class of a random subject
        self._end_tick = None  # the last tick of a class
        self._done = None  # a flag, true if the episode ends
        self._current_tick = None  # is the class current tick
        self.random_sogg = None
        self.random_class = None
        self._total_reward = None
        self._step_reward_history = None
        self.episode_history = None
        self.global_history_action = {}  # TRAINING HISTORY. the hystory of all actions, rewards and number of tick of all trainig apisode.
        self.num_episode = 0
        self.reward_threshold = rew_threshold
        self._last_action = []
        self._class_weigth = 0
        self._sogg = sogg
        self._class = classe
        
        # spaces
        self._starting_window = window_size
        self._window = window_size
        self.shape = (self._episode_max_len + self._starting_window, 7)
        self.action_space = spaces.Box(low=0, high=1, shape=(len(Actions4),), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape,
                                            dtype=np.float32)  
        

    def reset(self):
        self._done = False

        self._window = self._starting_window

        self.random_sogg, self.random_class = self._get_random_SoggClass()  # selct a random sogg and random class from the dataset
        if self._is_train:
            self._end_tick = len(self.data[self.random_sogg][self.random_class])  # the last tick of a class

        else:
            self._end_tick = len(self.data)

        tick_list = [i for i in range(self._window,
                                      self._end_tick - self._window)]  # create a list of the tick of the episode starting from a start tick

        if self._is_train:
            self._current_tick = random.choice(tick_list)  # select random starting tick from the list of tick
        else:
            self._current_tick = tick_list[0]

        self._time = 0
        self._step_rew = 0.
        self._total_reward = 0.

        self.global_history_action[
            'episode{}'.format(self.num_episode)] = self.episode_history  # update the training hystory
        
        self.num_episode += 1
        self.episode_history = {}
        
        return self._get_observation()  # return the observations of shape (_window, num_of_features) from the current_tick

    def step(self, action):
        self._done = False
        self._current_tick += 1
        self._time += 1
        if self._current_tick == self._end_tick:
            self._done = True

        step_reward = self._calculate_reward(action)
        self._last_action.append(action)

        self._step_rew = step_reward
        self._total_reward += step_reward

        observation = self._get_observation()
        self._window += 1
        # self.observation_space = spaces.Box(low=-np.inf, high=np.inf,shape=(self._window,7), dtype=np.float64)
        info = dict(
            total_reward=self._total_reward,
            tick=self._time,
            step_action=action

        )

        self._update_history(info)

        return observation, step_reward, self._done, info

    def _get_random_SoggClass(self):

        if self._is_train:
            random_sogg = random.choice(list(self.data.keys()))
            random_class = random.choice(list(self.data[random_sogg].keys()))
        else:
            random_sogg = self._sogg
            random_class = self._class

        return random_sogg, random_class

    def _get_observation(self):

        obs = np.zeros((self._episode_max_len + self._starting_window, 7))
        if self._is_train:
            features = self.data[self.random_sogg][self.random_class]
            obs[:self._window] = features[self._current_tick - self._window:self._current_tick]
        else:
            features = self.data
            obs[:self._window] = np.array(features[self._current_tick - self._window:self._current_tick])

        return obs

    def _update_history(self, info):

        if not self.episode_history:
            self.episode_history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.episode_history[key].append(value)

    def _calculate_reward(self, action):
        
        sm = torch.nn.Softmax()
        a = torch.FloatTensor(action)
        actions_prob = sm(a)
        actions_prob_sort=np.array(actions_prob)
        actions_prob_sort.sort()
        pred = actions_prob_sort[-1]
        pred1 = actions_prob_sort[-2]
        diff=pred-pred1


        penality_coeff = 1
        treshold = self.reward_threshold

        if (pred == actions_prob[0]) and self.random_class == 'Relax':
            if diff > treshold:
                step_reward = actions_prob[0]
                
                self._done = True
            else:
                step_reward = (-0.0003 * (self._time ** (1 / 3)))

        elif (pred == actions_prob[0]) and self.random_class != 'Relax':

            if diff > treshold:
                step_reward = -actions_prob[0] * penality_coeff
   
                self._done = True
            else:
                step_reward = (-0.0003 * (self._time ** (1 / 3)))

        if (pred == actions_prob[1]) and self.random_class == 'EmotionalStress':
            if diff > treshold:
                step_reward = actions_prob[1]

                self._done = True
            else:
                step_reward = (-0.0003 * (self._time ** (1 / 3)))

        elif (pred == actions_prob[1]) and self.random_class != 'EmotionalStress':

            if diff > treshold:
                step_reward = -actions_prob[1] * penality_coeff

                self._done = True
            else:
                step_reward = (-0.0003 * (self._time ** (1 / 3)))

        if (pred == actions_prob[2]) and self.random_class == 'PhysicalStress':
            if diff > treshold:
                step_reward = actions_prob[2]

                self._done = True
            else:
                step_reward = (-0.0003 * (self._time ** (1 / 3)))

        elif (pred == actions_prob[2]) and self.random_class != 'PhysicalStress':

            if diff > treshold:
                step_reward = -actions_prob[2] * penality_coeff

                self._done = True
            else:
                step_reward = (-0.0003 * (self._time ** (1 / 3)))
        
        if (pred == actions_prob[3]) and self.random_class == 'CognitiveStress':
            if diff > treshold:
                step_reward = actions_prob[3]

                self._done = True
            else:
                step_reward = (-0.0003 * (self._time ** (1 / 3)))

        elif (pred == actions_prob[3]) and self.random_class != 'CognitiveStress':

            if diff > treshold:
                step_reward = -actions_prob[3] * penality_coeff

                self._done = True
            else:
                step_reward = (-0.0003 * (self._time ** (1 / 3)))




        step_reward = np.array(step_reward)


        if self._time >= self._episode_max_len:
            self._done = True

        return step_reward







class PhysioEnv3Classes(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, data, window_size,rew_threshold, episode_max_len=300, is_train=True, sogg='Sogetto0', classe='Relax0'):
        
        self._is_train = is_train  # a flag to specify if the env is for training or test
        self.data = data 
        

        # episode
        self._episode_max_len = episode_max_len
        self._time = None  # number of ticks since the beginning of the episode. an episode starts with a random tick in the in a random class of a random subject
        self._end_tick = None  # the last tick of a class
        self._done = None  # a flag, true if the episode ends
        self._current_tick = None  # is the class current tick
        self.random_sogg = None
        self.random_class = None
        self._total_reward = None
        self._step_reward_history = None
        self.episode_history = None
        self.global_history_action = {}  # TRAINING HISTORY. the hystory of all actions, rewards and number of tick of all trainig apisode.
        self.num_episode = 0
        self.reward_threshold = rew_threshold
        self._last_action = []
        self._class_weigth = 0
        self._sogg = sogg
        self._class = classe
        
        # spaces
        self._starting_window = window_size
        self._window = window_size
        self.shape = (self._episode_max_len + self._starting_window, 7)
        self.action_space = spaces.Box(low=0, high=1, shape=(len(Actions3),), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape,
                                            dtype=np.float32)  
        

    def reset(self):
        self._done = False

        self._window = self._starting_window

        self.random_sogg, self.random_class = self._get_random_SoggClass()  # selct a random sogg and random class from the dataset
        if self._is_train:
            self._end_tick = len(self.data[self.random_sogg][self.random_class])  # the last tick of a class

        else:
            self._end_tick = len(self.data)

        tick_list = [i for i in range(self._window,
                                      self._end_tick - self._window)]  # create a list of the tick of the episode starting from a start tick

        if self._is_train:
            self._current_tick = random.choice(tick_list)  # select random starting tick from the list of tick
        else:
            self._current_tick = tick_list[0]

        self._time = 0
        self._step_rew = 0.
        self._total_reward = 0.

        self.global_history_action[
            'episode{}'.format(self.num_episode)] = self.episode_history  # update the training hystory
        
        self.num_episode += 1
        self.episode_history = {}
        
        return self._get_observation()  # return the observations of shape (_window, num_of_features) from the current_tick

    def step(self, action):
        self._done = False
        self._current_tick += 1
        self._time += 1
        if self._current_tick == self._end_tick:
            self._done = True

        step_reward = self._calculate_reward(action)
        self._last_action.append(action)

        self._step_rew = step_reward
        self._total_reward += step_reward

        observation = self._get_observation()
        self._window += 1
        # self.observation_space = spaces.Box(low=-np.inf, high=np.inf,shape=(self._window,7), dtype=np.float64)
        info = dict(
            total_reward=self._total_reward,
            tick=self._time,
            step_action=action

        )

        self._update_history(info)

        return observation, step_reward, self._done, info

    def _get_random_SoggClass(self):

        if self._is_train:
            random_sogg = random.choice(list(self.data.keys()))
            random_class = random.choice(list(self.data[random_sogg].keys()))
        else:
            random_sogg = self._sogg
            random_class = self._class

        return random_sogg, random_class

    def _get_observation(self):

        obs = np.zeros((self._episode_max_len + self._starting_window, 7))
        if self._is_train:
            features = self.data[self.random_sogg][self.random_class]
            obs[:self._window] = features[self._current_tick - self._window:self._current_tick]
        else:
            features = self.data
            obs[:self._window] = np.array(features[self._current_tick - self._window:self._current_tick])

        return obs

    def _update_history(self, info):

        if not self.episode_history:
            self.episode_history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.episode_history[key].append(value)

    def _calculate_reward(self, action):
        
        sm = torch.nn.Softmax()
        a = torch.FloatTensor(action)
        actions_prob = sm(a)
        actions_prob_sort=np.array(actions_prob)
        actions_prob_sort.sort()
        pred = actions_prob_sort[-1]
        pred1 = actions_prob_sort[-2]
        diff=pred-pred1


        penality_coeff = 1
        treshold = self.reward_threshold

        if (pred == actions_prob[0]) and self.random_class == 'Relax':
            if diff > treshold:
                step_reward = actions_prob[0]
                
                self._done = True
            else:
                step_reward = (-0.0003 * (self._time ** (1 / 3)))

        elif (pred == actions_prob[0]) and self.random_class != 'Relax':

            if diff > treshold:
                step_reward = -actions_prob[0] * penality_coeff
   
                self._done = True
            else:
                step_reward = (-0.0003 * (self._time ** (1 / 3)))

        if (pred == actions_prob[1]) and self.random_class == 'EmotionalStress':
            if diff > treshold:
                step_reward = actions_prob[1]

                self._done = True
            else:
                step_reward = (-0.0003 * (self._time ** (1 / 3)))

        elif (pred == actions_prob[1]) and self.random_class != 'EmotionalStress':

            if diff > treshold:
                step_reward = -actions_prob[1] * penality_coeff

                self._done = True
            else:
                step_reward = (-0.0003 * (self._time ** (1 / 3)))

        if (pred == actions_prob[2]) and self.random_class == 'PhysicalStress':
            if diff > treshold:
                step_reward = actions_prob[2]

                self._done = True
            else:
                step_reward = (-0.0003 * (self._time ** (1 / 3)))

        elif (pred == actions_prob[2]) and self.random_class != 'PhysicalStress':

            if diff > treshold:
                step_reward = -actions_prob[2] * penality_coeff

                self._done = True
            else:
                step_reward = (-0.0003 * (self._time ** (1 / 3)))
    

        step_reward = np.array(step_reward)


        if self._time >= self._episode_max_len:
            self._done = True

        return step_reward

