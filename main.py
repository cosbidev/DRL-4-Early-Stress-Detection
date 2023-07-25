# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 11:52:53 2023

@author: Leonardo Furia
"""

from gym import spaces
import torch.nn as nn
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3 import SAC
from lib.utils import DataDownloader
from lib.utils import calculate_rew_threshold
from lib.utils import calculate_rew_mean_threshold
from lib.utils import Evaluate4Classes
from lib.utils import Evaluate3Classes
from lib.env import PhysioEnv4Classes
from lib.env import PhysioEnv3Classes



NUM_CLASSES=4
PATH='./Results'


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[1]
        self.cnn = nn.Sequential(
            nn.Conv1d(n_input_channels, 128, kernel_size=5, stride=1, padding=0),
            nn.GELU(),
            nn.MaxPool1d(5, stride=2),

            nn.Conv1d(128, 128, kernel_size=5, stride=2, padding=0),
            nn.GELU(),
            nn.MaxPool1d(7, stride=3),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float().permute(0, 2, 1)
            ).shape[1]
        print(n_flatten)

        self.linear = nn.Sequential(nn.Linear(n_flatten, 256),
                                    nn.GELU(),                                  
                                    nn.Linear(256, 128),
                                    nn.GELU(),
                                    nn.Linear(128,features_dim)
                                    )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations.permute(0, 2, 1)))




policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=128),
    share_features_extractor=False,
    use_expln=True,
    normalize_images=False,
    n_critics=2,
    net_arch=dict(pi=[128,32], qf=[128,32])
)



rew_threshold=calculate_rew_threshold(num_classes=NUM_CLASSES)


#mean_rew_threshold to stop training
rew_mean_threshold=calculate_rew_mean_threshold(num_classes=NUM_CLASSES)

if NUM_CLASSES==4:

    for s in range(1,21):
    
        train, test = DataDownloader(test_index_subject=s) # download and create dataset
        
        env = PhysioEnv4Classes(data=train,rew_threshold=rew_threshold, window_size=8, episode_max_len=300)
        
        callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=rew_mean_threshold, verbose=1)
        eval_callback = EvalCallback(env, callback_on_new_best=callback_on_best, verbose=1)
        
        
        model = SAC("MlpPolicy", env, use_sde=True, learning_rate=0.0001, batch_size=256, learning_starts=5000,
                use_sde_at_warmup=True, verbose=1, buffer_size=15000, gamma=0.99, policy_kwargs=policy_kwargs,
                tensorboard_log=PATH+'/4Classe_Tensorboard_Sogg{}'.format(s))  # ,action_noise=noise
        
        model.learn(40000, callback=eval_callback)
        
        Evaluate4Classes(model=model,test=test,path=PATH,episode_max_len=300,window_size=8,subject=s)
    
else:
    
    for s in range(1,21):
    
        train, test = DataDownloader(test_index_subject=s) # download and create dataset
        
        env = PhysioEnv3Classes(data=train,rew_threshold=rew_threshold, window_size=8, episode_max_len=300)
        
        callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=rew_mean_threshold, verbose=1)
        eval_callback = EvalCallback(env, callback_on_new_best=callback_on_best, verbose=1)
        
        
        model = SAC("MlpPolicy", env, use_sde=True, learning_rate=0.0001, batch_size=256, learning_starts=5000,
                use_sde_at_warmup=True, verbose=1, buffer_size=15000, gamma=0.99, policy_kwargs=policy_kwargs,
                tensorboard_log=PATH+'/3Classe_Tensorboard_Sogg{}'.format(s))  # ,action_noise=noise
        
        model.learn(40000, callback=eval_callback)
        
        Evaluate3Classes(model=model,test=test,path=PATH,episode_max_len=300,window_size=8,subject=s)
