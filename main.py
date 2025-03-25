import os
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import random
import gymnasium as gym
from gymnasium.spaces import Box 
from gymnasium.wrappers import TimeLimit
import matplotlib.pyplot as plt
from vizdoom import gymnasium_wrapper
import cv2


env =gym.make("VizdoomCorridor-v0", render_mode='human')

def step(final_action):

    next_state, reward, done, truncated, info = env.step(final_action)
        


    return next_state, reward, done, truncated, info




def reset():
    final_action = 0
    state_,info = env.reset()
    
    return state_, info

    

    









