# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 08:26:06 2024

@author: VuralBayraklii
"""
from model import DQN
import random
import numpy as np
import gymnasium as gym
import pygame
import flappy_bird_env
import cv2
import matplotlib.pyplot as plt
import os

env = gym.make("FlappyBird-v0", render_mode="rgb_array")

state_size = (80, 75, 1)

action_size = env.action_space.n

def preprocess_state(state):

    image = cv2.resize(state, (75, 80), interpolation=cv2.INTER_AREA)  # OpenCV ile boyutlandırma

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.equalizeHist(image)

    image = (image - 128) / 128 - 1
    
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=3)

    return image
        
num_episodes = 500

num_timesteps = 20000

batch_size = 8

num_screens = 4     

dqn = DQN(state_size, action_size)

done = False

time_step = 0

for i in range(num_episodes):
    
    Total = 0
    
    state = preprocess_state(env.reset()[0])
    
    for t in range(num_timesteps):
        
        env.render()

        time_step += 1
        
        if time_step % dqn.update_rate == 0:
            dqn.update_target_network()
        
        action = dqn.epsilon_greedy(state)
        
        next_state, reward, done, _, sozluk = env.step(action)
        
        next_state = preprocess_state(next_state)
        
        dqn.store_transistion(state, action, reward, next_state, done)
        
        state = next_state
        
        Total += reward

        if done:
            print('Episode: ',i, ',' 'Total', Total)
            break
            
        if len(dqn.replay_buffer) > batch_size:
            dqn.train(batch_size)
            dqn.save_model(os.psth.join(os.getcwd(), "dqn_model.h5"))

