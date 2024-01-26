# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 08:41:47 2024

@author: VuralBayraklii
"""
import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam

class DQN:
    def __init__(self, state_size, action_size):
        
        self.state_size = state_size
        
        self.action_size = action_size

        self.replay_buffer = deque(maxlen=5000)

        self.gamma = 0.9  

        self.epsilon = 0.8   
        
        self.update_rate = 1000    
        
        self.main_network = self.build_network()
        
        self.target_network = self.build_network()
        
        self.target_network.set_weights(self.main_network.get_weights())
        
    def build_network(self):
        model = Sequential()
        
        model.add(Conv2D(32, (8, 8), strides=4, padding='same', input_shape=self.state_size))
        model.add(Activation('relu'))
        
        model.add(Conv2D(64, (4, 4), strides=1, padding='same'))
        model.add(Activation('relu'))
        
        model.add(Conv2D(128, (3, 3), strides=1, padding='same'))
        model.add(Activation('relu'))
        
        model.add(Flatten())

        model.add(Dense(512, activation='relu'))
        
        model.add(Dense(self.action_size, activation='linear'))
        
        model.compile(loss='mse', optimizer=Adam())

        return model

    def store_transistion(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))
        
    def epsilon_greedy(self, state):
        if random.uniform(0,1) < self.epsilon:
            return np.random.randint(self.action_size)
        
        Q_values = self.main_network.predict(state)
        
        return np.argmax(Q_values[0])

    def train(self, batch_size):
        
        minibatch = random.sample(self.replay_buffer, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            if not done:
            
                target_Q = (reward + self.gamma * np.amax(self.target_network.predict(next_state)))
                
            else:
                target_Q = reward
                
            
            Q_values = self.main_network.predict(state)
            
            Q_values[0][action] = target_Q
            
            self.main_network.fit(state, Q_values, epochs=1, verbose=0)
            
    def update_target_network(self):
        self.target_network.set_weights(self.main_network.get_weights())
