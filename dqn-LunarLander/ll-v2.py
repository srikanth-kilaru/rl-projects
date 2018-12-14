import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.optimizers import RMSprop, Adam
import random 
from collections import deque
import matplotlib.pyplot as plt
from keras import backend as K
from IPython import display
from pylab import rcParams
import warnings

rcParams['figure.figsize'] = 10,5

#%matplotlib inline
#%matplotlib notebook


class DeepQNet:
    def __init__(self, 
                 simulator, 
                 model_params, 
                 use_target,
                 target_update_freq,
                 gamma, 
                 eps_init, 
                 eps_min, 
                 eps_decay, 
                 batch_size,
                 min_samples,
                 memory_size):
        
        self.simulator = simulator
        
        self.state_dim = simulator.observation_space.shape[0]
        self.num_actions = simulator.action_space.n    
        self.model = self.create_model(model_params)
        self.use_target = use_target
        if self.use_target:
            self.target_update_freq = target_update_freq
            self.target_model = self.create_model(model_params)
                
        self.eps = eps_init
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.gamma = gamma
        
        self.batch_size = batch_size
        self.min_samples = min_samples
        self.memory = deque(maxlen=memory_size)
        
        self.steps = 0
        
    def create_model(self, params):
        layers = params['layers']
        hidden_activation = 'relu'
        final_activation = 'linear'
        model = Sequential()
        model.add(Dense(layers[0], input_dim=self.state_dim,
                        activation=hidden_activation))
        for i in layers[1:]:
            model.add(Dense(i, activation=hidden_activation))
        model.add(Dense(self.num_actions, activation=final_activation))
        model.compile(loss=params['loss'], optimizer=params['optimizer'])
        model.summary()
        return model
    
    def choose_action(self, state, force_random=False):
        if force_random or random.random() < self.eps:
            action = random.randrange(self.num_actions)
        else: 
            action = np.argmax(self.model.predict(state)[0])
            
        return action
    
    def record(self, state, action, next_state, reward, done):
        self.memory.append((state, action, next_state, reward, done))

        self.steps += 1
        if done:
            self.eps = max(self.eps_min, self.eps_decay*self.eps)
            
        if self.use_target and self.steps % self.target_update_freq == 0:
            self.target_model.set_weights(self.model.get_weights())
        
    def run_episode(self):
        state = self.simulator.reset()
        state = state.reshape((1,-1))
        done = False
        steps = 0
        reward = 0
        
        while not done:
            steps += 1
            action = self.choose_action(state)
            next_state, r, done, _ = self.simulator.step(action)
            next_state = next_state.reshape((1,-1))
            
            reward += r
            self.record(state, action, next_state, r, done)
            self.train()
            state = next_state
            
        return reward
    
    def train(self):
        if len(self.memory) < self.min_samples:
            return 0
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, next_states, rewards, done = zip(*batch)
        
        states = np.asarray(states).reshape((self.batch_size, -1))
        next_states = np.asarray(next_states).reshape((self.batch_size, -1))
        
        q_model = self.model
        nq_model = self.target_model if self.use_target else self.model
        
        q = q_model.predict(states)
        nq = nq_model.predict(next_states)
        
        targets = np.asarray(rewards)
        for i, d in enumerate(done):
            if not d:
                targets[i] += self.gamma*np.amax(nq[i])
            
            
        y = q
        for i, a in enumerate(actions):
            y[i, a] = targets[i]
            
        X = states
        return self.model.fit(X, y, epochs=1, verbose=0).history['loss']
        
def run_exp(dqn, max_episodes, long=100, short=5, early_stop=200):
    fig,ax = plt.subplots()
    ax.clear()
    rewards = []
    avg_rewards_long = []
    avg_rewards_short =[]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for i in range(max_episodes):
            reward = dqn.run_episode()
            rewards.append(reward)

            avg_rewards_long.append(np.mean(rewards[-long:]))
            avg_rewards_short.append(np.mean(rewards[-short:]))

            ax.plot(np.arange(len(rewards)), rewards, color='black',
                    linewidth=0.5)
            ax.plot(np.arange(len(avg_rewards_short)),
                    avg_rewards_short, color='orange')
            ax.plot(np.arange(len(avg_rewards_long)),
                    avg_rewards_long, color='blue')
            ax.set_title(f'Ep {i + 1}/{max_episodes}, Rewards = {int(reward)}/{int(avg_rewards_short[-1])}/{int(avg_rewards_long[-1])}')
            fig.canvas.draw()
            
            if avg_rewards_long[-1] >= early_stop:
                return True
    
        return False

def hubert_loss(y_true, y_pred):
    err = y_pred - y_true
    return K.mean( K.sqrt(1+K.square(err))-1, axis=-1 )

if __name__ == '__main__':
    
    max_episodes = 1000
    
    # success = False
    # while not success:
    model_params = {
        'loss': hubert_loss,
        'optimizer' : Adam(lr=0.0005),
        'layers': [128, 128]
    }
    
    env = gym.make('LunarLander-v2')
    dqn = DeepQNet(env,
                   model_params=model_params,
                   use_target=True,
                   target_update_freq=500,
                   gamma=0.99, 
                   eps_init=1.0, 
                   eps_min=0, 
                   eps_decay=0.98,
                   batch_size=32,
                   min_samples=1000,
                   memory_size=500000)
    
    success = run_exp(dqn, max_episodes, early_stop=200)
    
    test_rewards = []
    max_simulations = 1000

    print('Running simulations')
    for i in range(max_simulations):
        print(f'Running test simulation {i} of {max_simulations}...', end='\r')
        
        state = env.reset().reshape((1,-1))
        done = False
        reward = 0
        while not done:
            env.render()
            action = dqn.choose_action(state)
            next_state, r, done, _ = env.step(action)
            state = next_state.reshape((1,-1))
            reward += r
            
            test_rewards.append(reward)
            '''
            print()
            n, bins, patches = plt.hist(np.asarray(test_rewards), 100)
            plt.show()
            np.mean(test_rewards)
            '''
