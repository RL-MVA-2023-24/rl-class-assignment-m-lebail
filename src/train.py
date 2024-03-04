from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from copy import deepcopy
import numpy as np
import tqdm

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)


    
class DeepQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        nb_neurons = 256
        self.fc1 = nn.Linear(state_dim, nb_neurons)
        self.fc2 = nn.Linear(nb_neurons, nb_neurons)
        self.fc3 = nn.Linear(nb_neurons, nb_neurons)
        self.fc4 = nn.Linear(nb_neurons, nb_neurons)
        self.fc5 = nn.Linear(nb_neurons,action_dim)
    def forward(self, x):
        x = F.elu(self.fc1(x))
        x_1 = self.fc2(x)
        x = F.elu(x+x_1)
        x_2 = self.fc3(x)
        x = F.elu(x+x_2)
        x_3 = self.fc4(x)
        x = F.elu(x+x_3)
        x = self.fc5(x)
        
        return x



class ProjectAgent:

    def __init__(self):
        #device = "cuda" if next(model.parameters()).is_cuda else "cpu"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nb_actions = env.action_space.n
        self.gamma = 0.97
        self.batch_size = 256
        buffer_size = 1000000
        self.memory = ReplayBuffer(buffer_size,self.device)
        self.epsilon_max = 1.
        self.epsilon_min = 0.01
        self.epsilon_stop = 40000
        self.epsilon_delay = 400
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.model = DeepQNetwork(env).to(self.device)
        self.target_model = deepcopy(self.model).to(self.device)
        self.criterion = torch.nn.SmoothL1Loss()
        lr = 0.001
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.nb_gradient_steps = 2
        self.update_target_strategy = 'ema' 
        self.update_target_freq = 500
        self.update_target_tau = 0.005
        self.monitoring_nb_trials =  0
        self.pre_fill = 5000
        self.check_save_model = 200

    def greedy_action(self, state):
      with torch.no_grad():
          Q = self.model(torch.Tensor(state).unsqueeze(0).to(self.device))
          return torch.argmax(Q).item()


    def MC_eval(self, env, nb_trials):
        MC_total_reward = []
        MC_discounted_reward = []
        for _ in range(nb_trials):
            x,_ = env.reset()
            done = False
            trunc = False
            total_reward = 0
            discounted_reward = 0
            step = 0
            while not (done or trunc or (step>=200)):
                a = self.greedy_action(x)
                y,r,done,trunc,_ = env.step(a)
                x = y
                total_reward += r
                discounted_reward += self.gamma**step * r
                step += 1
            MC_total_reward.append(total_reward)
            MC_discounted_reward.append(discounted_reward)
        return np.mean(MC_discounted_reward), np.mean(MC_total_reward)



    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)

            # Double Q-Learning
            
            with torch.no_grad():
                action_indices = self.model(Y).max(1)[1].unsqueeze(1) 
        
            # Use the target model to evaluate the action
            QYmax = self.target_model(Y).gather(1, action_indices).squeeze() 
        
            update = R + (1 - D) * self.gamma * QYmax  
        
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train(self, env, max_episode):
        episode_return = []
        
        episode = 0
        best_so_far = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)
            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = self.greedy_action(state)
            # step
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward
                
            # train
            for _ in range(self.nb_gradient_steps):
                self.gradient_step()
            # update target network if needed
            if self.update_target_strategy == 'replace':
                if step % self.update_target_freq == 0:
                    self.target_model.load_state_dict(self.model.state_dict())
            if self.update_target_strategy == 'ema':
                target_state_dict = self.target_model.state_dict()
                model_state_dict = self.model.state_dict()
                tau = self.update_target_tau
                for key in model_state_dict:
                    target_state_dict[key] = tau*model_state_dict[key] + (1-tau)*target_state_dict[key]
                self.target_model.load_state_dict(target_state_dict)
           
            # next transition
            step += 1
            #print('step : ',step)
            if done or trunc or (step==200):
                episode += 1
                # Monitoring
                if self.monitoring_nb_trials>0:
                
                    episode_return.append(episode_cum_reward)
                    print("Episode ", '{:2d}'.format(episode),
                          ", epsilon ", '{:6.2f}'.format(epsilon),
                          ", batch size ", '{:4d}'.format(len(self.memory)),
                          ", ep return ", '{:e}'.format(episode_cum_reward),
                          sep='')
                else:

                    episode_return.append(episode_cum_reward)
                    print("Episode ", '{:2d}'.format(episode),
                          ", epsilon ", '{:6.2f}'.format(epsilon),
                          ", batch size ", '{:4d}'.format(len(self.memory)),
                          ", ep return ", '{:e}'.format(episode_cum_reward),
                          sep='')
                

                if episode % self.check_save_model == 0:
                    MC_discounted_reward_mean, MC_total_reward_mean = self.MC_eval(env, 10)
                    print("done compute cum reward")
                    if MC_total_reward_mean > best_so_far:
                        self.save('best_agent.pth')
                        best_so_far = MC_total_reward_mean
                        print("Best total cumulated reward is : ",'{:e}'.format(best_so_far))

                state, _ = env.reset()
                episode_cum_reward = 0
            else:
                state = next_state
        return episode_return


    def act(self, observation, use_random=False):
        return self.greedy_action(observation)

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        

    def load(self):
        self.model.load_state_dict(torch.load('best_agent.pth',map_location=torch.device("cpu")))
        self.model.eval()  



def pre_fill_buffer(env,agent):
    # pre-fill the replay buffer
    x,_ = env.reset()
    evolution = tqdm.tqdm(total=agent.pre_fill)
    for t in range(agent.pre_fill):
        a = env.action_space.sample()
        y, r, d, tr, _ = env.step(a)
        agent.memory.append(x, a, r, y, d)
        if d or tr:
            x,_ = env.reset()
        else:
            x = y
        evolution.update(1)
    evolution.close()

