import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from network import ActorCriticNetwork, ActorCriticNetworkCNN
import torch.nn.functional as F
import torch.autograd as autograd

class PPOAgent:
    def __init__(self, input_dim: int, output_dim: int, learning_rate: float = 3e-4, gamma: float = 0.99, epsilon: float = 0.2, value_coef: float = 0.5, entropy_coef: float = 0.01, cnn_policy: bool = False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if cnn_policy:
            self.actor_critic = ActorCriticNetworkCNN(input_dim, output_dim).to(self.device)
        else:
            self.actor_critic = ActorCriticNetwork(input_dim, output_dim).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        action_probs, _ = self.actor_critic(state.to(self.device))
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), action_probs[0, action]

    def update(self, states, actions, old_log_probs, old_action_probs, rewards, next_states, dones):
        states = autograd.Variable(torch.FloatTensor(states).to(self.device), requires_grad=True)
        actions = autograd.Variable(torch.LongTensor(actions).to(self.device), requires_grad=False)
        old_log_probs = autograd.Variable(torch.FloatTensor(old_log_probs).to(self.device), requires_grad=False)
        old_action_probs = autograd.Variable(torch.FloatTensor(old_action_probs).to(self.device), requires_grad=False)
        rewards = autograd.Variable(torch.FloatTensor(rewards).to(self.device), requires_grad=False)
        next_states = autograd.Variable(torch.FloatTensor(next_states).to(self.device), requires_grad=False)
        dones = autograd.Variable(torch.FloatTensor(dones).to(self.device), requires_grad=False)

        _, next_values = self.actor_critic(next_states)
        returns = self.compute_returns(rewards, next_values, dones)
        advantages = returns - self.actor_critic(states)[1].detach().cpu()

        for _ in range(10):  
            action_probs, state_values = self.actor_critic(states)
            action_probs = action_probs
            state_values = state_values
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages.to(self.device)
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages.to(self.device)
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(state_values.squeeze(), returns.to(self.device))
            entropy = dist.entropy().mean()
            
            loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def compute_returns(self, rewards, next_values, dones):
        returns = []
        R = next_values[-1] * (1 - dones[-1])
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * (1 - dones[step])
            returns.insert(0, R)
        return torch.tensor(returns)

    def save(self, filepath: str):
        torch.save(self.actor_critic.state_dict(), filepath)

    def load(self, filepath: str):
        self.actor_critic.load_state_dict(torch.load(filepath))