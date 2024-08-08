import gym
import torch
import torch.nn as nn
import gym_game
from policy_gradient_agent import PPOAgent

env = gym.make('Snake-v1', render_mode='human')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = PPOAgent(state_dim, action_dim, cnn_policy=True)
model_path = 'ppo_snake_best.pth'
agent.load(model_path)

fps = 100
import time
state, _ = env.reset()
for _ in range(500):
    action, _, _ = agent.select_action(state)
    next_state, reward, done, score, _ = env.step(action)
    state = next_state
    time.sleep(1/fps)
    if done:
        break
env.close()