from gym.envs.registration import register

import gym

__author__ = "Ming Yu"

register(
    id="Snake-v1",
    entry_point="gym_game.environment:SnakeEnv",
)


def make(name, **kwargs):
    return gym.make(name, **kwargs)
