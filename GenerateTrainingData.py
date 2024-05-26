import gym
import imageio
import os
import numpy as np
import pandas as pd

env = gym.make("MountainCar-v0", render_mode = 'rgb_array')
env.action_space.seed(82)

n = 500

observation, info = env.reset(seed=205)

fs = []
data = []
i = 0

title = 'random'


for _ in range(n):
    frame = env.render()
    fs.append(frame)

    lastobs = observation
    if title == 'random':
        action = env.action_space.sample()
    else:
        if observation[1] > 0:
            action = 2
        else:
            action = 1
    observation, reward, terminated, truncated, info = env.step(action)

    data.append([i, lastobs[0], lastobs[1], action, observation[0], observation[1], reward, terminated, truncated])

    i += 1

    if terminated or truncated:
        observation, info = env.reset()
        i = 0

data = np.array(data)
df = pd.DataFrame(data, columns=['Frame', 'Previous Position', 'Previous Velocity', 'Action', 
                                 'Position', 'Velocity', 'Reward', 'Terminal', 'Truncated'])
df.to_csv('videos/data_' + title+'_test_agent.csv')
env.close()

imageio.mimwrite(os.path.join('./videos/', title+'_test_agent.gif'), fs, fps=60)