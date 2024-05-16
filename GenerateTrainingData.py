import gym
import imageio
import os

env = gym.make("CartPole-v1", render_mode = 'rgb_array')
env.action_space.seed(82)

n = 100

observation, info = env.reset(seed=82)

fs = []

for _ in range(n):
    frame = env.render()
    fs.append(frame)

    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print("observation : ",observation);
    
    if terminated or truncated:
        observation, info = env.reset()
        
env.close()

imageio.mimwrite(os.path.join('./videos/', 'random_agent.gif'), fs, fps=60)