from random import random
from ctrl_env import CtrlEnv
import matplotlib.pyplot as plt

env = CtrlEnv()
episodes = 5

for episode in range(episodes):
    done = False
    obs = env.reset(x0=0, ref=1)
    for step in range(20):
        action = env.action_space.sample()
        print("Step {} → action {}".format(step + 1, action))
        obs, reward, done, info = env.step(action)
        # print('obs=', obs, 'reward=', reward, 'done=', done)
        
        if done:
            print("Goal reached!", "reward=", reward, f'done:{done}')
            break
    env.render()
    # fig, ax = plt.subplots()
    # env.plot_info(ax, info, 'Test', 'Time(s)', u=False)
    # fig.show()
input('ok?')
