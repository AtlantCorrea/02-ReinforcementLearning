import gym
from ctrl_env.ctrl_env import CtrlEnv

env = CtrlEnv()
env.reset()
observation = env.reset()
for _ in range(1000):
   action = env.action_space.sample()  
   observation, reward, done, info = env.step(action)

   if done:
      break
env.render()
env.reset()
input('ok?')