import gym
from ctrl_env.ctrl_env import CtrlEnv

env = CtrlEnv()
observation = env.reset(ref=1)
for _ in range(1000):
   action = env.action_space.sample()  
   observation, reward, done, info = env.step(action)

   if done:
      break
env.render()
input('ok?')