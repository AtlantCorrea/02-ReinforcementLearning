import numpy as np
from ctrl_env.ctrl_env import CtrlEnv


env = CtrlEnv()
observation = env.reset(ref=1)
for _ in range(500):
   action = np.float32(0)
   observation, reward, done, info = env.step(action)

   if done:
      break
env.render()
input('ok?')