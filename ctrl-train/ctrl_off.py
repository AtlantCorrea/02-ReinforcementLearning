import numpy as np
from ctrl_env.ctrl_env import CtrlEnv

R = []
env = CtrlEnv()
observation = env.reset(ref=1)
for _ in range(500):
   action = np.float32(0)
   observation, reward, done, info = env.step(action)
   R.append(reward)
   if done:
      break
env.render()
print(f'mean_rew: {np.mean(R)}')
input('ok?')