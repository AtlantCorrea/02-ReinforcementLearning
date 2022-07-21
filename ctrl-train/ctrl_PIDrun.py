from PID import PID_ctrl
from ctrl_env.ctrl_env import CtrlEnv

env = CtrlEnv()
observation = env.reset(ref=1)
pid_agent = PID_ctrl(kp=1, ki=0.05, kd=0.5)
print(observation)
for _ in range(1000):
   error = env.ref - observation[0]
   action = pid_agent.step(error)
   observation, reward, done, info = env.step(action)

   if done:
      break
env.render()
input('ok?')