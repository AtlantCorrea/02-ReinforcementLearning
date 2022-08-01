import gym
import os
import numpy as np
from stable_baselines3 import  SAC, A2C, DDPG, PPO, TD3 #ACKRT, HER(HerReplayBuffer,), GAIL, TRPO
from ctrl_env.ctrl_env import CtrlEnv
import matplotlib.pyplot as plt

env = CtrlEnv()
env.reset()

rl_folder = input('Folder:')
rl_model = input("Modelo: ")
rl_step = input("Step: ")

models_dir = f'{rl_folder}/{rl_model}'
model_path = f'{models_dir}/{rl_step}.zip'

if 'SAC' in rl_model:
    model = SAC.load(model_path, env=env)
elif 'A2C' in rl_model:
    model = A2C.load(model_path, env=env)
elif 'DDPG'in rl_model:
    model = DDPG.load(model_path, env=env)
elif 'PPO' in rl_model:
    model = PPO.load(model_path, env=env)
elif 'TD3' in rl_model:
    model = TD3.load(model_path, env=env)
else:
    print(f'{rl_model} → No disponible aún...')

# elif rl_model == 'HER':
#     model = HerReplayBuffer.load(model_path, env=env)
# elif rl_model == 'ACKTR':
#     model = ACKTR.load(model_path, env=env)

txt_d = ''
if os.path.exists(f'{models_dir}\Description.txt'):
    with open(f'{models_dir}\Description.txt') as f:
        txt_d = f.readline()

print(f'\n\n'+'-'*10)
print(f'Model: {rl_folder}\{rl_model}\{rl_step}.zip')        
print(f"Descripcion: {txt_d}\n"+'-'*10)

episides = 2
for ep in range(-episides, episides+1):
    r = []
    obs = env.reset(ref = ep)
    # obs = env.reset(x0= random.randint(-3,3), ref= random.randint(-3,3))
    done = False
    for i in range(250):
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        r.append(reward)
    if done:
        # break
        pass
    env.render(title=f'Ref={ep}')
    input(f'Reference = {ep}\tmean_rew: {np.mean(r):.4}\t → ok?')


input('ok?')