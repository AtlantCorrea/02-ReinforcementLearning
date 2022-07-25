import gym
from stable_baselines3 import  SAC, A2C, DDPG, PPO, TD3 #ACKRT, HER(HerReplayBuffer,), GAIL, TRPO
import os
from datetime import datetime
from ctrl_env.ctrl_env import CtrlEnv


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


n_stamp = len(os.listdir(rl_folder))
start_step = int(rl_step)*1000//1000000

if not os.path.exists(f'{models_dir}_{start_step}k_r{n_stamp}'):
    os.makedirs(f'{models_dir}_{start_step}k_r{n_stamp}')


TIMESTEPS = 10000
model.save(f'{models_dir}_{start_step}k_r{n_stamp}/{0}')
for i in range(30):
    model.learn(total_timesteps=TIMESTEPS, 
    reset_num_timesteps=False, 
    tb_log_name=f'{rl_model}_r{n_stamp}')

    model.save(f'{models_dir}_{start_step}k_r{n_stamp}/{TIMESTEPS*i}')
    
env.close()

