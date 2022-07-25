import gym
from stable_baselines3 import  SAC, A2C, DDPG, PPO, TD3 #ACKRT, HER(HerReplayBuffer,), GAIL, TRPO
import os
from datetime import datetime
from ctrl_env.ctrl_env import CtrlEnv

now = datetime.now().strftime("%b%d")
models_dir = f'trains/{now}_models'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

rl_model = 'SAC'
n_stamp = len(os.listdir(models_dir))
models_dir = f'{models_dir}/{rl_model}{n_stamp}'
logdir = f'trains/{now}_logs'

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = CtrlEnv()
env.reset()

if rl_model == 'SAC':
    model = SAC('MlpPolicy', env, verbose=1, tensorboard_log=logdir)
elif rl_model == 'A2C':
    model = A2C('MlpPolicy', env, verbose=1, tensorboard_log=logdir)
elif rl_model == 'DDPG':
    model = DDPG('MlpPolicy', env, verbose=1, tensorboard_log=logdir)
elif rl_model == 'PPO':
    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)
elif rl_model == 'TD3':
    model = TD3('MlpPolicy', env, verbose=1, tensorboard_log=logdir)
else:
    print(f'{rl_model} → No disponible aún...')

# elif rl_model == 'HER':
#     model = HerReplayBuffer('MlpPolicy', env, tensorboard_log=logdir)
# elif rl_model == 'ACKTR':
#     model = ACKTR('MlpPolicy', env, verbose=1, tensorboard_log=logdir)


TIMESTEPS = 10000
for i in range(30):
    model.learn(total_timesteps=TIMESTEPS,
     reset_num_timesteps=False,
     tb_log_name=f'{rl_model}{n_stamp}')
    model.save(f'{models_dir}/{TIMESTEPS*i}')
    
env.close()

