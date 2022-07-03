import gym
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

episides = 2
for ep in range(-episides, episides+1):
    obs = env.reset(x0 = ep)
    # obs = env.reset(x0= random.randint(-3,3), ref= random.randint(-3,3))
    done = False
    for i in range(250):
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
    if done:
        # break
        pass
    env.render()
    input('ok?')