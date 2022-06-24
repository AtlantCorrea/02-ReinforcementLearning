from binascii import rlecode_hqx
import gym
from stable_baselines3 import  SAC, A2C, DDPG, PPO, TD3 #ACKRT, HER(HerReplayBuffer,), GAIL, TRPO

env = gym.make('Pendulum-v1')
env.reset()

rl_model = input("Modelo: ")
rl_step = input("Step: ")

models_dir = f'models/{rl_model}'
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

episides = 10
for ep in range(episides):
    obs = env.reset()
    done = False
    for i in range(200):
        env.render()
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)

env.close()