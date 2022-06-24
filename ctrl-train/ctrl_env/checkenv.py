from stable_baselines3.common.env_checker import check_env
from ctrl_env import CtrlEnv

env = CtrlEnv()
check_env(env)

print('all ok?')