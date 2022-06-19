from gym.envs.registration import register

register(
    id='ctrl-v0',
    entry_point='gym_ctrl.env:CtrlEnv',
)