import gym
env = gym.make("Pendulum-v1")
observation = env.reset()
for _ in range(1000):
   env.render()
   action = env.action_space.sample()  # User-defined policy function
   observation, reward, done, info = env.step(action)

   if done:
      observation, info = env.reset(return_info=True)
env.close()