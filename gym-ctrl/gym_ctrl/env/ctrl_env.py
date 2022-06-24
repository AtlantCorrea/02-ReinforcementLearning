import gym
import numpy as np
from gym import logger
from gym import spaces
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import matplotlib.colors as mcolors

def random_value(amplitud):
  norm_value = (np.random.rand(1)[0]-0.5)*2
  random_value = norm_value * amplitud
  return random_value

class CtrlEnv(gym.Env):
    def __init__(self, x0 = random_value(3), ref = random_value(3), T=10):
    # def __init__(self, x0 = 0, ref = 1, T=10):
        super(CtrlEnv, self).__init__()
        self.mdel_param = {'m': 1, 'b':1, 'k':1}
        self.scale_action = 5

        self.T = T
        self.x0 = x0
        self.ref = ref
        self.error = ref-x0

        self.x = np.array([[self.x0,0, self.ref]])
        self.u = np.array([0])
        self.t = np.array([0])
        self.r = np.array([-np.absolute(self.error)])
        self.dt = 20/1000
        self.plot = None

        self.x_threshold = 1.5 * np.absolute(self.ref) 

        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        # self.observation_space = spaces.Box(low=-10, high=10, shape=(3,), dtype=np.float32)

        # self.steps_beyond_done = None
        self.epoch = 0

    def reset(self, x0 = random_value(3), ref = random_value(3), T=10):
    # def reset(self, x0 = 0, ref = 1, T=10):
        self.T = T
        self.x0 = x0
        self.ref = ref
        self.error = ref-x0

        self.x = np.array([[self.x0,0, self.ref]])
        self.u = np.array([0])
        self.t = np.array([0])
        self.r = np.array([-np.absolute(self.error)])
        self.dt = 20/1000
        
        self.x_threshold = 1.5 * np.absolute(self.ref)
        observation = self.x[0]
        self.epoch = 0
        return np.array(observation, dtype=np.float32)  # reward, done, info can't be included

    def step(self, action):
        # Timestep
        t = self.t[-1] + self.dt
        self.t = np.vstack((self.t, t))

        # Action
        action = action * self.scale_action
        self.u = np.vstack((self.u, action))

        # Observation
        x1, x2 = self.observation(action)
        observation = np.array([x1,x2,self.ref])
        self.x = np.vstack((self.x, observation))
      
        # Error
        error = self.get_error(x1)
        reward = self.reward(error, t)
        self.r = np.vstack((self.r, reward))

        # Done & Info
        done, x_limit = self.done(x1, t)
        if x_limit:
          reward = -1e9
          self.r = np.vstack((self.r[:-1], reward))
        info = self.info(error, reward)
        self.plot = info

        self.epoch+=1
        return np.array(observation, dtype=np.float32), reward, done, info

    def render(self):
        print('render estaba implementado pero lo eliminé, buscar en versiones 3.0')

    def close (self):
        pass

    def observation(self, action):
        u = action
        x = self.x[-1]
        t_i = np.linspace(0, self.dt, 5)
        sol_array =  odeint(self.model, x[:-1], t_i, args = (u,))
        return sol_array[-1]

    def reward(self, error, t):
        reward = -np.absolute(error) * (1 + float(t/2)**1.2) 
        return reward

    def done(self, x1, t):
        x_limit = bool( np.absolute(x1) > self.x_threshold)
        t_limit = bool( t > self.T)
        done = x_limit or t_limit
        return done, x_limit

    def info(self, error, reward):
        info = {'x':self.x,
                'u':self.u,
                't':self.t,
                'r':self.r,
                'dt':self.dt,
                'ref':self.ref,
                'error': error,
                'epoch': self.epoch,
                # 'episode':{'r':reward,'l':self.epoch}}
                # 'episode':{'r':3,'l':4}
                }
        return info

    def get_error(self, obs):
        return self.ref - obs
        
    def model(self, x, t, u = 0):
        x1,x2 = x
        m = self.mdel_param['m']    
        k = self.mdel_param['k']    
        b = self.mdel_param['b']
        
        dx1dt = x2
        dx2dt = -k/m*x1 -b/m*x2 + u/m 
        return [dx1dt, dx2dt]


    def plot_info(self, ax, info = None, title='', xlabel='', x=True, x_dot=True, u=True, r=True, error=True, ref=True):
        if info is None:
          info = self.plot
        time = info['t']
        if x:
          color = mcolors.TABLEAU_COLORS['tab:olive']
          line = '-'
          ax.plot(time, info['x'][:,0], color = color, linestyle= line, label='Obs. x')
        if x_dot:
          color = mcolors.TABLEAU_COLORS['tab:gray']
          line= '-'
          ax.plot(time, info['x'][:,1], color = color, linestyle= line, label="Obs. x'")
        if u:
          color = mcolors.TABLEAU_COLORS['tab:green']  
          line= '-'
          ax.plot(time, info['u'], color = color, linestyle= line, label='Action u')
        if r:
          color = mcolors.TABLEAU_COLORS['tab:red']
          line = '-'
          ax.plot(time, info['r'], color = color, linestyle= line, label='Reward')
        if error:
          color = mcolors.TABLEAU_COLORS['tab:brown']
          line= '-'
          ax.plot(time, info['x'][:,2] - info['x'][:,0], color = color, linestyle= line, label='Error')
        if ref:
          ax.plot(time, np.ones(len(time))*info['ref'] , 'k:', label='Reference')
        ax.plot(time, np.zeros(len(time)) , 'k-', linewidth=0.7)
        
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.legend(bbox_to_anchor=(1.04,1), borderaxespad=0)
        ax.grid()
        return ax

if __name__=='__main__':
  ## Test
  import gym_ctrl
  env = gym.make('ctrl-v0')
  obs = env.reset(x0=0)

  print(env.observation_space)
  print(env.action_space)
  print(env.action_space.sample())

  n_steps = 700
  for step in range(n_steps):
    action = env.action_space.sample()
    print("Step {} → action {}".format(step + 1, action))
    obs, reward, done, info = env.step(action)
    # print('obs=', obs, 'reward=', reward, 'done=', done)
    
    if done:
      print("Goal reached!", "reward=", reward, f'done:{done}')
      break

  fig, ax = plt.subplots()
  env.plot_info(ax, info, 'Test', 'Time(s)', u=False)
  fig.show()
  input('ok?')