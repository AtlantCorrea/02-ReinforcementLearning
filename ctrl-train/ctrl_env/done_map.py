from cProfile import label
from ctrl_env import CtrlEnv
import matplotlib.pyplot as plt
import numpy as np

env = CtrlEnv()

X,Y =[],[]
for x in range(-3,3):
    Y.append(np.absolute(x))
    X.append(x)

plt.plot(X,Y)
plt.show()
input('ok')




i=0
l, t, r, z, x = [], [], [], [], []
for ref in range (-3,3):
    for x0 in range(-3,3):
        env.reset(x0=x0, ref=ref)
        for x1 in range(-5,5):
            _,d = env.done(x1, i)
            if d:
                l.append(0.5)
            else:
                l.append(0)
            r.append(ref)
            z.append(x0)
            x.append(x1)
            t.append(i)
            i+=1

plt.plot(t,l, label='dones')
plt.plot(t,r, label='Ref')
plt.plot(t,z, label='X0')
plt.plot(t,x, label='x0')
plt.grid()
plt.legend()
plt.show()
input('ok?')
