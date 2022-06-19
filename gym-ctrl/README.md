
#### **Description**

This environment corresponds to the version of a feedback-control problem where we care about the step response of a system plant in a closed loop with a controller(the agent).
There will be a reference change at t=0 and the objective is to control the output(y) by changing the input (u).
Every time step:
    - The environment will give the system actual state to the agent → $S = (ref, x1, x2)$.
    - The agent (the controller) will provide actions/inputs(u) to the environment.
    - The environment will get the transition to the new state based on that input(the action).
The episode may end by:
- Exceding the outputs límits → if $|x| > 1.5 \times |Reference|$
- By completing the time límit → $T = 10s$
At the end of the episode the environment will give the reward to the agent.

#### **Action Space**

The action is a `ndarray` with shape `(1,)` which can take values `[-1,1]` indicating the input(u) signal for the system.

| Num | Action                    |
|-----|---------------------------|
| a   | Lower limit to the input  |
| b   | Higer limit to the input  |

#### **Observation Space**

The observation is a `ndarray` with shape `(3,)` with the values corresponding to the following positions and velocities:

| Num | Observation           | Min                  | Max                |
|-----|-----------------------|----------------------|--------------------|
| 0   | Sys. Position         |  -10                 |  10                |
| 1   | Sys. Velocity         |  -10                 |  10                |
| 2   | Reference             |   -3                 |   3                |


**Note:** While the ranges above denote the possible values for observation space of each element, it is not reflective of the allowed values 
of the state space in an unterminated episode. 
Particularly:
-  The system position (index 1) can be take values between `(-inf, inf)`, but the episode terminates if the system
        leaves the `(-1.5 reference, 1.5 reference)` range.
-  The system velocity can be observed between  `(-inf, inf)`, but the episode terminates if the velocity is not in
        the range `(-1.5 reference, 1.5 reference)`.

#### **Rewards**

Since the goal is to keep the response/output(y) as close to the reference as possible, a negative reward will be given every step taken and it's value will be the Error between the reference and the response/output.

The reward it's given by:
$$R = |ref - x| \times (1 + t^2)$$

#### **Starting State**

All observations are assigned a uniformly random value in `(-3, 3)`

#### **Episode Termination**

The episode terminates if any one of the following occurs:
• The position of the system goes further than ±1.5 times the reference.
• Episode length is grater than 10s ( T epochs) 

#### **Arguments**

Exiten tres argumentos que se pueden utilizar:
- $x_0$ para establecer la posición inicial.
- $ref$ para establecer la referencia.
- $T$ para estableer el largo máximo de los episodios. 