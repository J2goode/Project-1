import logging
import random
import numpy as np
import torch as t
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


# environment parameters
FRAME_TIME = 0.1  # time interval (10th of a second)
GRAVITY_ACCEL = 9.81  # gravity constant (in m/s^2)
AIR_DENS = 1.204  # density of air (in kg/m^3)
PLATFORM_HEIGHT = 0.08  # landing platform height (in m)

# Rocket parameters
MASS_CONST = 10  # mass of rocket (in kg)
AREA_CONST = 5  # cross-sectional area (in m^2)
BOOST_ACCEL = 10  # thrust constant (in m/s^2)
DRAG_COEFF = 1.05  # coefficient of drag (depends on cross section, 1.05 is rough estimate of rocket base)






class Dynamics(nn.Module):

    def __init__(self):
        super(Dynamics, self).__init__()

    @staticmethod
    def forward(state, action):
        """
        action: thrust or no thrust
        state[0] = y
        state[1] = y_dot
        """

        # calculate drag force, then the downward acting force
        DRAG_FORCE = 0.5*AIR_DENS*DRAG_COEFF*AREA_CONST*state[1]**2
        DOWN_FORCE = MASS_CONST*GRAVITY_ACCEL-DRAG_FORCE

        # calculate downward acceleration
        DOWN_ACCEL = DOWN_FORCE/MASS_CONST

        # Note: Here gravity is used to change velocity which is the second element of the state vector
        # Normally, we would do x[1] = x[1] + gravity * delta_time
        # but this is not allowed in PyTorch since it overwrites one variable (x[1]) that is part of the computational graph to be differentiated.
        # Therefore, I define a tensor dx = [0., gravity * delta_time], and do x = x + dx. This is allowed...
        delta_state_down = t.tensor([0., DOWN_ACCEL * FRAME_TIME])

        # Thrust
        # Note: Same reason as above. Need a 2-by-1 tensor.
        delta_state = BOOST_ACCEL * FRAME_TIME * t.tensor([0., -1.]) * action

        # Update position
        distance = t.tensor([-PLATFORM_HEIGHT,0])
        state = state + distance

        # Update velocity
        state = state + delta_state + delta_state_down

        # Update state
        # Note: Same as above. Use operators on matrices/tensors as much as possible. Do not use element-wise operators as they are considered inplace.
        step_mat = t.tensor([[1., FRAME_TIME],
                             [0., 1.]])
        state = t.matmul(step_mat, state)

        return state


class Controller(nn.Module):

    def __init__(self, dim_input, dim_hidden, dim_output):
        """
        dim_input: # of system states
        dim_output: # of actions
        dim_hidden: up to you
        """
        super(Controller, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.Tanh(),
            nn.Linear(dim_hidden, dim_output),
            # You can add more layers here
            nn.Sigmoid()
        )

    def forward(self, state):
        action = self.network(state)
        return action


class Simulation(nn.Module):

    def __init__(self, controller, dynamics, T):
        super(Simulation, self).__init__()
        self.state = self.initialize_state()
        self.controller = controller
        self.dynamics = dynamics
        self.T = T
        self.action_trajectory = []
        self.state_trajectory = []

    def forward(self, state):
        self.action_trajectory = []
        self.state_trajectory = []
        for _ in range(T):
            action = self.controller.forward(state)
            state = self.dynamics.forward(state, action)
            self.action_trajectory.append(action)
            self.state_trajectory.append(state)
        return self.error(state)

    @staticmethod
    def initialize_state():
        state = [random.uniform(PLATFORM_HEIGHT,11), 0.]  # made rocket start at random height from the ground
        # Still zero initial velocity, was too hard to control for
        return t.tensor(state, requires_grad=False).float()

    def error(self, state):
        return state[0]**2 + state[1]**2


class Optimize:
    def __init__(self, simulation):
        self.simulation = simulation
        self.parameters = simulation.controller.parameters()
        self.optimizer = optim.LBFGS(self.parameters, lr=0.01)

    def step(self):
        def closure():
            loss = self.simulation(self.simulation.state)
            self.optimizer.zero_grad()
            loss.backward()
            return loss

        self.optimizer.step(closure)
        return closure()

    def train(self, epochs):
        for epoch in range(epochs):
            loss = self.step()
            print('[%d] loss: %.3f' % (epoch + 1, loss))

        self.visualize()

    def visualize(self):
        data = np.array([self.simulation.state_trajectory[i].detach().numpy() for i in range(self.simulation.T)])
        x = data[:, 0]
        y = data[:, 1]
        plt.plot(x, y)
        plt.grid()
        plt.show()


# Now it's time to run the code!

T = 100  # number of time steps
dim_input = 2  # state space dimensions
dim_hidden = 6  # latent dimensions
dim_output = 1  # action space dimensions
d = Dynamics()  # define dynamics
c = Controller(dim_input, dim_hidden, dim_output)  # define controller
s = Simulation(c, d, T)  # define simulation
print("SIMULATION START")
print("Initial state: ", s.state.float())
o = Optimize(s)  # define optimizer
o.train(60)  # solve the optimization problem