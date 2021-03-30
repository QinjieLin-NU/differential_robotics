import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import jax
import math
import numpy as np
from jax import random
from jax import grad, jit, vmap, ops,lax

from jax.experimental import loops
import scipy.io as sio
import matplotlib.pyplot as plt
import time
class cartpole():
    def __init__(self):
        self.viewer = None
        return

    def dynamics_PDP(self,state,action,params):
        """
        state: x,q,dx,dq
        this dynamics is referrd from PDP, also the same as
        https://sites.wustl.edu/slowfastdynamiccontrolapproaches/cart-pole-system/cart-pole-dynamics-system/
        """
        g = 10.0
        dt = 0.05  # seconds between state updates
        kinematics_integrator = "euler"

        
        # get state and actions, which is different from gym env
        x, q, dx, dq = state
        U = action 
        mc,mp,l = params
        
        # computer the acceleration
        ddx = (U + mp * jnp.sin(q) * (l * dq * dq + g * jnp.cos(q))) / (
                    mc + mp * jnp.sin(q) * jnp.sin(q))  # acceleration of x
        ddq = (-U * jnp.cos(q) - mp * l * dq * dq * jnp.sin(q) * jnp.cos(q) - (
                    mc + mp) * g * jnp.sin(q)) / (
                        l * mc + l * mp * jnp.sin(q) * jnp.sin(q))  # acceleration of theta
        
        #computer current state
        if kinematics_integrator == "euler":
            x = x + dx * dt
            dx = dx + ddx * dt
            q = q + dq * dt
            dq = dq + ddq * dt
        else:
            dx = dx + ddx * dt
            x = x + dx * dt
            dq = dq + ddq * dt
            q = q + dq * dt
        return jnp.reshape(jnp.array([x, q, dx, dq]),(4,))

    def dynamics(self,state,action,params):
        """
        params[0,1,2]: pole mass, length, cart mass
        """
        tau = 0.05  # seconds between state updates
        kinematics_integrator = "euler"
        gravity = 10.0

        x, x_dot, theta, theta_dot = state

        force = action 

        masscart,masspole,length = params
        total_mass = masspole + masscart
        polemass_length = masspole * length

        costheta = jnp.cos(theta)
        sintheta = jnp.sin(theta)

        temp = (force + polemass_length * theta_dot ** 2 * sintheta) / total_mass
        thetaacc = (gravity * sintheta - costheta * temp) / (
            length * (4.0 / 3.0 - masspole * costheta ** 2 / total_mass)
        )
        xacc = temp - polemass_length * thetaacc * costheta / total_mass

        if kinematics_integrator == "euler":
            x = x + tau * x_dot
            x_dot = x_dot + tau * xacc
            theta = theta + tau * theta_dot
            theta_dot = theta_dot + tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + tau * xacc
            x = x + tau * x_dot
            theta_dot = theta_dot + tau * thetaacc
            theta = theta + tau * theta_dot

        return jnp.reshape(jnp.array([x, x_dot, theta, theta_dot]),(4,))
    
    def loss_fn(self,params,states,actions,next_states):
        """
        the trajectories are generated from state[0:n] and corresponding action[0:n]
        """
        with loops.Scope() as s:
            s.loss = jnp.zeros((actions.shape[0],4))
            s.predicted_states = jnp.zeros((actions.shape[0],4))
            for i in s.range(s.predicted_states.shape[0]):
                s.predicted_states = ops.index_update(s.predicted_states, i,
                                                    self.dynamics_PDP(states[i],actions[i],params) )
            s.loss = s.predicted_states - next_states
            return jnp.mean(jnp.square(s.loss))  # mse

    def loss_fn_traj(self,params,states,actions,next_states):
        """
        the trajctories is generated from initial state[0] and action[0:n]
        """
        with loops.Scope() as s:
            s.loss = jnp.zeros((actions.shape[0],4))
            s.predicted_states = jnp.zeros((actions.shape[0],4))
            s.predicted_states = ops.index_update(s.predicted_states, 0,
                                                    dynamics(states[0],actions[0],params) )
            for i in s.range(s.predicted_states.shape[0]-1):
                s.predicted_states = ops.index_update(s.predicted_states, i+1,
                                                    self.dynamics_PDP(s.predicted_states[i],actions[i+1],params) )
            s.loss = s.predicted_states - next_states
            return jnp.mean(jnp.square(s.loss))  # mse

    def render(self,params,state,num=1,other_state=None):
        masscart,masspole,length = params
        total_mass = masspole + masscart
        polemass_length = masspole * length
        polemass_length = 1.0 # we make it fixed here

        x, theta,x_dot, theta_dot = state

        screen_width = 800#600
        screen_height = 600#400

        world_width = 2.4 * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (1 * polemass_length)# 2 *
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            #add cart
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)

            #add pole
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(0.8, 0.6, 0.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)

            #add second pole
            self.poletrans2 = None
            if(num > 1):
                # cart2 = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
                # self.carttrans2 = rendering.Transform()
                # cart2.add_attr(self.carttrans2)
                # self.viewer.add_geom(cart2)

                pole2= rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
                pole2.set_color(0.4, 0.99, 0.99)
                self.poletrans2 = rendering.Transform(translation=(0, axleoffset))
                pole2.add_attr(self.poletrans2)
                pole2.add_attr(self.carttrans)
                self.viewer.add_geom(pole2)


            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(0.5, 0.5, 0.8)
            self.viewer.add_geom(self.axle)

            #add track
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        if state is None:
            return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        # x = state
        cartx = x * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-theta)
        if(num>1):
            true_x, true_theta, _, _ = other_state
            true_cartx = true_x * scale + screen_width / 2.0  # MIDDLE OF CART
            self.poletrans2.set_rotation(-true_theta)
            # self.carttrans2.set_translation(true_cartx, carty)

        return self.viewer.render(return_rgb_array="human" == "rgb_array")


def render_actions(env,params,init_state,actions,true_states):
    current_state = jnp.array(init_state)
    env.render(params,current_state,num=2,other_state=current_state)
    for i in range(len(actions)):
        action = actions[i]
        next_state = env.dynamics_PDP(current_state,action,params)
        current_state = next_state
        true_state = true_states[i]
        env.render(params,current_state,num=2,other_state = true_state)
        time.sleep(0.1)

def render_state(env,params,states):
    for state in states:
        current_state = jnp.array(state)
        env.render(params,current_state)
        time.sleep(0.1)

def render_result():
    params_list = [ [1.4034629,  1.295249,   0.59333175],
    [1.3975228,  1.2837119, 0.76814085],
    [1.3741106, 1.27681,   0.8425436],
    [1.2707177,  1.2467496,  0.92553055],
    [1.2260766, 1.2267696, 0.9381445],
    [1.0870844, 1.1178433, 0.9686284],
    [1.073072, 1.1016167, 0.9724776] ]


    for p in params_list[0::1]:
        render_actions(env,p,test_s[0],test_a,test_sdot)
        render_state(env,true_params,test_s)


if __name__ == "__main__":
    load_data = sio.loadmat('/home/qinjielin/RL_Ws/Pontryagin-Differentiable-Programming/Examples/SysID/cartpole/data/cartpole_iodata.mat')
    data = load_data['cartpole_iodata'][0, 0]
    actions_data = data[0]
    states_data = data[1]
    true_params = data[2][0]
    sigma =2 

    #initialize the environment
    env = cartpole()
    grad_loss = jax.jacfwd(env.loss_fn,argnums=0)
    grad_loss_traj = jax.jacfwd(env.loss_fn_traj,argnums=0)

    #initialize the parameter
    params = jnp.array(true_params+ sigma * np.random.rand(len(true_params)) - sigma / 2)#cart mass, pole mas, pole length
    loss_list = []
    params_list = []
    batch_size = 5
    test_s,test_a,test_sdot = jnp.array(states_data[0][0:-1]),jnp.reshape(jnp.array(actions_data[0]),(20,)),jnp.array(states_data[0][1:])

    for i in range(10000):
        traj_index = i%3#data only has 3 trajectories
        states = jnp.array(states_data[traj_index][0:-1])
        actions =  jnp.reshape(jnp.array(actions_data[traj_index]),(states.shape[0],))
        next_states = jnp.array(states_data[traj_index][1:])
        
        grad = grad_loss(params,states,actions,next_states)
        loss = env.loss_fn(params,test_s,test_a,test_sdot)
        params -= 0.01 * grad
        loss_list.append(loss)
        params_list.append(params)

        if(i%100==0):
            print("step:",i,"loss:",loss,"params:",params)

    plt.plot(loss_list)
    plt.show()
