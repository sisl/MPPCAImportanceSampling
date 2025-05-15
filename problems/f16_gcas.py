# packages
from copy import copy
from functools import partial
import numpy as np
from numpy import deg2rad
import torch
from tqdm import tqdm
from typing import Tuple

# jax imports
import jax
import jax.numpy as jnp
from jax_f16.f16_types import S
from jax_f16.f16_utils import f16state
from jax_f16.highlevel.controlled_f16 import controlled_f16

# file imports
from problems import Problem


class F16GCAS(Problem):
    def __init__(self, 
                 T:int=100, 
                 inner_steps:int=30, 
                 vt0:float=500.,
                 alpha0:float=0.037027,
                 alpha_failure:float=0.4363323129985824,
                 beta0:float=0.,
                 phi0:float=0.0,
                 theta0:float=-0.5235987755982988,
                 psi0:float=0.,
                 p0:float=0.,
                 q0:float=0.,
                 r0:float=0.,
                 alt0:float=950.,
                 power0:float=9.,
                 ) -> None:
        """
        Initializes the F16 simulation parameters.
        Args:
            T (int): Total simulation steps. Default is 100.
            inner_steps (int): Number of inner steps per simulation step. Default is 20.
            vt0 (float): Initial true airspeed. Default is 540.
            alpha0 (float): Initial angle of attack in radians. Default is 0.037027.
            alpha_failure (float): Angle of attack at which failure occurs in radians. Default is 0.349065.
            beta0 (float): Initial sideslip angle in radians. Default is 0.
            phi0 (float): Initial roll angle in radians. Default is 0.
            theta0 (float): Initial pitch angle in radians. Default is -0.471238898.
            psi0 (float): Initial yaw angle in radians. Default is 0.
            p0 (float): Initial roll rate in radians per second. Default is 0.
            q0 (float): Initial pitch rate in radians per second. Default is 0.
            r0 (float): Initial yaw rate in radians per second. Default is 0.
            alt0 (float): Initial altitude. Default is 950.
            power0 (float): Initial power setting. Default is 9.
        Returns:
            None
        """
    
        self.name = "f16gcas"
        self.T = T
        self.inner_steps = inner_steps
        self.vt0 = vt0
        self.alpha0 = alpha0
        self.alpha_failure = alpha_failure
        self.beta0 = beta0
        self.phi0 = phi0
        self.theta0 = theta0
        self.psi0 = psi0
        self.p0 = p0
        self.q0 = q0
        self.r0 = r0
        self.alt0 = alt0
        self.power0 = power0
        
        self.trajectory_variables = [S.ALPHA, S.PHI, S.THETA, S.ALT]
        self.disturbance_variables = [S.PHI, S.THETA]
        self.scale = torch.Tensor([0.01,0.01]).reshape(1,1,len(self.disturbance_variables))
        
    def obj_function(self, samples:torch.Tensor, return_trajectories=False) -> torch.Tensor:
        """
        Evaluates the objective function for the f16gcas problem. This function 
        takes an input tensor of samples (sensor disturbances), simulates the F-16
        dynamics, and computes the function values based on the resulting trajectories.
        Args:
            samples (torch.Tensor): (n, d) tensor of input samples (sensor disturbances).
        Returns:
            f_evals (torch.Tensor): (n,) tensor of objective function evaluations.
        """

        # extract disturbance samples
        disturbance_samples = samples[:,2:].reshape(-1, self.T, len(self.disturbance_variables))
        disturbance_samples = disturbance_samples * self.scale
        disturbance_samples = torch.cumsum(disturbance_samples,dim=1) # Markov chain noise, simulates sensor drift
        
        # extract initial state noise
        phi_noise = samples[:,0]
        theta_noise = samples[:,1]
        
        trajectories = self._simulate(disturbance_samples, phi_noise, theta_noise)
        f_evals = self._obj_function_wo_simulation(trajectories)
        
        if return_trajectories:
            return f_evals, trajectories
        else:
            return f_evals

    def _obj_function_wo_simulation(self, trajectories:torch.Tensor) -> torch.Tensor:
        """
        Function to evaluate the objective value based on the F-16 trajectories (not sensor disturbances).
        Args:
            samples (torch.Tensor): (n, d) tensor of aircraft trajectories.
        Returns:
            f_evals (torch.Tensor): (n,) tensor of objective function evaluations.
        """
        
        trajectories = trajectories.reshape(trajectories.shape[0],-1,len(self.trajectory_variables))
        h_tilde = 1/950 * trajectories[:,:,3].min(dim=1).values
        alpha_tilde = 1/self.alpha_failure * ((self.alpha_failure-trajectories[:,:,0]).min(dim=1).values)
        
        f_evals = torch.minimum(h_tilde, alpha_tilde)

        return f_evals
    
    def _simulate(self, disturbances:torch.Tensor, phi_noise:torch.Tensor, theta_noise:torch.Tensor) -> torch.Tensor:
        """
        Simulates the F-16 dynamics under the GCAS controller.
        Args:
            disturbances (torch.Tensor): (n, d) tensor of sensor drift disturbances.
            phi_noise (torch.Tensor): (n,) tensor of initial roll angles.
            theta_noise (torch.Tensor): (n,) tensor of initial pitch angles.
        Returns:
            trajectories (torch.Tensor): (n, d) tensor of aircraft trajectories.
        """

        disturbances = jax.numpy.array(disturbances.numpy())  # convert to jax
        
        trajectories = []

        for i in tqdm(range(disturbances.shape[0]), desc="simulating dynamics"):
            vt = self.vt0
            alpha = self.alpha0
            phi = self.phi0 + 0.15 * phi_noise[i].item()
            theta = self.theta0 + 0.05 * theta_noise[i].item() 
            alt = self.alt0

            f16_state = f16state(
                float(vt), 
                [float(alpha), float(self.beta0)], 
                [float(phi), float(theta), float(self.psi0)], 
                [float(self.p0), float(self.q0), float(self.r0)], 
                [0, 0, float(alt)], float(self.power0), 
                [0, 0, 0]
                )

            xs = sim_gcas(f16_state, T=self.T, inner_steps=self.inner_steps, disturbances=disturbances[i])
            
            trajectories.append(xs[:,self.trajectory_variables])

        trajectories = torch.Tensor(copy(jax.device_get(jax.numpy.stack(trajectories,axis=0))))
        
        return trajectories
    
    def sample(self, n:int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:        
        """
        Generates n samples from a standard normal distribution and evaluates the objective function.
        Args:
            n (int): The number of samples to generate.
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - all_noise (torch.Tensor): (n, d) tensor of disturbances.
                - f_evals (torch.Tensor): (n,) tensor of objective function evaluations.
                - trajectories (torch.Tensor): (n, d) tensor of aircraft trajectories.
        """

        sensor_noise = torch.randn((n,self.T,len(self.disturbance_variables))) 
        sensor_noise_scaled = sensor_noise* self.scale
        sensor_noise_cumsum = torch.cumsum(sensor_noise_scaled,dim=1) # Markov chain noise, simulates sensor drift
        
        # initial state noise
        phi_noise = torch.randn(n)
        theta_noise = torch.randn(n)

        trajectories = self._simulate(sensor_noise_cumsum, phi_noise, theta_noise)
        
        f_evals = self._obj_function_wo_simulation(trajectories)
        
        # prepend initial state noise
        all_noise = torch.concatenate([phi_noise.reshape(n,1), theta_noise.reshape(n,1), sensor_noise.reshape(n,-1)],dim=1)
        
        return all_noise, f_evals, trajectories
    
    def plot(self, ax, failure_trajectories, non_failure_trajectories, n_failures:int=50, n_non_failures:int=150):
        """Plot candidate failure and non-failure trajectories."""

        # helper function to cut off parts of the failure trajectories that are in the failure regions to declutter the plot
        def trim_failure_trajectory(failure_trajectory):
            for i in range(1,self.T+1):
                lfe = self._obj_function_wo_simulation(failure_trajectory[:i].reshape(1,i,len(self.trajectory_variables)))
                if lfe <= 0:
                    i_eff = int(torch.minimum(torch.Tensor([failure_trajectory.shape[0]]),torch.Tensor([i+2])).item())
                    return failure_trajectory[:i_eff].reshape(i_eff,len(self.trajectory_variables))
            
            raise ValueError("No failure found in trajectory despite negative objective function.")
        
        assert failure_trajectories.shape[0] >= n_failures
        assert non_failure_trajectories.shape[0] >= n_non_failures
        
        # plot failures and non-failures
        for i in range(n_non_failures):
            ax.plot(non_failure_trajectories[i,:,0], 
                    non_failure_trajectories[i,:,3], 
                    linewidth=1, alpha=0.1, color=(0.5,0.5,0.5))
        
        for i in range(n_failures):
            trimmed_failure_trajectory = trim_failure_trajectory(failure_trajectories[i])
            ax.plot(trimmed_failure_trajectory[:,0], 
                    trimmed_failure_trajectory[:,3], 
                    linewidth=1, alpha=0.3, color=(0.8,0,0))
        
        ax.axvspan(self.alpha_failure, self.alpha_failure*1.15, color=(1,0.9,0.9), alpha=1.0)
        ax.axhspan(-0.15*self.alt0, 0, color=(1,0.9,0.9), alpha=1.0)
        
        ax.set_xlim(0,self.alpha_failure*1.15)
        ax.set_ylim(-0.15*self.alt0,self.alt0*1.1)
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(r"$h$",labelpad=1)
        
        
#########################################################################
######## THE FOLLOWING CODE SHOULD NOT NEED ANY MODIFICATIONS ###########
#########################################################################

class GCAS:
    STANDBY = 0
    ROLL = 1
    PULL = 2
    WAITING = 3


class GcasAutopilot:
    """ground collision avoidance autopilot"""

    def __init__(self, init_mode=0, stdout=False):

        assert init_mode in range(4)

        # config
        self.cfg_eps_phi = deg2rad(5)       # max abs roll angle before pull
        self.cfg_eps_p = deg2rad(10)        # max abs roll rate before pull
        self.cfg_path_goal = deg2rad(0)     # min path angle before completion
        self.cfg_k_prop = 4                 # proportional control gain
        self.cfg_k_der = 2                  # derivative control gain
        self.cfg_flight_deck = 1000         # altitude at which GCAS activates
        self.cfg_min_pull_time = 2          # min duration of pull up

        # self.cfg_nz_des = 5
        self.cfg_nz_des = 5.1

        self.pull_start_time = 0
        self.stdout = stdout

        self.waiting_cmd = jnp.zeros(4)
        self.waiting_time = 2
        self.mode = init_mode

    def log(self, s):
        "print to terminal if stdout is true"

        if self.stdout:
            print(s)

    def are_wings_level(self, x_f16):
        "are the wings level?"

        phi = x_f16[S.PHI]

        radsFromWingsLevel = jnp.round(phi / (2 * jnp.pi))

        return jnp.abs(phi - (2 * jnp.pi)  * radsFromWingsLevel) < self.cfg_eps_phi

    def is_roll_rate_low(self, x_f16):
        "is the roll rate low enough to switch to pull?"

        p = x_f16[S.P]

        return abs(p) < self.cfg_eps_p

    def is_above_flight_deck(self, x_f16):
        "is the aircraft above the flight deck?"

        alt = x_f16[S.ALT]

        return alt >= self.cfg_flight_deck

    def is_nose_high_enough(self, x_f16):
        "is the nose high enough?"

        theta = x_f16[S.THETA]
        alpha = x_f16[S.ALPHA]
        # Determine which angle is "level" (0, 360, 720, etc)
        radsFromNoseLevel = jnp.round((theta-alpha)/(2 * jnp.pi))
        # Evaluate boolean
        return ((theta-alpha) - 2 * jnp.pi * radsFromNoseLevel) > self.cfg_path_goal

    def get_u_ref(self, x_f16):
        """get the reference input signals"""
        def roll_or_pull():
            roll_condition = jnp.logical_and(self.is_roll_rate_low(x_f16), self.are_wings_level(x_f16))
            return jax.lax.cond(roll_condition, lambda _: self.pull_nose_level(), lambda _: self.roll_wings_level(x_f16), None)

        def standby_or_roll():
            standby_condition = jnp.logical_and(jnp.logical_not(self.is_nose_high_enough(x_f16)), jnp.logical_not(self.is_above_flight_deck(x_f16)))
            return jax.lax.cond(standby_condition, lambda _: roll_or_pull(), lambda _: jnp.zeros(4), None)

        pull_condition = jnp.logical_and(self.is_nose_high_enough(x_f16), True)
        return jax.lax.cond(pull_condition, lambda _: jnp.zeros(4), lambda _: standby_or_roll(), None)
    

    def get_u_ref_orig(self, _t, x_f16):
        """get the reference input signals"""

        if self.mode == "waiting":
            # time-triggered start after two seconds
            if _t + 1e-6 >= self.waiting_time:
                self.mode = "roll"
        elif self.mode == "standby":
            if not self.is_nose_high_enough(x_f16) and not self.is_above_flight_deck(x_f16):
                self.mode = "roll"
        elif self.mode == "roll":
            if self.is_roll_rate_low(x_f16) and self.are_wings_level(x_f16):
                self.mode = "pull"
                self.pull_start_time = _t
        else:
            assert self.mode == "pull", f"unknown mode: {self.mode}"

            if self.is_nose_high_enough(x_f16) and _t >= self.pull_start_time + self.cfg_min_pull_time:
                self.mode = "standby"

        if self.mode == "standby":
            rv = np.zeros(4)
        elif self.mode == "waiting":
            rv = self.waiting_cmd
        elif self.mode == "roll":
            rv = self.roll_wings_level(x_f16)
        else:
            assert self.mode == "pull", f"unknown mode: {self.mode}"
            rv = self.pull_nose_level()

        return rv

    def pull_nose_level(self):
        "get commands in mode PULL"
        rv = jnp.array([self.cfg_nz_des, 0.0, 0.0, 0.0]) 

        return rv

    def roll_wings_level(self, x_f16):
        "get commands in mode ROLL"

        phi = x_f16[S.PHI]
        p = x_f16[S.P]
        # Determine which angle is "level" (0, 360, 720, etc)
        radsFromWingsLevel = jnp.round(phi / (2 * jnp.pi))
        # PD Control until phi == pi * radsFromWingsLevel
        ps = -(phi - (2 * jnp.pi) * radsFromWingsLevel) * self.cfg_k_prop - p * self.cfg_k_der
        # Build commands to roll wings level
        rv = jnp.array([0.0, ps, 0.0, 0.0])

        return rv


def inner_step(autopilot, x, dt, dist, inner_steps=10):
    for _ in range(inner_steps):
        #x_obs is the observed  state that is used by GCAS to output control signal
        x_obs = x
        x_obs = x_obs.at[S.PHI].add(dist[0])
        x_obs = x_obs.at[S.THETA].add(dist[1])
        u = autopilot.get_u_ref(x_obs)
        xdot = controlled_f16(x, u).xd
        x = x + xdot * dt
    return x


@partial(jax.jit, static_argnames=["T", "dt", "inner_steps"])
def sim_gcas(
        f16_state,
        T=150,
        dt=1/500,
        inner_steps=10,
        disturbances=None
        ):
    
    ap = GcasAutopilot()
    
    x = f16_state
    alts = jnp.zeros(T)

    def body_fun(carry, i):
        alts, x = carry
        alts = alts.at[i].set(x[S.ALT])
        x = inner_step(ap, x, dt, disturbances[i], inner_steps=inner_steps)
        return (alts, x), x
    
    (alts, x), xs = jax.lax.scan(body_fun, (alts, x), jnp.arange(T))

    return xs
    
    
if __name__ == "__main__":
    n = 10000
    T = 100
    
    f16 = F16GCAS(T=T)  
    
    samples, lim_funs, samples_plot = f16.sample(n)
    f16.plot(samples, n_failures=2, n_non_failures=10)
    
    test = f16.obj_function(samples)
    
    assert torch.allclose(lim_funs, test)   # guarantees that there is no randomness in the objective function
    assert samples.shape == torch.Size((n,2*T))
    assert lim_funs.shape == torch.Size((n,))
    assert samples_plot.shape == torch.Size((n,T,4))
    assert isinstance(samples,torch.Tensor)
    assert isinstance(lim_funs,torch.Tensor)
    assert isinstance(samples_plot,torch.Tensor)
    
        