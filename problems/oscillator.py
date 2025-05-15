# packages
from matplotlib.patches import Rectangle
import torch
from typing import Tuple

# file imports
from problems import Problem


class Oscillator(Problem):
    def __init__(self, d:int=200, t_max:float=2.0, dt:float=0.004, z0:float=0.0, 
                 x0:float=1.5, m:float=1000., c:float=628.3185307, 
                 k:float=39478.417604, gamma:float=1.0, S:float=0.005):
        """
        Initializes the parameters for the Duffing oscillator model. Code is 
        inspired by https://github.com/Julien6431/Importance-Sampling-VAE
        Args:
            d (int): The input dimension, default is 200.
            t_max (float): The maximum time for the simulation, default is 2.0.
            dt (float): The time step for the simulation, default is 0.004.
            z0 (float): The initial velocity, default is 0.0.
            x0 (float): The initial displacement, default is 1.5.
            m (float): The mass of the oscillator, default is 1000.0.
            c (float): The damping coefficient, default is 628.3185307.
            k (float): The stiffness coefficient, default is 39478.417604.
            gamma (float): The nonlinearity parameter, default is 1.0.
            S (float): The spectral density, default is 0.005.
        Attributes:
            d (int): The input dimension.
            z0 (float): The initial velocity.
            x0 (float): The initial displacement.
            m (float): The mass of the oscillator.
            c (float): The damping coefficient.
            k (float): The stiffness coefficient.
            gamma (float): The nonlinearity parameter.
            del_w (float): The frequency increment.
            S (float): The spectral density.
            sigma (torch.Tensor): The standard deviation of the noise.
            t_max (float): The maximum time for the simulation.
            dt (float): The time step for the simulation.
            t_k (torch.Tensor): The time steps for the simulation.
        """
        
        self.d = d
        self.z0 = z0 
        self.x0 = x0
        self.m = m
        self.c = c
        self.k = k
        self.gamma = gamma
        self.del_w  = 30*torch.pi/self.d
        self.S = S
        self.sigma = torch.sqrt(torch.tensor(2*S*self.del_w))
        self.t_max = t_max  
        self.dt = dt
        self.t_k = torch.arange(0, self.t_max, self.dt)
        
        self.name = f"oscillator-{self.d}"
    
    def obj_function(self, samples:torch.Tensor, return_trajectories=False) -> torch.Tensor:
        """
        Evaluates the objective function for the oscillator problem. This function
        takes a input tensor of samples (spectral coefficients), performs Euler's 
        method to compute the oscillator trajectories, and computes the function 
        values based on the resulting displacements.
        Args:
            samples (torch.Tensor): (n, d) tensor of input samples (spectral coefficients).
        Returns:
            f_evals (torch.Tensor): (n,) tensor of objective function evaluations.
        """
        
        trajectories = self._simulate(samples)
        f_evals = self._obj_function_wo_simulation(trajectories)

        if return_trajectories:
            return f_evals, trajectories
        else:      
            return f_evals
        
    def _obj_function_wo_simulation(self, trajectories:torch.Tensor) -> torch.Tensor:
        """
        Evaluate the objective function based on the oscillator displacement (not spectral coefficients).
        Args:
            trajectories (torch.Tensor): (n, d) tensor of oscillator displacements.
        Returns:
            f_evals (torch.Tensor): (n,) tensor of objective function evaluations.
        """

        final_displacement = trajectories[:,-1]
        n = trajectories.shape[0]
        matrice = torch.zeros((n,2))
        matrice[:,0] = 0.1*torch.ones(n) - final_displacement
        matrice[:,1] = final_displacement + 0.05*torch.ones(n)
        
        f_evals = torch.min(matrice, dim=1)[0]

        return f_evals
    
    def _simulate(self, coefficients:torch.Tensor) -> torch.Tensor:
        """
        Simulates the Duffing oscillator dynamics.
        Args:
            coefficients (torch.Tensor): (n, d) tensor of input spectral coefficients.
        Returns:
            trajectories (torch.Tensor): (n, d) tensor of oscillator displacements.
        """

        def f_t2(t, U):
            d_2 = int(self.d/2)
            N = U.shape[0]
            res = torch.zeros(N)

            for i in range(d_2):
                w_i = i*self.del_w
                res = res + U[:,i]*torch.cos(w_i*t) + U[:,i + d_2]*torch.sin(w_i*t)
                
            return -self.m*self.sigma*res

        def F_1_2(i,X):
            return X[:,i]

        def F_2_2(i,Z,X,U):
            return (-self.c*X[:,i] - self.k*(Z[:,i] + self.gamma*(Z[:,i]**3)) + f_t2(self.t_k[i], U))/self.m
        
        n = coefficients.shape[0]
        len_t = len(self.t_k)
        trajectories = torch.zeros((n,len_t))
        X = torch.zeros((n,len_t))
        trajectories[:,0] = self.z0*torch.ones(n)
        X[:,0] = self.x0*torch.ones(n)
        # Euler's method to simulate oscillator displacement
        for i in range(len_t-1):
            trajectories[:,i+1] = trajectories[:,i] + self.dt*F_1_2(i,X)
            X[:,i+1] = X[:,i] + self.dt*F_2_2(i, trajectories, X, coefficients)
        
        return trajectories

    def sample(self, n:int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generates n samples from a standard normal distribution and evaluates the objective function.
        Args:
            n (int): The number of samples to generate.
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - coefficients (torch.Tensor): (n, d) tensor containing the generated spectral coefficients.
                - f_evals (torch.Tensor): (n,) tensor of objective function evaluations.
                - trajectories (torch.Tensor): (n, d) tensor oscillator displacements.
        """
        
        coefficients = torch.randn((n, self.d))
        trajectories = self._simulate(coefficients)
        f_evals = self._obj_function_wo_simulation(trajectories)
        
        return coefficients, f_evals, trajectories
    
    def plot(self, ax, failure_trajectories, non_failure_trajectories, n_failures:int=50, n_non_failures:int=150):
        """Plot candidate failure and non-failure trajectories."""

        assert failure_trajectories.shape[0] >= n_failures
        assert non_failure_trajectories.shape[0] >= n_non_failures
        
        # plot failures and non-failures
        t = torch.arange(non_failure_trajectories.shape[1])*self.dt
        for i in range(n_failures):
            ax.plot(t, failure_trajectories[i], linewidth=1, alpha=0.3, color=(0.8,0,0))
            
        for i in range(n_non_failures):
            ax.plot(t, non_failure_trajectories[i], linewidth=1, alpha=0.1, color=(0.5,0.5,0.5))
        
        ax.add_patch(Rectangle((t[-1], 0.1),1,2,linewidth=0, color=(1,0.9,0.9), alpha=1.0))
        ax.add_patch(Rectangle((t[-1], -0.05-2),1,2,linewidth=0, color=(1,0.9,0.9), alpha=1.0))
        
        ax.set_xlim(-0.02*self.t_max, 1.12*self.t_max)
        ax.set_ylim(-0.3,0.3)
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$x$",labelpad=1)
        
           
if __name__ == "__main__":
    d = 200
    n = 1000

    oscillator = Oscillator(input_dim=d)
    coefficients, f_evals, trajectories = oscillator.sample(n)
    
    assert coefficients.shape == torch.Size((n,d))
    assert f_evals.shape == torch.Size((n,))
    assert trajectories.shape == torch.Size((n,500))
    assert isinstance(coefficients,torch.Tensor)
    assert isinstance(f_evals,torch.Tensor)
    assert isinstance(trajectories,torch.Tensor)

    