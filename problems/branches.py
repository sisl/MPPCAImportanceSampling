# packages
from matplotlib.patches import Polygon
import torch
from typing import Tuple

# file imports
from problems import Problem


class Branches(Problem):
    def __init__(self, d:int=40) -> None:
        """
        Initializes the Branches class for a given dimensionality.
        Args:
            d (int, optional): Dimensionality of the branches problem. Defaults to 40.
        """
        
        self.d = d
        self.name = f"branches-{self.d}"
    
    def obj_function(self, samples:torch.Tensor, return_trajectories=False) -> torch.Tensor:
        """
        Evaluates the objective function for the branches problem.
        Args:
            samples (torch.Tensor): (n, d) tensor of input samples.
            return_trajectories (bool): dummy flag to return trajectories (no trajectories for branches problem).
        Returns:
            f_evals (torch.Tensor): (n,) tensor of objective function evaluations.
        """
        
        d = torch.tensor(samples.shape[1]) 
        side_1 = 3.5 + 1/torch.sqrt(d)*samples.sum(dim=1)
        side_2 = 3.5 + -1/torch.sqrt(d)*samples.sum(dim=1)
        side_3 = 3.5 + 1/torch.sqrt(d)*(samples[:, :d//2].sum(dim=1) - samples[:, d//2:].sum(dim=1))
        side_4 = 3.5 + 1/torch.sqrt(d)*(-samples[:, :d//2].sum(dim=1) + samples[:, d//2:].sum(dim=1))
        sides = torch.stack([side_1, side_2, side_3, side_4], dim=1)

        f_evals = torch.min(sides, dim=1)[0]

        if return_trajectories:
            return f_evals, samples
        else:
            return f_evals
    
    def sample(self, n:int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generates n samples from a standard normal distribution and evaluates the objective function.
        Args:
            n (int): The number of samples to generate.
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - samples (torch.Tensor): A tensor of shape (n, d) containing the generated samples.
                - f_evals (torch.Tensor): (n,) tensor of objective function evaluations.
                - samples (torch.Tensor): (n, d) tensor of "trajectories" (no trajectories for branches problem)
        """
        
        samples = torch.randn((n, self.d))
        f_evals = self.obj_function(samples)
        
        return samples, f_evals, samples
    
    def plot(self, ax, failure_trajectories, non_failure_trajectories, n_failures:int=150, n_non_failures:int=600):
        """Plot candidate failure and non-failure samples."""

        assert failure_trajectories.shape[0] >= n_failures
        assert non_failure_trajectories.shape[0] >= n_non_failures
        
        # shade failure regions
        alpha = 3.5*torch.sqrt(torch.Tensor([self.d])).item()   # boundary of failure region
        
        ax.add_patch(Polygon([[-2*alpha, -alpha],[alpha,2*alpha],[-2*alpha,2*alpha]],
                             linewidth=0, color=(1,0.9,0.9), alpha=1.0))
        ax.add_patch(Polygon([[alpha, -2*alpha],[-2*alpha,alpha],[-2*alpha,-2*alpha]],
                             linewidth=0, color=(1,0.9,0.9), alpha=1.0))
        ax.add_patch(Polygon([[2*alpha, alpha],[-alpha,-2*alpha],[2*alpha,-2*alpha]],
                             linewidth=0, color=(1,0.9,0.9), alpha=1.0))
        ax.add_patch(Polygon([[-alpha, 2*alpha],[2*alpha,-alpha],[2*alpha,2*alpha]],
                             linewidth=0, color=(1,0.9,0.9), alpha=1.0))
        
        # plot failures and non-failures
        for i in range(n_non_failures):
            ax.scatter(non_failure_trajectories[i,:int(self.d//2)].sum(), 
                       non_failure_trajectories[i,int(self.d//2):].sum(), 
                       s=3, alpha=0.3, color=(0.5,0.5,0.5))
        
        for i in range(n_failures):
            ax.scatter(failure_trajectories[i,:int(self.d//2)].sum(), 
                       failure_trajectories[i,int(self.d//2):].sum(), 
                       s=3, alpha=0.7, color=(0.8,0,0))
        
        ax.set_xlim(-1.2*alpha, 1.2*alpha)
        ax.set_ylim(-1.2*alpha, 1.2*alpha)
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$",labelpad=1)        
    
    
if __name__ == "__main__":
    d=100
    n=1000
    
    branches = Branches(d=d)
    samples, f_evals, samples_plot = branches.sample(n)
    
    assert samples.shape == torch.Size((n,d))
    assert f_evals.shape == torch.Size((n,))
    assert samples_plot.shape == torch.Size((n,d))
    assert isinstance(samples, torch.Tensor)
    assert isinstance(f_evals, torch.Tensor)
    assert isinstance(samples_plot, torch.Tensor)