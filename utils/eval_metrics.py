# packages
import torch
from prdc import compute_prdc
from typing import Optional
from contextlib import redirect_stdout, redirect_stderr
import os


def metrics(mc_failures:torch.Tensor, model_failures:torch.Tensor, n_max: Optional[int]=None, n_sample:int=25, nearest_k:int=5, print_results:bool=True):
    """
    Compute coverage metrics for model failures compared to Monte Carlo (MC) failures.
    Args::
        mc_failures (torch.Tensor): (n1, d) tensor containing features of MC failures.
        model_failures (torch.Tensor): (n2, d) tensor containing features of model failures.
        n_max (int, optional): Maximum number of samples to use. If None, it defaults to 
            the minimum of the number of MC failures and model failures. Defaults to None.
        n_sample (int, optional): Number of samples to draw for computing metrics. Defaults to 100.
        nearest_k (int, optional): Number of nearest neighbors to use in the PRDC computation. Defaults to 5.
        print_results (bool, optional): Whether to print the results. Defaults to True.
    
    Returns:
        dict: A dictionary containing the mean and standard deviation of the coverage metric:
            - "coverage_mean" (torch.Tensor): Mean coverage.
            - "coverage_std" (torch.Tensor): Standard deviation of coverage.
    """
    
    def sample(X,n):
        assert n<=X.shape[0]
        
        return X[torch.randint(0,X.shape[0],(n,))]
    
    assert mc_failures.shape[1] == model_failures.shape[1]
    
    if n_max is None:
        n_max = min(mc_failures.shape[0],model_failures.shape[0])
        print(n_max)
    
    else:
        assert n_max <= min(mc_failures.shape[0],model_failures.shape[0])
    
    with open(os.devnull, 'w') as fnull:
        # this supresses the output of the compute_prdc function
        with redirect_stdout(fnull), redirect_stderr(fnull):     
            results = [compute_prdc(real_features=sample(mc_failures,n_max), fake_features=sample(model_failures,n_max), nearest_k=nearest_k) for _ in range(n_sample)]
    
    coverages = torch.Tensor([r["coverage"] for r in results])
        
    coverage_mean = torch.mean(coverages)
    coverage_std = torch.std(coverages)
    
    if print_results:
        print(f"Coverage:\t{coverage_mean:.4f} ± {coverage_std:.4f}")
    
    return {"coverage_mean": coverage_mean, "coverage_std": coverage_std}


if __name__ == "__main__":
    x1 = torch.randn((500,50))
    x2 = torch.randn((700,50))
    
    metrics(x1, x2, n_max=300)