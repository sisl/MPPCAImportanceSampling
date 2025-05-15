# packages
import glob
from pathlib import Path
from scipy.stats import beta
import torch
from typing import Tuple


class Problem():
    def __repr__(self):
        return self.name
    
    def sample(self, n:int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError("The sample method has not been implemented for this problem.")
    
    def obj_function(self, samples:torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("The objective function method has not been implemented for this problem.")
    
    def load_mc_dataset(self, data_root:str="./mc_data/", return_only_samples:bool=False):
        self.data_root = data_root
        data_path = self.data_root+self.name+"/"
        
        # check what files are available
        files = glob.glob(data_path+"*.pt")
        file_stems = [Path(f).stem for f in files]
        
        if "samples" not in file_stems:
            raise ValueError("No samples.pt found. You need at least samples.pt.")
        
        elif return_only_samples:
            samples = torch.load(data_path+"samples.pt")
            lfes = torch.load(data_path+"obj_function_evaluations.pt")

        elif "trajectories" not in file_stems:
            # need to simulate the system and evaluate the objective function values
            samples = torch.load(data_path+"samples.pt")
            lfes, trajectories = self.obj_function(samples, return_trajectories=True)
            
        elif "obj_function_evaluations" not in file_stems:
            # only need objective function evaluations
            samples = torch.load(data_path+"samples.pt")
            trajectories = torch.load(data_path+"trajectories.pt")
            lfes = self._obj_function_wo_simulation(trajectories)
        else:
            # samples, trajectories, and obj_function evaluations are available
            samples = torch.load(data_path+"samples.pt")
            trajectories = torch.load(data_path+"trajectories.pt")
            lfes = torch.load(data_path+"obj_function_evaluations.pt")
             
        
        failure_samples = samples[lfes <= 0]
        non_failure_samples = samples[lfes > 0]
        
        if not return_only_samples:
            failure_trajectories = trajectories[lfes <= 0]
            non_failure_trajectories = trajectories[lfes > 0]
        else:
            failure_trajectories = None
            non_failure_trajectories = None
        
        no_failures = failure_samples.shape[0]
        no_non_failures = samples.shape[0] - failure_samples.shape[0]
        pf_mean = beta(no_failures,no_non_failures).mean().item()
        
        return failure_samples, non_failure_samples, failure_trajectories, non_failure_trajectories, pf_mean
    
    def load_results(self, method, results_root:str="./results/"):
        self.results_root = results_root
        results_path = self.results_root+self.name+"/"+method+"/"
        
        # check what files are available
        files = glob.glob(results_path+"*.pt")
        file_stems = [Path(f).stem for f in files]
        
        if "samples" not in file_stems:
            raise ValueError("No samples.pt found. You need at least samples.pt.")
        
        elif "trajectories" not in file_stems:
            # need to simulate the system and evaluate the objective function values
            samples = torch.load(results_path+"samples.pt")
            lfes, trajectories = self.obj_function(samples, return_trajectories=True)
            
        elif "obj_function_evaluations" not in file_stems:
            # only need objective function evaluations
            samples = torch.load(results_path+"samples.pt")
            trajectories = torch.load(results_path+"trajectories.pt")
            lfes = self._obj_function_wo_simulation(trajectories)
        else:
            # samples, trajectories, and obj_function evaluations are available
            samples = torch.load(results_path+"samples.pt")
            trajectories = torch.load(results_path+"trajectories.pt")
            lfes = torch.load(results_path+"obj_function_evaluations.pt")
             
        
        failure_samples = samples[lfes <= 0]
        non_failure_samples = samples[lfes > 0]
        
        failure_trajectories = trajectories[lfes <= 0]
        non_failure_trajectories = trajectories[lfes > 0]
        
        
        no_failures = failure_samples.shape[0]
        no_non_failures = samples.shape[0] - failure_samples.shape[0]
        pf_mean = beta(no_failures,no_non_failures).mean().item()
        
        return failure_samples, non_failure_samples, failure_trajectories, non_failure_trajectories, pf_mean
          