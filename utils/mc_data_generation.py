# packages
import os
import sys
from tqdm import tqdm
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from problems import Problem


class MCDataGeneration():
    """
    Generate Monte Carlo datasets for the three systems.
        
    Args:
        problem (Problem): problem class (Branches, Oscillator, F16GCAS).
        n (int): number of Monte Carlos samples to draw.
        batch_size (int): batch size.
        seed (int): random seed for reproducibility.
        datapath (str): path to save the datasets in. 
    """

    def __init__(self, problem:Problem, n:int=5000, batch_size:int=1000, seed=42, datapath:str="./mc_data/"):
        self.problem = problem
        self.n = int(n)
        self.batch_size = int(batch_size)
        self.seed = seed
        self.data_root = datapath
        self.data_path = datapath+self.problem.name+"/"
    
    def build_dataset(self):      
        """Create Monte Carlo datasets by appending smaller batches."""
        samples_per_batch = [self.batch_size] * (self.n // self.batch_size) + ([self.n % self.batch_size] if (self.n % self.batch_size) != 0 else [])
        print(f"Building MC dataset for problem {self.problem.name}. {len(samples_per_batch)} batches with batch size {self.batch_size}")
        
        # setup the directories
        os.makedirs(self.data_root, exist_ok=True)
        os.makedirs(self.data_path, exist_ok=True)
        
        samples = []
        obj_funcs = []
        plotting_samples = []
        
        # set seed
        torch.manual_seed(self.seed)
        
        for sb in tqdm(samples_per_batch):
            s, of, ps = self.problem.sample(sb)
            
            samples.append(s)
            obj_funcs.append(of)
            plotting_samples.append(ps)
        
        samples = torch.concatenate(samples, dim=0)
        obj_funcs = torch.concatenate(obj_funcs, dim=0)
        plotting_samples = torch.concatenate(plotting_samples, dim=0)
        
        torch.save(samples, self.data_path+"samples.pt")
        torch.save(obj_funcs, self.data_path+"obj_function_evaluations.pt")
        torch.save(plotting_samples, self.data_path+"trajectories.pt")
        
        print(f"Saved MC dataset in {self.data_path}")
        
        return samples, obj_funcs, plotting_samples
          
        
if __name__ == "__main__":
    from problems import Branches
    from problems import Oscillator
    from problems import F16GCAS
    import argparse
    
    # Argument parser
    parser = argparse.ArgumentParser(description='Generate Monte Carlo datasets for different systems.')
    parser.add_argument('--problem', type=str, required=True, choices=['Branches', 'Oscillator', 'F16GCAS'], help='The problem to generate data for.')
    parser.add_argument('--d', type=int, default=40, help='Dimension parameter for the problem (if applicable).')
    parser.add_argument('--n', type=int, default=500000, help='Number of Monte Carlo samples to draw.')
    parser.add_argument('--batch_size', type=int, default=10000, help='Batch size.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--datapath', type=str, default='./mc_data/', help='Path to save the datasets in.')

    args = parser.parse_args()

    # Select problem class based on argument
    if args.problem == 'Branches':
        problem = Branches(d=args.d)
    elif args.problem == 'Oscillator':
        problem = Oscillator(d=args.d)
    elif args.problem == 'F16GCAS':
        problem = F16GCAS()

    # Generate dataset
    MCDataGeneration(problem, n=args.n, batch_size=args.batch_size, seed=args.seed, datapath=args.datapath).build_dataset()
    
