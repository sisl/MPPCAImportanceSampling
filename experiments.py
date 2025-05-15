#*******************************************************************************
# imports and setup
#*******************************************************************************
# packages
import argparse
import json
import numpy as np
import os
import torch
from torch.distributions import MultivariateNormal

# file imports
from is_methods.cross_entropy import cross_entropy_is
from is_methods.sequential import sequential_is

from models.gmm import GMM
from models.mppca import MPPCA

from utils.eval_metrics import metrics
from utils.ndb import NDB

from problems import Branches, Oscillator, F16GCAS


def save_metrics(dir, Pfs, rel_errs, log_probs, coverages, NDBs, total_samples, samples_per_level):
    metrics = {}
    metrics['Pfs'] = np.array(Pfs).tolist()
    metrics['Pfs mean'] = float(Pfs.mean())
    metrics['Pfs std'] = float(Pfs.std())
    metrics['rel errs'] =  np.array(rel_errs).tolist()
    metrics['rel errs mean'] = float(rel_errs.mean())
    metrics['rel errs std'] = float(rel_errs .std())
    metrics['log probs'] = np.array(log_probs).tolist()
    metrics['log probs mean'] = float(log_probs.mean())
    metrics['log probs std'] = float(log_probs.std())
    metrics['coverages'] = np.array(coverages).tolist()
    metrics['coverages mean'] = float(coverages.mean())
    metrics['coverages std'] = float(coverages.std())
    metrics['NDBs'] = np.array(NDBs).tolist()
    metrics['NDBs mean'] = float(NDBs.mean())
    metrics['NDBs std'] = float(NDBs.std())
    metrics['total samples'] = np.array(total_samples).tolist()
    metrics['total samples mean'] = float(total_samples.mean())
    metrics['total samples std'] = float(total_samples.std())
    metrics['samples_per_level'] = float(samples_per_level)

    # save results
    with open(os.path.join(dir, "results.json"), "w") as outfile:
        json.dump(metrics, outfile)


parser = argparse.ArgumentParser()
parser.add_argument("--problem", type=str, default="branches",
                    choices=["branches", "oscillator", "f16gcas"])
parser.add_argument("--is_method", type=str, default="CE",
                    choices=["CE", "SIS"])
parser.add_argument("--proposal", type=str, default="MPPCA",
                    choices=["MPPCA", "GMM"])
parser.add_argument("--n_features", type=int, default=40)
parser.add_argument("--n_components", type=int, default=8)
parser.add_argument("--n_factors", type=int, default=8)
parser.add_argument("--n_samples", type=int, default=10000)
parser.add_argument("--n_samples_fail_metrics", type=int, default=1000)
parser.add_argument("--n_max_attempt_fail_sample", type=int, default=25)
parser.add_argument("--rho", type=int, default=0.2)
parser.add_argument("--n_trials", type=int, default=3)
args = parser.parse_args()

# set up the problem
if args.problem == "branches":
    assert args.n_features in [40, 60]
    problem = Branches(d=args.n_features)
elif args.problem == "oscillator":
    assert args.n_features in [100, 200]
    problem = Oscillator(d=args.n_features)
elif args.problem == "f16gcas":
    assert args.n_features == 202
    problem = F16GCAS()
else:
    raise ValueError

results_dir = f"./results/{problem.name}/{args.is_method}-{args.proposal}"
os.makedirs(results_dir, exist_ok=True)
obj_func = problem.obj_function
real_failures, _, _, _, ref_Pf = problem.load_mc_dataset(return_only_samples=True)


#*******************************************************************************
# perform importance sampling
#*******************************************************************************
print("\n****************************************")
print(f"{args.is_method} with {args.proposal} proposals")
print(f"System: {args.problem} with {args.n_features} dimensions")
print("Ref Pf:\t {:.6f}".format(ref_Pf))
print("****************************************\n")

all_Pfs = torch.zeros(args.n_trials)
all_rel_errs = torch.zeros(args.n_trials)
all_log_probs = torch.zeros(args.n_trials)
all_coverages = torch.zeros(args.n_trials)
all_NDBs = torch.zeros(args.n_trials)
all_total_samples = torch.zeros(args.n_trials)

for i in range(args.n_trials):
    # instantiate proposal density
    if args.proposal == "MPPCA":
        model = MPPCA(
            n_components=args.n_components,
            n_features=args.n_features,
            n_factors=args.n_factors
        )
    elif args.proposal == "GMM":
        model = GMM(
            n_components=args.n_components,
            n_features=args.n_features
        )
    else:
        raise ValueError

    torch.manual_seed(i)
    # importance sampling
    print('Trial {}'.format(i+1))
    if args.is_method == 'CE':
        [Pf, model, total_samples] = \
            cross_entropy_is(args.n_samples, args.rho, obj_func, model)
    elif args.is_method == 'SIS':
        [Pf, model, total_samples] = \
            sequential_is(args.n_samples, args.rho, obj_func, model)
    
    # sample from the learned proposal
    fail_sample_attempt_counter = 0
    q_fails = torch.empty((0,args.n_features))
    while (q_fails.shape[0] < args.n_samples_fail_metrics) and (fail_sample_attempt_counter < args.n_max_attempt_fail_sample):
        print(f"Finding failure samples attempt {fail_sample_attempt_counter+1}/{args.n_max_attempt_fail_sample}")
        q_samples = model.sample(args.n_samples)
        q_lfe = problem.obj_function(q_samples)
        q_fails = torch.concatenate([q_fails, q_samples[q_lfe<=0]], dim=0)
        fail_sample_attempt_counter += 1
        print("more sampling", q_fails.shape[0], fail_sample_attempt_counter)
    
    # save samples
    torch.save(q_samples, os.path.join(results_dir, "samples.pt"))
              
    if q_fails.shape[0] < args.n_samples_fail_metrics:
        # too few failures were found 
        print("Not enough failure samples were found to compute meaningful metrics.")
        all_Pfs[i] = Pf
        all_rel_errs[i] = (Pf - ref_Pf) / ref_Pf
        all_log_probs[i] = torch.nan
        all_coverages[i] = torch.nan
        all_NDBs[i] = torch.nan
        all_total_samples[i] = total_samples
    
    else:
        # compute metrics
        print("Computing metrics...")
        # log probs
        log_probs = MultivariateNormal(torch.zeros(args.n_features), torch.eye(args.n_features)).log_prob(q_fails)
        # coverage metric
        prdc_metrics = metrics(real_failures, q_fails, n_sample=25, n_max=args.n_samples_fail_metrics, nearest_k=5, print_results=False)
        # NDB
        number_of_bins = 40
        ndb_obj = NDB(training_data=real_failures, number_of_bins=number_of_bins, significance_level=0.05, whitening=False)
        results = ndb_obj.evaluate(q_fails, 'Validation')

        all_Pfs[i] = Pf
        all_rel_errs[i] = (Pf - ref_Pf) / ref_Pf
        all_log_probs[i] = log_probs.mean()
        all_coverages[i] = prdc_metrics['coverage_mean']
        all_NDBs[i] = results["NDB"]/number_of_bins 
        all_total_samples[i] = total_samples

    save_metrics(results_dir, all_Pfs, all_rel_errs, all_log_probs, all_coverages, all_NDBs, all_total_samples, args.n_samples)

print("\n----------------------------------------")
print("Avg Failure Prob: {:.6f}".format(all_Pfs.mean()))
print("Avg Log Prob: {:.2f}".format(all_log_probs.mean()))
print("Avg Coverage: {:.4f}".format(all_coverages.mean()))
print("Avg NDB: {:.4f}".format(all_NDBs.mean()))
print("Avg Samples: {:.1f}".format(all_total_samples.mean()))
print("----------------------------------------\n")