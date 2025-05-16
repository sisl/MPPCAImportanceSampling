# MPPCA Importance Sampling

This repository contains code and experiments for the paper:

> Kruse, L. A., Schlichting, M. R., & Kochenderfer, M. J., _Scalable Importance Sampling in High Dimensions with Low-Rank Mixture Proposals_, CoDIT 2025.

## Dependencies

See `environment.yaml` for required packages, or use this to create a Conda environment with all dependencies:
```bash
conda env create -f environment.yaml
```

This repository was tested with Python 3.10 and PyTorch 2.7.

## End-to-end Experiments

Run the `experiments.sh` file to reproduce all experimental results. This script kicks off the following steps:
1. Generate Monte Carlo datasets for reference data and metric calculations.
2. Perform importance sampling for all five problems (`branches` with 40 and 60 dimensions, `oscillator` with 100 and 200 dimensions, and `f16gcas`.
3. Generate a results table and figure using the logged metrics and trajectories.

Individual steps are next discussed in more detail:

## Data

The experiments require Monte Carlo datasets for reference failure probabilities and coverage/NDB metric calculations. Datasets can be created using the following syntax:

```
python3 utils/mc_data_generation.py --problem Branches --d 40
python3 utils/mc_data_generation.py --problem Branches --d 60
python3 utils/mc_data_generation.py --problem Oscillator --d 100
python3 utils/mc_data_generation.py --problem Oscillator --d 200
python3 utils/mc_data_generation.py --problem F16GCAS
```

## Experiments
Example commands to reproduce the `f16gcas` results are shown below. To change the problem, replace `f16gcas` with either `branches` or `oscillator` and update the `--n_features` flag with the corresponding problem dimensionality.

```
python3 experiments.py --problem "f16gcas" --is_method "CE" --proposal "MPPCA" --n_features 202 --n_components 8 --n_factors 8
python3 experiments.py --problem "f16gcas" --is_method "SIS" --proposal "MPPCA" --n_features 202 --n_components 8 --n_factors 8
python3 experiments.py --problem "f16gcas" --is_method "CE" --proposal "GMM" --n_features 202 --n_components 8
python3 experiments.py --problem "f16gcas" --is_method "SIS" --proposal "GMM" --n_features 202 --n_components 8
```

## Results

All saved metrics can be viewed in tabular form by running the following command:
```
python3 utils/results_table.py
```

Furthermore, code to reproduce the results plot in the paper is provided:
```
python3 utils/results_fig.py
```
![Results](assets/results.png#gh-light-mode-only)
![Results](assets/results_dark.png#gh-dark-mode-only)

## How to Cite
If you find this code useful in your research, please cite the following publication:
```
@inproceedings{kruse2025scalable,
  title={Scalable Importance Sampling in High Dimensions with Low-Rank Mixture Proposals},
  author={Kruse, Liam A and Schlichting, Marc R and Kochenderfer, Mykel J},
  booktitle={2025 11th International Conference on Control, Decision and Information Technologies (CoDIT)},
  year={2025}
}
```
