# MC data generation
python3 utils/mc_data_generation.py --problem Branches --d 40
python3 utils/mc_data_generation.py --problem Branches --d 60
python3 utils/mc_data_generation.py --problem Oscillator --d 100
python3 utils/mc_data_generation.py --problem Oscillator --d 200
python3 utils/mc_data_generation.py --problem F16GCAS

# MPPCA-CEIS experiments
python3 experiments.py --problem "branches" --is_method "CE" --proposal "MPPCA" --n_features 40 --n_components 8 --n_factors 8
python3 experiments.py --problem "branches" --is_method "SIS" --proposal "MPPCA" --n_features 40 --n_components 8 --n_factors 8
python3 experiments.py --problem "branches" --is_method "CE" --proposal "GMM" --n_features 40 --n_components 8
python3 experiments.py --problem "branches" --is_method "SIS" --proposal "GMM" --n_features 40 --n_components 8

python3 experiments.py --problem "branches" --is_method "CE" --proposal "MPPCA" --n_features 60 --n_components 8 --n_factors 8
python3 experiments.py --problem "branches" --is_method "SIS" --proposal "MPPCA" --n_features 60 --n_components 8 --n_factors 8
python3 experiments.py --problem "branches" --is_method "CE" --proposal "GMM" --n_features 60 --n_components 8
python3 experiments.py --problem "branches" --is_method "SIS" --proposal "GMM" --n_features 60 --n_components 8

python3 experiments.py --problem "oscillator" --is_method "CE" --proposal "MPPCA" --n_features 100 --n_components 8 --n_factors 8
python3 experiments.py --problem "oscillator" --is_method "SIS" --proposal "MPPCA" --n_features 100 --n_components 8 --n_factors 8
python3 experiments.py --problem "oscillator" --is_method "CE" --proposal "GMM" --n_features 100 --n_components 8
python3 experiments.py --problem "oscillator" --is_method "SIS" --proposal "GMM" --n_features 100 --n_components 8

python3 experiments.py --problem "oscillator" --is_method "CE" --proposal "MPPCA" --n_features 200 --n_components 8 --n_factors 8
python3 experiments.py --problem "oscillator" --is_method "SIS" --proposal "MPPCA" --n_features 200 --n_components 8 --n_factors 8
python3 experiments.py --problem "oscillator" --is_method "CE" --proposal "GMM" --n_features 200 --n_components 8
python3 experiments.py --problem "oscillator" --is_method "SIS" --proposal "GMM" --n_features 200 --n_components 8

python3 experiments.py --problem "f16gcas" --is_method "CE" --proposal "MPPCA" --n_features 202 --n_components 8 --n_factors 8
python3 experiments.py --problem "f16gcas" --is_method "SIS" --proposal "MPPCA" --n_features 202 --n_components 8 --n_factors 8
python3 experiments.py --problem "f16gcas" --is_method "CE" --proposal "GMM" --n_features 202 --n_components 8
python3 experiments.py --problem "f16gcas" --is_method "SIS" --proposal "GMM" --n_features 202 --n_components 8

# results table
python3 utils/results_table.py

# results figure
python3 utils/results_fig.py
