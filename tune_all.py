import sys
import subprocess


### tune all models with bayesian optimization ###
models = ['rf', 'ridge', 'en', 'lgbm', 'dt']
objective = 'mse'
n_trials = [150, 200, 200, 200, 200]

for model in models:
    subprocess.run(['python', 'tune.py',
                    '--model', model,
                    '--objective', objective,
                    '--n_trials', str(n_trials[models.index(model)])])