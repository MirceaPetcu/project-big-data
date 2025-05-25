import sys
import subprocess


### tune all models with bayesian optimization ###
models = ['ridge', 'rf', 'en', 'lgbm', 'svr']
objective = 'mse'
n_trials = [200, 150, 200, 200, 100]

for model in models:
    subprocess.run(['python', 'tune.py',
                    '--model', model,
                    '--objective', objective,
                    '--n_trials', str(n_trials[models.index(model)])])