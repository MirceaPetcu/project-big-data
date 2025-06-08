import sys
import subprocess


### tune all models with bayesian optimization ###
models = ['ada', 'ridge', 'lgbm', 'dt', 'rf']
objective = 'mse'
n_trials = [100, 100, 100, 100, 100]

for model in models:
    subprocess.run(['python', 'tune.py',
                    '--model', model,
                    '--objective', objective,
                    '--n_trials', str(n_trials[models.index(model)])])