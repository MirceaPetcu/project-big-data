import sys
import subprocess


### tune all models with bayesian optimization ###
models = ['en', 'lgbm', 'svr', 'rf']
objective = 'mse'

for model in models:
    subprocess.run(['python', 'tune.py', '--model', model, '--objective', objective])