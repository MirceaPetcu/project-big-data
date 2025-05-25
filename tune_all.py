import sys
import subprocess


### tune all models with bayesian optimization ###
models = ['ridge', 'rf', 'svr', 'en', 'lgbm']
dr = ['pca', 'svd', 'fa']
objective = 'mse'

for model in models:
    for dr in dr:
        subprocess.run(['python', 'tune.py', '--model', model, '--dr', dr, '--objective', objective])