import os
import random
from sklearn.decomposition import PCA, TruncatedSVD, FactorAnalysis
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import optuna
from optuna.visualization import plot_optimization_history
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from scipy.stats import spearmanr, kendalltau
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.linear_model import ElasticNet, Ridge
import argparse
from sklearn.linear_model import ARDRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from umap import UMAP


def parse_args():
    parser = argparse.ArgumentParser(description='Hyperparameter tuning')
    parser.add_argument('--model', type=str, default='svr', help='Model to tune')
    parser.add_argument('--dr', type=str, default='pca', help='DR method')
    parser.add_argument('--objective', type=str, default='mse', help='Objective to optimize')
    parser.add_argument('--n_trials', type=int, default=200, help='Number of trials')
    return parser.parse_args()


args = parse_args()
np.random.seed(6)
random.seed(6)
df = pd.read_csv(os.path.join(os.getcwd(), 'data', 'preprocessed_data.csv'))

x = df.drop(columns=['log_shares']).values
y = df['log_shares'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=6)



optuna.logging.set_verbosity(optuna.logging.WARNING)


def get_model(regressor_params):
    match args.model:
        case 'rf':
            model = RandomForestRegressor(**regressor_params)
        case 'lgbm':
            model = lgb.LGBMRegressor(verbosity=-1,**regressor_params)
        case 'ridge':
            model = Ridge(**regressor_params)
        case 'ard':
            model = ARDRegression(**regressor_params)
        case 'svr':
            model = SVR(**regressor_params)
        case 'mlp':
            model = MLPRegressor(**regressor_params)
        case 'kr':
            model = KernelRidge(**regressor_params)
        case 'en':
            model = ElasticNet(**regressor_params)
        case _:
            model = RandomForestRegressor(**regressor_params)
    return model


def get_dr(dr_params):
    match args.dr:
        case 'pca':
            return PCA(**dr_params)
        case 'svd':
            return TruncatedSVD(**dr_params)
        case 'umap':
            return UMAP(**dr_params)
        case 'fa':
            return FactorAnalysis(**dr_params)
        case _:
            return PCA(**dr_params)

def objective(trial):
    global x_train, y_train
    match args.model:
        case 'rf':
            model_param = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', [None, 'sqrt', 'log2']),
                'bootstrap': True,
                'max_depth': trial.suggest_int('max_depth', 1, 100),
                'random_state': 6
            }
        case 'lgbm':
            model_param = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 1, 100),
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1),
                'num_leaves': trial.suggest_int('num_leaves', 2, 256),
                'lambda_l1': trial.suggest_float('lambda_l1', 1e-5, 1),
                'lambda_l2': trial.suggest_float('lambda_l2', 1e-5, 1),
                'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
                'random_state': 6
            }
        case 'ridge':
            model_param = {
                'alpha': trial.suggest_float('alpha', 1e-5, 100, log=True),
                'solver': 'auto',
                'random_state': 6
            }
        case 'en':
            model_param = {
                'alpha': trial.suggest_float('alpha', 1e-5, 100, log=True),
                'l1_ratio': trial.suggest_float('l1_ratio', 0, 1),
            }
        case 'svr':
            model_param = {
                'C': trial.suggest_float('C', 1e-5, 100, log=True),
                'epsilon': trial.suggest_float('epsilon', 1e-5, 1),
                'kernel': trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
                'degree': trial.suggest_int('degree', 1, 10),
            }
        case 'mlp':
            model_param = {
                'hidden_layer_sizes': trial.suggest_int('hidden_layer_sizes', 50, 200),
                'activation': trial.suggest_categorical('activation', ['identity', 'logistic', 'tanh', 'relu']),
                'solver': trial.suggest_categorical('solver', ['lbfgs', 'sgd', 'adam']),
                'alpha': trial.suggest_float('alpha', 1e-5, 1),
                'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'invscaling', 'adaptive']),
                'random_state': 6
            }
        case _:
            model_param = {}


    model = get_model(model_param)
    match args.dr:
        case 'pca':
            dr_params = {
                'n_components': trial.suggest_float('n_components', 0.7, 0.95)
            }
        case 'svd':
            dr_params = {
                'n_components': trial.suggest_int('n_components', 5, 37)
            }
        case 'umap':
            dr_params = {
                'n_components': trial.suggest_int('n_components', 5, 37),
                'n_neighbors': trial.suggest_int('n_neighbors', 5, 50),
            }
        case 'fa':
            dr_params = {
                'n_components': trial.suggest_int('n_components', 5, 37),
            }
        case _:
            dr_params = {}

    dr = get_dr(dr_params)
    x_train_dr = dr.fit_transform(x_train.copy())
    kf = KFold(n_splits=5, shuffle=True, random_state=6)
    mae = 0
    mse = 0
    for train_index, test_index in kf.split(x_train_dr):
        x_train_fold, x_test_fold = x_train_dr[train_index], x_train_dr[test_index]
        y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]
        model.fit(x_train_fold, y_train_fold)
        y_pred = model.predict(x_test_fold)
        mae += mean_absolute_error(y_test_fold, y_pred)
        mse += mean_squared_error(y_test_fold, y_pred)

    mae /= 5
    mse /= 5
    del x_train_dr
    trial.set_user_attr('mae', mae)
    trial.set_user_attr('mse', mse)
    match args.objective:
        case 'mae':
            return mae
        case 'mse':
            return mse
        case _:
            return mae


if __name__ == '__main__':
    args.n_trials = 20 if args.dr == 'umap' else args.n_trials
    n_warmup_steps = 2 if args.dr == 'umap' else 10
    study = optuna.create_study(**{
        'study_name': f'{args.model}_{args.dr}_{args.objective}',
        'storage': None,
        'load_if_exists': True,
        'direction': 'maximize' if args.objective in ['spearman', 'kendall'] else 'minimize',
        'sampler': optuna.samplers.TPESampler(seed=6),
        'pruner': optuna.pruners.MedianPruner(n_warmup_steps=n_warmup_steps)
    })

    study.optimize(**{
        'func': objective,
        'n_trials': args.n_trials,
        'n_jobs': -1,
        'show_progress_bar': True
    })

    print(study.best_params)
    print(study.best_value)
    print(study.best_trial)
    
    os.makedirs(os.path.join(os.getcwd(), 'results'), exist_ok=True)
    if not os.path.exists(os.path.join(os.getcwd(), 'results', 'results.csv')):
        df_results = pd.DataFrame(columns=['model', 'dr', 'objective', 'best_params', 'best_value'])
    else:
        df_results = pd.read_csv(os.path.join(os.getcwd(), 'results', 'results.csv'))
    trial_result = pd.DataFrame({
        'model': args.model,
        'dr': args.dr,
        'objective': args.objective,
        'best_params': str(study.best_params),
        'best_value': study.best_value,
    }, index=[0])
    df_results = pd.concat([df_results, trial_result], ignore_index=True)
    df_results.to_csv(os.path.join(os.getcwd(), 'results', 'results.csv'), index=False)

    os.makedirs(os.path.join(os.getcwd(), 'plots'), exist_ok=True)
    fig = plot_optimization_history(**{
        'study': study
    })
    fig.write_image(os.path.join(os.getcwd(), 'plots', f"optimization_history_{args.model}_{args.dr}_{args.objective}.png"))

    plt.show()

    