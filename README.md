# Online News Popularity Prediction

A comprehensive machine learning project for predicting the popularity of online news articles using the UCI Online News Popularity dataset. This project implements multiple regression models with hyperparameter optimization to predict the number of shares an article will receive.

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Models](#models)
- [Results](#results)
- [Visualizations](#visualizations)
- [Contributing](#contributing)

## 🎯 Overview

This project aims to predict the popularity of online news articles (measured by the number of shares) using various machine learning techniques. The project includes:

- **Exploratory Data Analysis (EDA)** to understand data patterns
- **Feature Engineering** to create meaningful predictors
- **Dimensionality Reduction** techniques (PCA, SVD, UMAP)
- **Multiple ML Models** with hyperparameter optimization
- **Bayesian Optimization** using Optuna for efficient hyperparameter tuning

## 📊 Dataset

The project uses the **Online News Popularity Dataset** from UCI Machine Learning Repository, which contains:

- **39,644 articles** from Mashable
- **58 predictive features** including:
  - Article metadata (publication day, data channel)
  - Content features (title tokens, content length, images, videos)
  - Natural language processing features (polarity, subjectivity)
  - Keyword features (best/worst keyword shares)
  - Time-based features (publication timing)
- **Target variable**: `shares` (number of times the article was shared)

### Key Features
- `kw_avg_avg`: Average keyword shares
- `LDA_*`: Latent Dirichlet Allocation topic probabilities
- `self_reference_*`: Self-reference share metrics
- `data_channel_is_*`: Article category indicators
- `num_hrefs`, `num_imgs`, `num_videos`: Content metadata

## 📁 Project Structure

```
project/
├── data/
│   ├── OnlineNewsPopularity.csv          # Original dataset
│   ├── train_selected_features.csv        # Processed training data
│   └── test_selected_features.csv         # Processed test data
├── eda.ipynb                          # Exploratory Data Analysis
├── dim_reduction.ipynb                # Dimensionality reduction experiments
├── models.ipynb                       # Model development and comparison
├── tune.py                            # Hyperparameter tuning script
└── tune_all.py                        # Batch tuning for all models
├── results/
│   └── results.csv                        # Model performance results
├── plots/
│   └── optimization_history_*.png         # Optimization visualization plots
├── requirements.txt                       # Python dependencies
└── README.md                             # Project documentation
```

## 🚀 Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd online-news-popularity
```

2. **Create virtual environment** (recommended)
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Single Model Tuning
Tune hyperparameters for a specific model:

```bash
python tune.py --model ridge --objective mse --n_trials 100
```

**Parameters:**
- `--model`: Model type (`ridge`, `rf`, `lgbm`, `dt`, `ada`)
- `--objective`: Optimization objective (`mse`, `mae`)
- `--n_trials`: Number of optimization trials

### Batch Model Tuning
Run hyperparameter optimization for all models:

```bash
python tune_all.py
```

### Jupyter Notebooks
Explore the analysis interactively:

1. **EDA**: `jupyter notebook eda.ipynb`
2. **Dimensionality Reduction**: `jupyter notebook dim_reduction.ipynb`
3. **Model Comparison**: `jupyter notebook models.ipynb`

## 🔬 Methodology

### 1. Data Preprocessing
- **Log transformation** of target variable (`log_shares = log(shares + 1)`)
- **Feature selection** based on correlation analysis and domain knowledge
- **Standardization** for models sensitive to feature scales
- **Train-test split** (80-20) with stratification on popularity

### 2. Feature Engineering
Created additional features to improve model performance:
- `engagement_score`: Combined measure of hrefs, images, videos, and keywords
- `href_ratio`: Ratio of self-references to total hrefs
- `title_token_density`: Title length relative to content
- `keyword_title_ratio`: Keywords per title token

### 3. Dimensionality Reduction
Explored multiple techniques:
- **PCA**: Principal Component Analysis
- **Factor Analysis**: Statistical factor modeling
- **NMF**: Non-negative matrix factorization

### 4. Model Selection & Tuning
- **Bayesian Optimization** with Optuna
- **5-fold Cross-validation** for robust performance estimation
- **Multiple algorithms** tested and compared
- **Automated hyperparameter search** across predefined ranges

## 🤖 Models

The project implements and compares the following regression models:

| Model | Description | Key Hyperparameters |
|-------|-------------|-------------------|
| **Ridge Regression** | L2 regularized linear regression | `alpha` |
| **Random Forest** | Ensemble of decision trees | `n_estimators`, `max_depth`, `min_samples_split` |
| **LightGBM** | Gradient boosting framework | `learning_rate`, `num_leaves`, `lambda_l1/l2` |
| **Decision Tree** | Single decision tree | `max_depth`, `min_samples_split/leaf` |
| **AdaBoost** | Adaptive boosting ensemble | `n_estimators`, `learning_rate` |


## 📈 Results

Model performance is evaluated using:
- **Mean Squared Error (MSE)**
- **Mean Absolute Error (MAE)**
- **R-squared (R2)**
- **Pearson correlation**

Results from hyperparameter optimization are automatically saved to `results/results.csv` and include:
- Best hyperparameters for each model
- Cross-validation performance metrics
- Optimization history plots

## 📊 Visualizations

The project generates various visualizations:

### EDA Plots
- Share distribution analysis (original vs log-transformed)
- Feature correlation heatmaps
- Category-wise share analysis (channels, weekdays)
- Feature importance rankings

### Optimization Plots
- Hyperparameter optimization history
- Model performance comparisons
- Feature importance visualizations

### Dimensionality Reduction
- 2D/3D projections of high-dimensional data
- Explained variance plots
- Component analysis visualizations

## 🎯 Key Insights

1. **Log transformation** significantly improves model performance due to share distribution skewness
2. **Keyword-related features** (`kw_avg_avg`, `kw_max_avg`) are among the strongest predictors
3. **Self-reference metrics** provide valuable signals for share prediction
4. **Content engagement features** (images, videos, hrefs) correlate with popularity
5. **Publication timing** and **data channels** show distinct sharing patterns

## 🛠️ Technical Details

- **Language**: Python 3.8+
- **Key Libraries**: scikit-learn, optuna, pandas, numpy, matplotlib, seaborn
- **Optimization**: Bayesian optimization with TPE sampler
- **Validation**: K-fold cross-validation (k=5)
- **Preprocessing**: StandardScaler normalization

## 📋 Requirements

See `requirements.txt` for full dependency list:
- optuna
- scikit-learn
- numpy
- pandas
- matplotlib
- xgboost
- lightgbm
- umap-learn
- kaleido


## 📚 References

- [Online News Popularity Dataset](https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity)
- [Optuna: Hyperparameter Optimization Framework](https://optuna.org/)
- [Scikit-learn Documentation](https://scikit-learn.org/)

