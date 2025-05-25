import os
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import LocallyLinearEmbedding
from umap import UMAP
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


df = pd.read_csv(os.path.join(os.getcwd(), 'data', 'preprocessed_data.csv'))

x = df.drop(['shares', 'log_shares', 'popular', 'weekday', 'channel'], axis=1)
y = df['log_shares']

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

standard_scaler = StandardScaler()
x_train = standard_scaler.fit_transform(x_train)
x_test = standard_scaler.transform(x_test)

pca = PCA(n_components=0.95)
pca.fit(x_train)

x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)

svd = TruncatedSVD(n_components=30)
svd.fit(x_train)

x_train_svd = svd.fit_transform(x_train)
x_test_svd = svd.transform(x_test)

umap = UMAP(n_components=30)
umap.fit(x_train)

x_train_umap = umap.fit_transform(x_train)
x_test_umap = umap.transform(x_test)

# Test Ridge regression performance without and with dimensionality reduction

# Ridge regression without dimensionality reduction
ridge_original = Ridge(alpha=1.0, random_state=42)
ridge_original.fit(x_train, y_train)
y_pred_original = ridge_original.predict(x_test)

# Ridge regression with PCA
ridge_pca = Ridge(alpha=1.0, random_state=42)
ridge_pca.fit(x_train_pca, y_train)
y_pred_pca = ridge_pca.predict(x_test_pca)

# Ridge regression with SVD
ridge_svd = Ridge(alpha=1.0, random_state=42)
ridge_svd.fit(x_train_svd, y_train)
y_pred_svd = ridge_svd.predict(x_test_svd)

# Ridge regression with UMAP
ridge_umap = Ridge(alpha=1.0, random_state=42)
ridge_umap.fit(x_train_umap, y_train)
y_pred_umap = ridge_umap.predict(x_test_umap)

# Evaluate performance
from sklearn.metrics import mean_squared_error, r2_score

print("Ridge Regression Performance Comparison:")
print("=" * 50)

# Original data
mse_original = mean_squared_error(y_test, y_pred_original)
r2_original = r2_score(y_test, y_pred_original)
print(f"Original data - MSE: {mse_original:.4f}, R²: {r2_original:.4f}")

# PCA
mse_pca = mean_squared_error(y_test, y_pred_pca)
r2_pca = r2_score(y_test, y_pred_pca)
print(f"PCA ({pca.n_components_} components) - MSE: {mse_pca:.4f}, R²: {r2_pca:.4f}")

# SVD
mse_svd = mean_squared_error(y_test, y_pred_svd)
r2_svd = r2_score(y_test, y_pred_svd)
print(f"SVD (30 components) - MSE: {mse_svd:.4f}, R²: {r2_svd:.4f}")

# UMAP
mse_umap = mean_squared_error(y_test, y_pred_umap)
r2_umap = r2_score(y_test, y_pred_umap)
print(f"UMAP (30 components) - MSE: {mse_umap:.4f}, R²: {r2_umap:.4f}")


# Plot distribution of predicted values for all methods
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))

# Create subplots for better visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Distribution of Predicted Values on Test Set', fontsize=16)

# Original data
axes[0, 0].hist(y_pred_original, bins=30, alpha=0.7, color='blue', edgecolor='black')
axes[0, 0].set_title(f'Original Data\nMSE: {mse_original:.4f}, R²: {r2_original:.4f}')
axes[0, 0].set_xlabel('Predicted Values')
axes[0, 0].set_ylabel('Frequency')

# PCA
axes[0, 1].hist(y_pred_pca, bins=30, alpha=0.7, color='green', edgecolor='black')
axes[0, 1].set_title(f'PCA ({pca.n_components_} components)\nMSE: {mse_pca:.4f}, R²: {r2_pca:.4f}')
axes[0, 1].set_xlabel('Predicted Values')
axes[0, 1].set_ylabel('Frequency')

# SVD
axes[1, 0].hist(y_pred_svd, bins=30, alpha=0.7, color='red', edgecolor='black')
axes[1, 0].set_title(f'SVD (30 components)\nMSE: {mse_svd:.4f}, R²: {r2_svd:.4f}')
axes[1, 0].set_xlabel('Predicted Values')
axes[1, 0].set_ylabel('Frequency')

# UMAP
axes[1, 1].hist(y_pred_umap, bins=30, alpha=0.7, color='orange', edgecolor='black')
axes[1, 1].set_title(f'UMAP (30 components)\nMSE: {mse_umap:.4f}, R²: {r2_umap:.4f}')
axes[1, 1].set_xlabel('Predicted Values')
axes[1, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# Also create an overlapping histogram for direct comparison
plt.figure(figsize=(12, 6))
plt.hist(y_pred_original, bins=30, alpha=0.6, label='Original', color='blue')
plt.hist(y_pred_pca, bins=30, alpha=0.6, label=f'PCA ({pca.n_components_} comp)', color='green')
plt.hist(y_pred_svd, bins=30, alpha=0.6, label='SVD (30 comp)', color='red')
plt.hist(y_pred_umap, bins=30, alpha=0.6, label='UMAP (30 comp)', color='orange')
plt.hist(y_test, bins=30, alpha=0.8, label='True Values', color='black', linestyle='--', histtype='step', linewidth=2)

plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Distribution Comparison: Predicted vs True Values on Test Set')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
