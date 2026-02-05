# visualize.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr, spearmanr, shapiro, jarque_bera
import seaborn as sns
import warnings
import os
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings('ignore')

# Global plot settings
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2


def detect_delimiter(file_path, encoding='gbk'):
    """Detects CSV delimiter automatically."""
    with open(file_path, 'r', encoding=encoding) as f:
        first_line = f.readline()

    delimiters = [',', '\t', ';', '|', ' ']
    for delim in delimiters:
        if delim in first_line:
            if first_line.count(delim) >= 6:
                return delim
    return ','


def load_data(train_path='./train1/train_data1.csv', test_path='./train1/test_data1.csv'):
    """Loads and merges training and testing data."""
    delimiter = detect_delimiter(train_path, encoding='gbk')

    train_data = pd.read_csv(train_path, encoding='gbk', sep=delimiter)
    test_data = pd.read_csv(test_path, encoding='gbk', sep=delimiter)

    train_data.columns = train_data.columns.str.strip()
    test_data.columns = test_data.columns.str.strip()

    full_data = pd.concat([train_data, test_data], ignore_index=True)
    full_data = full_data.dropna()

    print(f"Data loaded. Total samples: {len(full_data)}")
    return full_data


def translate_column_names(data):
    """Translates Chinese column names to English."""
    translation_dict = {
        '弹体重量': 'Mass',
        '弹体直径': 'Diameter',
        '弹体长度': 'Length',
        '侵彻速度': 'Velocity',
        '靶体抗压强度': '$f_c$',
        '弹体形状系数': 'CRH',
        '侵彻深度': 'Dop'
    }
    data_translated = data.copy()
    data_translated.columns = [translation_dict.get(col, col) for col in data.columns]
    return data_translated


def comprehensive_statistical_analysis(data):
    """Performs comprehensive statistical analysis."""
    print("\n" + "=" * 50)
    print("COMPREHENSIVE STATISTICAL ANALYSIS")
    print("=" * 50)

    # 1. Descriptive Statistics
    print("\n--- 1. Descriptive Statistics ---")
    desc_stats = data.describe().T
    desc_stats['CV (%)'] = (desc_stats['std'] / desc_stats['mean'] * 100).round(2)
    desc_stats['Range'] = desc_stats['max'] - desc_stats['min']
    desc_stats['IQR'] = desc_stats['75%'] - desc_stats['25%']
    print(desc_stats[['mean', 'std', 'min', '25%', '50%', '75%', 'max', 'CV (%)', 'Range', 'IQR']])

    # 2. Normality Tests
    print("\n--- 2. Normality Tests ---")
    print(f"{'Feature':<15} {'SW p-val':<15} {'JB p-val':<15} {'Normal?'}")
    for col in data.columns:
        _, sw_pval = shapiro(data[col])
        _, jb_pval = jarque_bera(data[col])
        is_normal = "Yes" if (sw_pval > 0.05 and jb_pval > 0.05) else "No"
        print(f"{col:<15} {sw_pval:<15.6f} {jb_pval:<15.6f} {is_normal}")

    # 3. Skewness and Kurtosis
    print("\n--- 3. Skewness and Kurtosis ---")
    print(f"{'Feature':<15} {'Skewness':<15} {'Kurtosis':<15}")
    for col in data.columns:
        print(f"{col:<15} {data[col].skew():<15.4f} {data[col].kurtosis():<15.4f}")

    # 4. Correlation Analysis
    print("\n--- 4. Correlation with Target (Dop) ---")
    target_col = data.columns[-1]
    print(f"{'Feature':<15} {'Pearson r':<15} {'Spearman ρ':<15}")
    for col in data.columns[:-1]:
        p_r, _ = pearsonr(data[col], data[target_col])
        s_r, _ = spearmanr(data[col], data[target_col])
        print(f"{col:<15} {p_r:<15.6f} {s_r:<15.6f}")

    # 5. Correlation Matrices
    print("\n--- 5. Pearson Correlation Matrix ---")
    print(data.corr(method='pearson').round(4))

    # 6. Outlier Detection (IQR)
    print("\n--- 6. Outlier Detection (1.5*IQR) ---")
    print(f"{'Feature':<15} {'Outliers Count'}")
    for col in data.columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        outliers = ((data[col] < (Q1 - 1.5 * (Q3 - Q1))) | (data[col] > (Q3 + 1.5 * (Q3 - Q1)))).sum()
        print(f"{col:<15} {outliers}")

    # 7. Linear Regression Stats
    print("\n--- 7. Linear Regression (Feature vs Dop) ---")
    print(f"{'Feature':<15} {'R²':<15} {'RMSE':<15}")
    for col in data.columns[:-1]:
        slope, intercept, r_value, _, _ = stats.linregress(data[col].values, data[target_col].values)
        y_pred = slope * data[col].values + intercept
        rmse = np.sqrt(np.mean((data[target_col].values - y_pred) ** 2))
        print(f"{col:<15} {r_value ** 2:<15.6f} {rmse:<15.6f}")

    # 8. Multicollinearity (VIF)
    print("\n--- 8. Multicollinearity (VIF) ---")
    X = data.iloc[:, :-1]
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    print(vif_data)

    # 9. Data Quality
    print("\n--- 9. Data Quality ---")
    print(f"Missing: {data.isnull().sum().sum()}, Duplicates: {data.duplicated().sum()}")


def plot_correlation_matrix(data, output_dir='./scatter_plots/'):
    """Generates and saves the correlation matrix scatter plot."""
    os.makedirs(output_dir, exist_ok=True)
    n_vars = len(data.columns)

    fig, axes = plt.subplots(n_vars, n_vars, figsize=(14, 14))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    corr_matrix = data.corr(method='pearson')

    for i in range(n_vars):
        for j in range(n_vars):
            ax = axes[i, j]

            if i == j:
                # Diagonal: Histogram + KDE
                ax.hist(data.iloc[:, i], bins=30, density=True, alpha=0.6, color='#3498db', edgecolor='white', linewidth=0.5)
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(data.iloc[:, i])
                x_range = np.linspace(data.iloc[:, i].min(), data.iloc[:, i].max(), 200)
                ax.plot(x_range, kde(x_range), color='#e74c3c', linewidth=2.5, alpha=0.8)
                ax.set_yticks([])
                for spine in ['left', 'top', 'right']:
                    ax.spines[spine].set_visible(False)
            else:
                # Off-diagonal: Scatter plot
                ax.scatter(data.iloc[:, j], data.iloc[:, i], s=20, alpha=0.5, color='#2c3e50', edgecolor='white', linewidth=0.3)
                ax.text(0.05, 0.95, f'r = {corr_matrix.iloc[i, j]:.3f}', transform=ax.transAxes, fontsize=9,
                        verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)

            ax.tick_params(labelsize=9, direction='in', length=4, width=1)

            # Labels only on edges
            if i == n_vars - 1:
                ax.set_xlabel(data.columns[j], fontsize=11, fontweight='bold')
                ax.xaxis.set_label_coords(0.5, -0.15)
            else:
                ax.set_xticklabels([])

            if j == 0:
                ax.set_ylabel(data.columns[i], fontsize=11, fontweight='bold')
                ax.yaxis.set_label_coords(-0.15, 0.5)
            else:
                ax.set_yticklabels([])

            for spine in ax.spines.values():
                spine.set_linewidth(1.2)
                spine.set_color('black')

    output_path = os.path.join(output_dir, 'correlation_matrix_scatter.png')
    plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nPlot saved to: {output_path}")


def main():
    print("Starting Analysis...")
    data = load_data()
    data_en = translate_column_names(data)
    comprehensive_statistical_analysis(data_en)
    plot_correlation_matrix(data_en)
    print("\nDone.")


if __name__ == "__main__":
    main()
