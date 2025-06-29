import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
import warnings
warnings.filterwarnings('ignore')


def analyze_features(year, race_name):
    print("Starting feature analysis for {} {}...".format(year, race_name))
    
    # Load data
    input_path = "data/processed/{}/{}/laps_features.csv".format(year, race_name)
    df = pd.read_csv(input_path)
    print("Loaded dataset with {} rows and {} columns".format(df.shape[0], df.shape[1]))
    
    # Create output directory
    output_dir = "data/processed/{}/{}/".format(year, race_name)
    os.makedirs(output_dir, exist_ok=True)

    # Select only numeric columns for correlation analysis
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    print("Found {} numeric columns for analysis".format(len(numeric_columns)))
    
    # Correlation analysis
    print("Creating correlation matrix...")
    numeric_df = df[numeric_columns]
    corr_matrix = numeric_df.corr()
    
    plt.figure(figsize=(16, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Show only lower triangle
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='RdBu_r', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Feature Correlation Matrix (Lower Triangle)', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_correlation_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Find highly correlated features
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.95:
                high_corr_pairs.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': corr_matrix.iloc[i, j]
                })
    
    if high_corr_pairs:
        print("\nFound {} highly correlated feature pairs (>0.95):".format(len(high_corr_pairs)))
        for pair in high_corr_pairs:
            print("  {} <-> {}: {:.3f}".format(pair['feature1'], pair['feature2'], pair['correlation']))
    else:
        print("\nNo highly correlated feature pairs found (>0.95)")
    
    # Feature importance analysis using Random Forest
    print("\nCalculating feature importance with Random Forest...")
    if 'LapTime' in numeric_df.columns:
        X = numeric_df.drop(['LapTime'], axis=1)
        y = numeric_df['LapTime']
        
        # Remove any rows with NaN values
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X_clean = X[mask]
        y_clean = y[mask]
        
        if len(X_clean) > 0:
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_clean, y_clean)
            
            # Get feature importance
            feature_importance = pd.DataFrame({
                'feature': X_clean.columns,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Plot top 20 features
            plt.figure(figsize=(12, 8))
            top_features = feature_importance.head(20)
            sns.barplot(data=top_features, x='importance', y='feature')
            plt.title('Top 20 Feature Importance (Random Forest)', fontsize=14)
            plt.xlabel('Feature Importance')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'feature_importance_rf.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            print("Top 10 most important features:")
            for idx, row in feature_importance.head(10).iterrows():
                print("  {}: {:.4f}".format(row['feature'], row['importance']))
            
            # Save feature importance to CSV
            feature_importance.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
    
    # Basic statistics summary
    print("\nGenerating feature statistics...")
    stats_summary = numeric_df.describe()
    stats_summary.to_csv(os.path.join(output_dir, 'feature_statistics.csv'))
    
    # Check for features with many missing values
    missing_data = (df.isnull().sum() / len(df)) * 100
    high_missing = missing_data[missing_data > 10].sort_values(ascending=False)
    
    if len(high_missing) > 0:
        print("\nFeatures with >10% missing values:")
        for feature, pct in high_missing.items():
            print("  {}: {:.1f}%".format(feature, pct))
    else:
        print("\nNo features with >10% missing values found")
    
    # Distribution analysis for key features
    print("\nCreating distribution plots for key features...")
    key_features = ['LapTime', 'Position', 'gap_to_leader', 'lap_delta', 'TyreLife']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, feature in enumerate(key_features):
        if feature in numeric_df.columns and i < len(axes):
            axes[i].hist(numeric_df[feature].dropna(), bins=30, alpha=0.7, edgecolor='black')
            axes[i].set_title('Distribution of {}'.format(feature))
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Frequency')
    
    # Remove empty subplot
    if len(key_features) < len(axes):
        fig.delaxes(axes[-1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'key_feature_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n✅ Feature analysis completed and saved in {}".format(output_dir))
    print("📊 Generated files:")
    print("   - feature_correlation_heatmap.png")
    print("   - feature_importance_rf.png")
    print("   - feature_importance.csv")
    print("   - feature_statistics.csv")
    print("   - key_feature_distributions.png")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze features for correlation and distribution.")
    parser.add_argument("--year", type=int, required=True, help="Year of the race")
    parser.add_argument("--race", type=str, required=True, help="Race name")
    args = parser.parse_args()

    analyze_features(args.year, args.race)
