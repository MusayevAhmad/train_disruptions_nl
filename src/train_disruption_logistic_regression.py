#!/usr/bin/env python3
"""
Train Disruption Predictive Analytics using Logistic Regression
Assignment 2: Descriptive + Predictive Analytics for Dutch Train Disruptions

Model: Logistic Regression
Objective: Predict whether a disruption will be "Long" (duration ≥ median threshold)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_curve, auc, brier_score_loss, 
    classification_report, confusion_matrix, f1_score
)
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def load_and_combine_data(datasets_path='datasets'):
    """
    Load all CSV files matching disruptions-20*.csv pattern and combine them.
    
    Args:
        datasets_path (str): Path to datasets directory
        
    Returns:
        pd.DataFrame: Combined dataset with all years
    """
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    
    datasets_dir = Path(datasets_path)
    csv_files = sorted(datasets_dir.glob('disruptions-20*.csv'))
    
    if not csv_files:
        print("⚠️  No disruptions-20*.csv files found. Falling back to disruptions-2024.csv")
        csv_files = [datasets_dir / 'disruptions-2024.csv']
    
    print(f"Found {len(csv_files)} CSV file(s):")
    for f in csv_files:
        print(f"  - {f.name}")
    
    all_data = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        all_data.append(df)
        print(f"  Loaded {csv_file.name}: {len(df):,} rows")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\n✅ Combined dataset: {len(combined_df):,} rows, {len(combined_df.columns)} columns")
    
    return combined_df


def engineer_features(df):
    """
    Engineer all required features from the raw data.
    
    Features created:
    - hour_bin: Categorical time-of-day bins
    - weekday: Day of week (0=Monday, 6=Sunday)
    - month: Month (1-12)
    - station_count: Number of affected stations
    - cause_group: Cleaned cause group (rare → Other)
    - is_engineering_work: Binary indicator
    
    Args:
        df (pd.DataFrame): Raw disruption data
        
    Returns:
        pd.DataFrame: Data with engineered features
    """
    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING")
    print("=" * 70)
    
    df = df.copy()
    
    # Parse start_time as timezone-naive datetime
    df['start_time'] = pd.to_datetime(df['start_time'], utc=True).dt.tz_localize(None)
    
    # Extract year
    df['year'] = df['start_time'].dt.year
    
    # Hour bin
    def assign_hour_bin(hour):
        if 0 <= hour <= 5:
            return 'night'
        elif 6 <= hour <= 9:
            return 'morning_peak'
        elif 10 <= hour <= 15:
            return 'midday'
        elif 16 <= hour <= 19:
            return 'evening_peak'
        else:  # 20-23
            return 'late'
    
    df['hour'] = df['start_time'].dt.hour
    df['hour_bin'] = df['hour'].apply(assign_hour_bin)
    
    # Weekday (0=Monday, 6=Sunday)
    df['weekday'] = df['start_time'].dt.dayofweek
    
    # Month (1-12)
    df['month'] = df['start_time'].dt.month
    
    # Station count (handle NaN as 0)
    def count_stations(station_codes):
        if pd.isna(station_codes) or station_codes == '':
            return 0
        return len([s.strip() for s in str(station_codes).split(',') if s.strip()])
    
    df['station_count'] = df['rdt_station_codes'].apply(count_stations)
    
    # Clean cause_group: keep top 8, map rest to "Other"
    cause_counts = df['cause_group'].value_counts()
    top_causes = cause_counts.head(8).index.tolist()
    
    df['cause_group'] = df['cause_group'].fillna('Other')
    df['cause_group'] = df['cause_group'].apply(
        lambda x: x if x in top_causes else 'Other'
    )
    
    # is_engineering_work indicator
    df['is_engineering_work'] = (
        df['cause_group'].str.lower() == 'engineering work'
    ).astype(int)
    
    print(f"✅ Features engineered:")
    print(f"   - hour_bin categories: {sorted(df['hour_bin'].unique())}")
    print(f"   - weekday range: {df['weekday'].min()}-{df['weekday'].max()}")
    print(f"   - month range: {df['month'].min()}-{df['month'].max()}")
    print(f"   - station_count range: {df['station_count'].min()}-{df['station_count'].max()}")
    print(f"   - cause_group categories: {sorted(df['cause_group'].unique())}")
    print(f"   - is_engineering_work: {df['is_engineering_work'].sum()} positives")
    
    return df


def compute_threshold(df, years_for_threshold=None):
    """
    Compute the median duration threshold for defining "Long" disruptions.
    
    Args:
        df (pd.DataFrame): Data with duration_minutes and year
        years_for_threshold (list): Years to use for computing threshold.
                                    If None, use all available years.
    
    Returns:
        float: Median duration threshold in minutes
    """
    print("\n" + "=" * 70)
    print("COMPUTING THRESHOLD")
    print("=" * 70)
    
    if years_for_threshold is None:
        # Use all available years
        df_threshold = df[df['duration_minutes'].notna()]
        print(f"Using all available data: {len(df_threshold):,} rows")
    else:
        # Use specific years
        df_threshold = df[
            (df['year'].isin(years_for_threshold)) & 
            (df['duration_minutes'].notna())
        ]
        print(f"Using years {years_for_threshold}: {len(df_threshold):,} rows")
    
    threshold = df_threshold['duration_minutes'].median()
    print(f"✅ Median duration threshold: {threshold:.1f} minutes")
    
    return threshold


def create_target(df, threshold):
    """
    Create binary target variable is_long.
    
    Args:
        df (pd.DataFrame): Data with duration_minutes
        threshold (float): Threshold for "Long" classification
        
    Returns:
        pd.DataFrame: Data with is_long target variable
    """
    df = df.copy()
    df['is_long'] = (df['duration_minutes'] >= threshold).astype(int)
    
    n_long = df['is_long'].sum()
    n_total = len(df)
    pct_long = 100 * n_long / n_total
    
    print(f"\n📊 Class Distribution:")
    print(f"   - Long (≥{threshold:.1f} min): {n_long:,} ({pct_long:.1f}%)")
    print(f"   - Short (<{threshold:.1f} min): {n_total - n_long:,} ({100 - pct_long:.1f}%)")
    
    return df


def split_data(df):
    """
    Split data into train/validation/test sets using forward-in-time logic.
    
    If years 2021-2024 are available:
        - Train: 2021-2023
        - Validation: 2023 Q4 (Oct-Dec)
        - Test: 2024
    
    If only 2024 is available:
        - Train: earliest 80% (first 10% becomes validation)
        - Validation: last 10% of train set
        - Test: latest 20%
    
    Args:
        df (pd.DataFrame): Full dataset
        
    Returns:
        tuple: (train_df, val_df, test_df, split_scheme)
    """
    print("\n" + "=" * 70)
    print("TRAIN/VALIDATION/TEST SPLIT")
    print("=" * 70)
    
    available_years = sorted(df['year'].unique())
    print(f"Available years: {available_years}")
    
    # Check if we have multi-year data (2021-2024)
    has_multi_year = all(year in available_years for year in [2021, 2022, 2023, 2024])
    
    if has_multi_year:
        print("\n✅ Multi-year scheme: Train=2021-2023, Val=2023 Q4, Test=2024")
        
        # Train: 2021-2023
        train_df = df[df['year'].isin([2021, 2022, 2023])].copy()
        
        # Validation: 2023 Q4 (Oct-Dec)
        val_df = df[
            (df['year'] == 2023) & 
            (df['month'].isin([10, 11, 12]))
        ].copy()
        
        # Test: 2024
        test_df = df[df['year'] == 2024].copy()
        
        # Remove validation set from training set
        train_df = train_df[~train_df.index.isin(val_df.index)]
        
        split_scheme = "multi-year (2021-2023 train, 2023 Q4 val, 2024 test)"
        
    else:
        print("\n✅ Single-year scheme: Time-ordered 80/10/10 split")
        
        # Sort by time
        df_sorted = df.sort_values('start_time').reset_index(drop=True)
        
        n = len(df_sorted)
        train_end_idx = int(0.8 * n)
        val_end_idx = int(0.9 * n)
        
        train_df = df_sorted.iloc[:train_end_idx].copy()
        val_df = df_sorted.iloc[train_end_idx:val_end_idx].copy()
        test_df = df_sorted.iloc[val_end_idx:].copy()
        
        split_scheme = "2024-only (earliest 80% train, next 10% val, latest 10% test)"
    
    print(f"\n📊 Split sizes:")
    print(f"   - Train: {len(train_df):,} rows")
    print(f"   - Validation: {len(val_df):,} rows")
    print(f"   - Test: {len(test_df):,} rows")
    
    # Print class balance for each set
    for name, data in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        if 'is_long' in data.columns:
            pct = 100 * data['is_long'].mean()
            print(f"   - {name} % Long: {pct:.1f}%")
    
    return train_df, val_df, test_df, split_scheme


def build_pipeline():
    """
    Build the ML pipeline with ColumnTransformer and LogisticRegression.
    
    Returns:
        tuple: (ColumnTransformer, LogisticRegression model)
    """
    print("\n" + "=" * 70)
    print("BUILDING PIPELINE")
    print("=" * 70)
    
    # Define categorical and numeric features
    cat_features = ['hour_bin', 'weekday', 'month', 'cause_group']
    num_features = ['station_count', 'is_engineering_work']
    
    # ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), 
             cat_features),
            ('num', 'passthrough', num_features)
        ],
        verbose_feature_names_out=False
    )
    
    print(f"✅ Pipeline components:")
    print(f"   - Categorical features (one-hot encoded): {cat_features}")
    print(f"   - Numeric features (passthrough): {num_features}")
    
    return preprocessor, cat_features, num_features


def tune_hyperparameters(X_train, y_train, X_val, y_val, preprocessor):
    """
    Tune the regularization parameter C using validation set PR-AUC.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        X_val (pd.DataFrame): Validation features
        y_val (pd.Series): Validation target
        preprocessor (ColumnTransformer): Feature preprocessor
        
    Returns:
        tuple: (best_C, best_model, best_pr_auc)
    """
    print("\n" + "=" * 70)
    print("HYPERPARAMETER TUNING")
    print("=" * 70)
    
    C_values = [0.5, 1.0, 2.0]
    results = []
    
    print(f"Tuning C ∈ {C_values} using validation PR-AUC...")
    print(f"Training objective: Log loss (cross-entropy)")
    
    for C in C_values:
        # Create and train model
        model = LogisticRegression(
            penalty='l2',
            C=C,
            class_weight='balanced',
            max_iter=1000,
            random_state=RANDOM_SEED
        )
        
        # Fit preprocessor on training data
        X_train_transformed = preprocessor.fit_transform(X_train)
        model.fit(X_train_transformed, y_train)
        
        # Evaluate on validation set
        X_val_transformed = preprocessor.transform(X_val)
        y_val_proba = model.predict_proba(X_val_transformed)[:, 1]
        
        # Compute PR-AUC (positive class = 1 = Long)
        precision, recall, _ = precision_recall_curve(y_val, y_val_proba)
        pr_auc = auc(recall, precision)
        
        results.append({'C': C, 'PR_AUC': pr_auc, 'model': model})
        print(f"   C={C:.1f}: PR-AUC={pr_auc:.4f}")
    
    # Select best C
    best_result = max(results, key=lambda x: x['PR_AUC'])
    best_C = best_result['C']
    best_model = best_result['model']
    best_pr_auc = best_result['PR_AUC']
    
    print(f"\n✅ Best C: {best_C:.1f} (Validation PR-AUC: {best_pr_auc:.4f})")
    
    return best_C, best_model, best_pr_auc


def select_decision_threshold(y_val, y_val_proba):
    """
    Select optimal decision threshold by maximizing F1 score on validation set.
    
    Args:
        y_val (pd.Series): Validation true labels
        y_val_proba (np.array): Validation predicted probabilities
        
    Returns:
        float: Optimal decision threshold
    """
    print("\n" + "=" * 70)
    print("DECISION THRESHOLD SELECTION")
    print("=" * 70)
    
    # Try different thresholds
    thresholds = np.arange(0.1, 0.9, 0.01)
    f1_scores = []
    
    for thresh in thresholds:
        y_pred = (y_val_proba >= thresh).astype(int)
        f1 = f1_score(y_val, y_pred, pos_label=1)
        f1_scores.append(f1)
    
    # Find best threshold
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    print(f"✅ Optimal threshold: {best_threshold:.3f} (Validation F1: {best_f1:.4f})")
    
    return best_threshold


def evaluate_model(model, preprocessor, X_test, y_test, threshold, best_C):
    """
    Evaluate the model on the test set.
    
    Args:
        model: Trained LogisticRegression model
        preprocessor: Fitted ColumnTransformer
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test true labels
        threshold (float): Decision threshold
        best_C (float): Best regularization parameter
        
    Returns:
        dict: Dictionary containing all evaluation metrics
    """
    print("\n" + "=" * 70)
    print("MODEL EVALUATION (TEST SET)")
    print("=" * 70)
    
    # Transform test data
    X_test_transformed = preprocessor.transform(X_test)
    
    # Get predictions
    y_test_proba = model.predict_proba(X_test_transformed)[:, 1]
    y_test_pred = (y_test_proba >= threshold).astype(int)
    
    # PR-AUC
    precision, recall, _ = precision_recall_curve(y_test, y_test_proba)
    pr_auc = auc(recall, precision)
    
    # Brier score
    brier = brier_score_loss(y_test, y_test_proba)
    
    # Classification report
    report = classification_report(y_test, y_test_pred, output_dict=True, 
                                   target_names=['Short', 'Long'])
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    
    # Extract metrics for Long class (class 1)
    precision_long = report['Long']['precision']
    recall_long = report['Long']['recall']
    f1_long = report['Long']['f1-score']
    
    print(f"\n📊 Test Set Metrics:")
    print(f"   - PR-AUC (Long): {pr_auc:.4f}")
    print(f"   - Brier Score: {brier:.4f}")
    print(f"   - Precision (Long): {precision_long:.4f}")
    print(f"   - Recall (Long): {recall_long:.4f}")
    print(f"   - F1 Score (Long): {f1_long:.4f}")
    
    print(f"\n📋 Confusion Matrix:")
    print(f"   True Negatives:  {cm[0, 0]:,}")
    print(f"   False Positives: {cm[0, 1]:,}")
    print(f"   False Negatives: {cm[1, 0]:,}")
    print(f"   True Positives:  {cm[1, 1]:,}")
    
    metrics = {
        'PR_AUC': pr_auc,
        'Brier_Score': brier,
        'Precision_Long': precision_long,
        'Recall_Long': recall_long,
        'F1_Long': f1_long,
        'Best_C': best_C,
        'Decision_Threshold': threshold,
        'Confusion_Matrix': cm,
        'y_test_proba': y_test_proba,
        'y_test_pred': y_test_pred
    }
    
    return metrics


def plot_figure_1(df, output_path):
    """
    Figure 1: Hour × Weekday heatmap of disruption counts.
    
    Args:
        df (pd.DataFrame): Full dataset
        output_path (str): Path to save the figure
    """
    print("\n📊 Generating Figure 1: Hour × Weekday Heatmap...")
    
    # Create pivot table
    heatmap_data = df.groupby(['hour', 'weekday']).size().unstack(fill_value=0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot heatmap
    im = ax.imshow(heatmap_data.values, cmap='YlOrRd', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(range(7))
    ax.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    ax.set_yticks(range(24))
    ax.set_yticklabels(range(24))
    
    # Labels and title
    ax.set_xlabel('Weekday', fontsize=12)
    ax.set_ylabel('Hour', fontsize=12)
    ax.set_title('Disruptions by Hour × Weekday', fontsize=14, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Count', rotation=270, labelpad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved to {output_path}")


def plot_figure_2(df, output_path):
    """
    Figure 2: Monthly trend per year.
    
    Args:
        df (pd.DataFrame): Full dataset
        output_path (str): Path to save the figure
    """
    print("\n📊 Generating Figure 2: Monthly Trend by Year...")
    
    # Get monthly counts by year
    monthly_data = df.groupby(['year', 'month']).size().reset_index(name='count')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot line for each year
    years = sorted(df['year'].unique())
    for year in years:
        year_data = monthly_data[monthly_data['year'] == year]
        ax.plot(year_data['month'], year_data['count'], marker='o', 
                label=str(year), linewidth=2, markersize=6)
    
    # Labels and title
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Monthly disruptions by year', fontsize=14, fontweight='bold')
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.grid(True, alpha=0.3)
    ax.legend(title='Year', loc='best')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved to {output_path}")


def plot_figure_3(df, output_path):
    """
    Figure 3: Duration distribution by cause group (box plot).
    
    Args:
        df (pd.DataFrame): Full dataset
        output_path (str): Path to save the figure
    """
    print("\n📊 Generating Figure 3: Duration by Cause Group...")
    
    # Clip extreme values at 99th percentile for visibility
    p99 = df['duration_minutes'].quantile(0.99)
    df_plot = df.copy()
    df_plot['duration_clipped'] = df_plot['duration_minutes'].clip(upper=p99)
    
    # Get top cause groups + Other
    cause_order = df_plot['cause_group'].value_counts().index.tolist()
    
    # Prepare data for box plot
    data_by_cause = [
        df_plot[df_plot['cause_group'] == cause]['duration_clipped'].values
        for cause in cause_order
    ]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Box plot
    bp = ax.boxplot(data_by_cause, labels=cause_order, patch_artist=True)
    
    # Style boxes
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    # Labels and title
    ax.set_xlabel('Cause Group', fontsize=12)
    ax.set_ylabel('Duration (minutes)', fontsize=12)
    ax.set_title('Duration distribution by cause group (minutes)', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved to {output_path}")


def plot_figure_4(df, output_path):
    """
    Figure 4: Share of Long disruptions by cause group.
    
    Args:
        df (pd.DataFrame): Full dataset with is_long column
        output_path (str): Path to save the figure
    """
    print("\n📊 Generating Figure 4: % Long by Cause Group...")
    
    # Calculate percentage of Long disruptions by cause group
    pct_long = df.groupby('cause_group')['is_long'].mean() * 100
    pct_long = pct_long.sort_values(ascending=True)  # Sort for horizontal bar
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Horizontal bar chart
    ax.barh(range(len(pct_long)), pct_long.values, color='steelblue', alpha=0.7)
    ax.set_yticks(range(len(pct_long)))
    ax.set_yticklabels(pct_long.index)
    
    # Labels and title
    ax.set_xlabel('% Long disruptions', fontsize=12)
    ax.set_ylabel('Cause Group', fontsize=12)
    ax.set_title('% of Long disruptions by cause group (≥ median threshold)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    
    # Add percentage labels
    for i, v in enumerate(pct_long.values):
        ax.text(v + 1, i, f'{v:.1f}%', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved to {output_path}")


def plot_figure_5(y_test, y_test_proba, brier, pr_auc, output_path):
    """
    Figure 5: Calibration curve.
    
    Args:
        y_test (pd.Series): Test true labels
        y_test_proba (np.array): Test predicted probabilities
        brier (float): Brier score
        pr_auc (float): PR-AUC score
        output_path (str): Path to save the figure
    """
    print("\n📊 Generating Figure 5: Calibration Curve...")
    
    # Compute calibration curve
    prob_true, prob_pred = calibration_curve(y_test, y_test_proba, n_bins=10)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot calibration curve
    ax.plot(prob_pred, prob_true, marker='o', linewidth=2, markersize=8, 
            label='Logistic Regression', color='steelblue')
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2, 
            label='Perfect Calibration')
    
    # Labels and title
    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Fraction of Positives', fontsize=12)
    ax.set_title(f'Calibration (test) — Brier = {brier:.3f}', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    # Add PR-AUC as text
    ax.text(0.95, 0.05, f'PR-AUC = {pr_auc:.3f}', 
            transform=ax.transAxes, fontsize=11,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved to {output_path}")


def save_artifacts(metrics, cat_features, num_features, train_df, val_df, test_df, 
                   threshold, artifacts_dir='artifacts'):
    """
    Save metrics and feature list to artifacts directory.
    
    Args:
        metrics (dict): Evaluation metrics
        cat_features (list): Categorical feature names
        num_features (list): Numeric feature names
        train_df (pd.DataFrame): Training data
        val_df (pd.DataFrame): Validation data
        test_df (pd.DataFrame): Test data
        threshold (float): Duration threshold
        artifacts_dir (str): Path to artifacts directory
    """
    print("\n" + "=" * 70)
    print("SAVING ARTIFACTS")
    print("=" * 70)
    
    # Create artifacts directory
    Path(artifacts_dir).mkdir(exist_ok=True)
    
    # Save metrics
    metrics_df = pd.DataFrame({
        'Metric': [
            'PR_AUC', 'Brier_Score', 'Precision_Long', 'Recall_Long', 'F1_Long',
            'Best_C', 'Decision_Threshold', 'Duration_Threshold_Minutes',
            'Train_Pct_Long', 'Val_Pct_Long', 'Test_Pct_Long',
            'Train_Size', 'Val_Size', 'Test_Size'
        ],
        'Value': [
            metrics['PR_AUC'],
            metrics['Brier_Score'],
            metrics['Precision_Long'],
            metrics['Recall_Long'],
            metrics['F1_Long'],
            metrics['Best_C'],
            metrics['Decision_Threshold'],
            threshold,
            100 * train_df['is_long'].mean(),
            100 * val_df['is_long'].mean(),
            100 * test_df['is_long'].mean(),
            len(train_df),
            len(val_df),
            len(test_df)
        ]
    })
    
    metrics_path = Path(artifacts_dir) / 'metrics_test.csv'
    metrics_df.to_csv(metrics_path, index=False)
    print(f"✅ Saved metrics to {metrics_path}")
    
    # Save feature list
    feature_list_path = Path(artifacts_dir) / 'feature_list.txt'
    with open(feature_list_path, 'w') as f:
        f.write("CATEGORICAL FEATURES (one-hot encoded):\n")
        for feat in cat_features:
            f.write(f"  - {feat}\n")
        f.write("\nNUMERIC FEATURES (passthrough):\n")
        for feat in num_features:
            f.write(f"  - {feat}\n")
    
    print(f"✅ Saved feature list to {feature_list_path}")


def print_model_summary(metrics, best_C, threshold, split_scheme, cat_features, num_features):
    """
    Print a concise model summary.
    
    Args:
        metrics (dict): Evaluation metrics
        best_C (float): Best regularization parameter
        threshold (float): Decision threshold
        split_scheme (str): Description of split scheme used
        cat_features (list): Categorical features
        num_features (list): Numeric features
    """
    print("\n" + "=" * 70)
    print("MODEL SUMMARY")
    print("=" * 70)
    
    print(f"\n📋 Model Configuration:")
    print(f"   - Algorithm: Logistic Regression (L2 regularization)")
    print(f"   - Training Objective: Log loss (cross-entropy)")
    print(f"   - Regularization (C): {best_C:.1f}")
    print(f"   - Class Weighting: Balanced")
    print(f"   - Decision Threshold: {threshold:.3f}")
    
    print(f"\n📊 Features Used:")
    print(f"   - Categorical (one-hot): {', '.join(cat_features)}")
    print(f"   - Numeric: {', '.join(num_features)}")
    
    print(f"\n📈 Performance (Test Set):")
    print(f"   - PR-AUC (Long): {metrics['PR_AUC']:.4f}")
    print(f"   - Brier Score: {metrics['Brier_Score']:.4f}")
    print(f"   - Precision (Long): {metrics['Precision_Long']:.4f}")
    print(f"   - Recall (Long): {metrics['Recall_Long']:.4f}")
    print(f"   - F1 Score (Long): {metrics['F1_Long']:.4f}")
    
    print(f"\n🔍 Data Split:")
    print(f"   - Scheme: {split_scheme}")
    
    print(f"\n💡 Interpretation:")
    pr_auc = metrics['PR_AUC']
    f1 = metrics['F1_Long']
    
    if pr_auc > 0.7 and f1 > 0.6:
        performance = "strong"
    elif pr_auc > 0.6 and f1 > 0.5:
        performance = "moderate"
    else:
        performance = "limited"
    
    print(f"   The model shows {performance} predictive performance for classifying")
    print(f"   train disruptions as Long vs Short. The PR-AUC of {pr_auc:.3f} indicates")
    print(f"   {performance} ability to discriminate between classes, while the F1 score")
    print(f"   of {f1:.3f} suggests {performance} balance between precision and recall.")


def main():
    """
    Main orchestration function to run the complete pipeline.
    """
    print("\n" + "=" * 70)
    print("TRAIN DISRUPTION PREDICTIVE ANALYTICS")
    print("Logistic Regression - Assignment 2")
    print("=" * 70)
    
    # Set up paths
    datasets_path = Path(__file__).parent.parent / 'datasets'
    figures_dir = Path(__file__).parent.parent / 'figures'
    artifacts_dir = Path(__file__).parent.parent / 'artifacts'
    
    # Create output directories
    figures_dir.mkdir(exist_ok=True)
    artifacts_dir.mkdir(exist_ok=True)
    
    # 1. Load data
    df = load_and_combine_data(datasets_path)
    
    # 2. Engineer features
    df = engineer_features(df)
    
    # 3. Compute threshold (use 2021-2024 if available)
    available_years = sorted(df['year'].unique())
    if all(year in available_years for year in [2021, 2022, 2023, 2024]):
        threshold_years = [2021, 2022, 2023, 2024]
    else:
        threshold_years = None
    
    threshold = compute_threshold(df, threshold_years)
    
    # 4. Create target
    df = create_target(df, threshold)
    
    # 5. Split data
    train_df, val_df, test_df, split_scheme = split_data(df)
    
    # Define feature columns
    feature_cols = ['hour_bin', 'weekday', 'month', 'cause_group', 
                    'station_count', 'is_engineering_work']
    
    X_train = train_df[feature_cols]
    y_train = train_df['is_long']
    X_val = val_df[feature_cols]
    y_val = val_df['is_long']
    X_test = test_df[feature_cols]
    y_test = test_df['is_long']
    
    # 6. Build pipeline
    preprocessor, cat_features, num_features = build_pipeline()
    
    # 7. Tune hyperparameters
    best_C, best_model, best_pr_auc_val = tune_hyperparameters(
        X_train, y_train, X_val, y_val, preprocessor
    )
    
    # 8. Select decision threshold
    X_val_transformed = preprocessor.transform(X_val)
    y_val_proba = best_model.predict_proba(X_val_transformed)[:, 1]
    decision_threshold = select_decision_threshold(y_val, y_val_proba)
    
    # 9. Evaluate on test set
    metrics = evaluate_model(best_model, preprocessor, X_test, y_test, 
                            decision_threshold, best_C)
    
    # 10. Generate figures
    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)
    
    plot_figure_1(df, figures_dir / 'fig1_heatmap.png')
    plot_figure_2(df, figures_dir / 'fig2_monthly.png')
    plot_figure_3(df, figures_dir / 'fig3_cause_duration.png')
    plot_figure_4(df, figures_dir / 'fig4_long_share.png')
    plot_figure_5(y_test, metrics['y_test_proba'], metrics['Brier_Score'], 
                  metrics['PR_AUC'], figures_dir / 'fig5_calibration.png')
    
    # 11. Save artifacts
    save_artifacts(metrics, cat_features, num_features, train_df, val_df, test_df,
                   threshold, artifacts_dir)
    
    # 12. Print summary
    print_model_summary(metrics, best_C, decision_threshold, split_scheme, 
                       cat_features, num_features)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE ✅")
    print("=" * 70)
    print(f"\n📁 Output files:")
    print(f"   Figures: {figures_dir}/")
    print(f"   Artifacts: {artifacts_dir}/")
    print("=" * 70)


if __name__ == '__main__':
    main()

