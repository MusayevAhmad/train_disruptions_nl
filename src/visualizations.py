import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from logger import Logger


class Visualizations:
    def __init__(self, figures_dir, artifacts_dir, logger=None):
        self.figures_dir = Path(figures_dir)
        self.artifacts_dir = Path(artifacts_dir)
        self.logger = logger or Logger()
        self.figures_dir.mkdir(exist_ok=True)
        self.artifacts_dir.mkdir(exist_ok=True)

    def generate_figures(self, df, y_test, metrics):
        self.logger.section("GENERATING FIGURES")
        self.plot_figure_1(df, self.figures_dir / 'fig1_heatmap.png')
        self.plot_figure_2(df, self.figures_dir / 'fig2_monthly.png')
        self.plot_figure_3(df, self.figures_dir / 'fig3_cause_duration.png')
        self.plot_figure_4(df, self.figures_dir / 'fig4_long_share.png')
        self.plot_figure_5(y_test, metrics['y_test_proba'], metrics['Brier_Score'], metrics['PR_AUC'], self.figures_dir / 'fig5_calibration.png')

    def save_artifacts(self, metrics, cat_features, num_features, train_df, val_df, test_df, threshold):
        self.logger.section("SAVING ARTIFACTS")
        metrics_df = pd.DataFrame({
            'Metric': [
                'PR_AUC', 'Brier_Score', 'Precision_Long', 'Recall_Long', 'F1_Long',
                'Decision_Threshold', 'Duration_Threshold_Minutes',
                'Train_Pct_Long', 'Val_Pct_Long', 'Test_Pct_Long',
                'Train_Size', 'Val_Size', 'Test_Size',
            ],
            'Value': [
                metrics['PR_AUC'],
                metrics['Brier_Score'],
                metrics['Precision_Long'],
                metrics['Recall_Long'],
                metrics['F1_Long'],
                metrics['Decision_Threshold'],
                threshold,
                100 * train_df['is_long'].mean(),
                100 * val_df['is_long'].mean(),
                100 * test_df['is_long'].mean(),
                len(train_df),
                len(val_df),
                len(test_df),
            ],
        })
        metrics_path = self.artifacts_dir / 'metrics_test.csv'
        metrics_df.to_csv(metrics_path, index=False)
        feature_list_path = self.artifacts_dir / 'feature_list.txt'
        with open(feature_list_path, 'w') as f:
            f.write("CATEGORICAL FEATURES (one-hot encoded):\n")
            for feat in cat_features:
                f.write(f"  - {feat}\n")
            f.write("\nNUMERIC FEATURES (passthrough):\n")
            for feat in num_features:
                f.write(f"  - {feat}\n")
        self.logger.artifacts_saved(metrics_path, feature_list_path)

    def print_summary(self, metrics, best_C, threshold, split_scheme, cat_features, num_features):
        self.logger.section("MODEL SUMMARY")
        pr_auc = metrics['PR_AUC']
        f1 = metrics['F1_Long']
        if pr_auc > 0.7 and f1 > 0.6:
            performance = "strong"
        elif pr_auc > 0.6 and f1 > 0.5:
            performance = "moderate"
        else:
            performance = "limited"
        
        interpretation = (
            f"   The model shows {performance} predictive performance for classifying\n"
            f"   train disruptions as Long vs Short. The PR-AUC of {pr_auc:.3f} indicates\n"
            f"   {performance} ability to discriminate between classes, while the F1 score\n"
            f"   of {f1:.3f} suggests {performance} balance between precision and recall."
        )
        
        self.logger.model_summary(
            config={
                'Algorithm': 'Logistic Regression (L2 regularization)',
                'Training Objective': 'Log loss (cross-entropy)',
                'Regularization (C)': f'{best_C:.1f}',
                'Class Weighting': 'Balanced',
                'Decision Threshold': f'{threshold:.3f}',
            },
            features={
                'categorical': cat_features,
                'numeric': num_features,
            },
            performance={
                'PR-AUC (Long)': metrics['PR_AUC'],
                'Brier Score': metrics['Brier_Score'],
                'Precision (Long)': metrics['Precision_Long'],
                'Recall (Long)': metrics['Recall_Long'],
                'F1 Score (Long)': metrics['F1_Long'],
            },
            split_scheme=split_scheme,
            interpretation=interpretation,
        )

    def plot_figure_1(self, df, output_path):
        heatmap_data = df.groupby(['hour', 'weekday']).size().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(heatmap_data.values, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(range(7))
        ax.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        ax.set_yticks(range(24))
        ax.set_yticklabels(range(24))
        ax.set_xlabel('Weekday', fontsize=12)
        ax.set_ylabel('Hour', fontsize=12)
        ax.set_title('Disruptions by Hour × Weekday', fontsize=14, fontweight='bold')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Count', rotation=270, labelpad=20)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.figure_generated("Figure 1: Hour × Weekday Heatmap", output_path)

    def plot_figure_2(self, df, output_path):
        monthly_data = df.groupby(['year', 'month']).size().reset_index(name='count')
        fig, ax = plt.subplots(figsize=(12, 6))
        years = sorted(df['year'].unique())
        for year in years:
            year_data = monthly_data[monthly_data['year'] == year]
            ax.plot(year_data['month'], year_data['count'], marker='o', label=str(year), linewidth=2, markersize=6)
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Monthly disruptions by year', fontsize=14, fontweight='bold')
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax.grid(True, alpha=0.3)
        ax.legend(title='Year', loc='best')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.figure_generated("Figure 2: Monthly Trend by Year", output_path)

    def plot_figure_3(self, df, output_path):
        p99 = df['duration_minutes'].quantile(0.99)
        df_plot = df.copy()
        df_plot = df_plot[df_plot['duration_minutes'].notna()]
        df_plot = df_plot[~df_plot['cause_group'].isin(['unknown', 'other'])]

        grouped = df_plot.groupby('cause_group')['duration_minutes']
        data_dict = {cause: grouped.get_group(cause).clip(upper=p99).values for cause in grouped.groups.keys()}
        data_dict = {cause: values for cause, values in data_dict.items() if len(values) > 0}

        if not data_dict:
            fig, ax = plt.subplots(figsize=(12, 3))
            ax.text(0.5, 0.5, 'No data available for selected cause groups', ha='center', va='center')
            ax.axis('off')
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            self.logger.figure_generated("Figure 3: Duration by Cause Group", output_path)
            return

        cause_order = sorted(data_dict.keys(), key=lambda c: np.median(data_dict[c]))
        data_by_cause = [data_dict[cause] for cause in cause_order]

        fig, ax = plt.subplots(figsize=(12, 6))
        bp = ax.boxplot(data_by_cause, labels=cause_order, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        ax.set_xlabel('Cause Group', fontsize=12)
        ax.set_ylabel('Duration (minutes)', fontsize=12)
        ax.set_title('Duration distribution by cause group (minutes)', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.figure_generated("Figure 3: Duration by Cause Group", output_path)

    def plot_figure_4(self, df, output_path):
        pct_long = df.groupby('cause_group')['is_long'].mean() * 100
        pct_long = pct_long.sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(pct_long)), pct_long.values, color='steelblue', alpha=0.7)
        ax.set_yticks(range(len(pct_long)))
        ax.set_yticklabels(pct_long.index)
        ax.set_xlabel('% Long disruptions', fontsize=12)
        ax.set_ylabel('Cause Group', fontsize=12)
        ax.set_title('% of Long disruptions by cause group (≥ median threshold)', fontsize=14, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)
        for i, v in enumerate(pct_long.values):
            ax.text(v + 1, i, f'{v:.1f}%', va='center', fontsize=9)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.figure_generated("Figure 4: % Long by Cause Group", output_path)

    def plot_figure_5(self, y_test, y_test_proba, brier, pr_auc, output_path):
        from sklearn.calibration import calibration_curve
        prob_true, prob_pred = calibration_curve(y_test, y_test_proba, n_bins=10)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(prob_pred, prob_true, marker='o', linewidth=2, markersize=8, label='Logistic Regression', color='steelblue')
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2, label='Perfect Calibration')
        ax.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax.set_ylabel('Fraction of Positives', fontsize=12)
        ax.set_title(f'Calibration (test) — Brier = {brier:.3f}', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.text(0.95, 0.05, f'PR-AUC = {pr_auc:.3f}', transform=ax.transAxes, fontsize=11, verticalalignment='bottom', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.figure_generated("Figure 5: Calibration Curve", output_path)


