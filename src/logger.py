class Logger:
    def __init__(self, width=70):
        self.width = width

    def section(self, title):
        print("\n" + "=" * self.width)
        print(title)
        print("=" * self.width)

    def info(self, message):
        print(f"   {message}")

    def success(self, message):
        print(f"✅ {message}")

    def files_found(self, files):
        print(f"Found {len(files)} CSV file(s):")
        for f in files:
            print(f"  - {f.name}")

    def file_loaded(self, filename, row_count):
        print(f"  Loaded {filename}: {row_count:,} rows")

    def combined_dataset(self, total_rows, total_cols):
        print(f"\n✅ Combined dataset: {total_rows:,} rows, {total_cols} columns")

    def features_engineered(self, features_dict):
        print("✅ Features engineered:")
        for key, value in features_dict.items():
            print(f"   - {key}: {value}")

    def threshold_info(self, years, row_count):
        if years:
            print(f"Using years {years}: {row_count:,} rows")
        else:
            print(f"Using all available data: {row_count:,} rows")

    def threshold_result(self, threshold):
        print(f"✅ Median duration threshold: {threshold:.1f} minutes")

    def class_distribution(self, threshold, n_long, n_total):
        pct_long = 100 * n_long / n_total
        print(f"\n📊 Class Distribution:")
        print(f"   - Long (≥{threshold:.1f} min): {n_long:,} ({pct_long:.1f}%)")
        print(f"   - Short (<{threshold:.1f} min): {n_total - n_long:,} ({100 - pct_long:.1f}%)")

    def split_info(self, available_years, scheme):
        print(f"Available years: {available_years}")
        if "multi-year" in scheme:
            print("\n✅ Multi-year scheme: Train=2021-2023, Val=2023 Q4, Test=2024")
        else:
            print("\n✅ Single-year scheme: Time-ordered 80/10/10 split")

    def split_sizes(self, train_size, val_size, test_size, train_pct, val_pct, test_pct):
        print(f"\n📊 Split sizes:")
        print(f"   - Train: {train_size:,} rows")
        print(f"   - Validation: {val_size:,} rows")
        print(f"   - Test: {test_size:,} rows")
        print(f"   - Train % Long: {train_pct:.1f}%")
        print(f"   - Val % Long: {val_pct:.1f}%")
        print(f"   - Test % Long: {test_pct:.1f}%")

    def pipeline_built(self, cat_features, num_features):
        print("✅ Pipeline components:")
        print(f"   - Categorical features (one-hot encoded): {cat_features}")
        print(f"   - Numeric features (passthrough): {num_features}")

    def tuning_info(self, c_values):
        print(f"Tuning C ∈ {c_values} using validation PR-AUC...")
        print("Training objective: Log loss (cross-entropy)")

    def tuning_result(self, c_value, pr_auc):
        print(f"   C={c_value:.1f}: PR-AUC={pr_auc:.4f}")

    def best_hyperparameter(self, best_c, best_pr_auc):
        print(f"\n✅ Best C: {best_c:.1f} (Validation PR-AUC: {best_pr_auc:.4f})")

    def threshold_selected(self, threshold, f1):
        print(f"✅ Optimal threshold: {threshold:.3f} (Validation F1: {f1:.4f})")

    def test_metrics(self, metrics):
        print(f"\n📊 Test Set Metrics:")
        print(f"   - PR-AUC (Long): {metrics['PR_AUC']:.4f}")
        print(f"   - Brier Score: {metrics['Brier_Score']:.4f}")
        print(f"   - Precision (Long): {metrics['Precision_Long']:.4f}")
        print(f"   - Recall (Long): {metrics['Recall_Long']:.4f}")
        print(f"   - F1 Score (Long): {metrics['F1_Long']:.4f}")

    def confusion_matrix(self, cm):
        print(f"\n📋 Confusion Matrix:")
        print(f"   True Negatives:  {cm[0, 0]:,}")
        print(f"   False Positives: {cm[0, 1]:,}")
        print(f"   False Negatives: {cm[1, 0]:,}")
        print(f"   True Positives:  {cm[1, 1]:,}")

    def figure_generated(self, figure_name, path):
        print(f"\n📊 Generating {figure_name}...")
        print(f"   ✅ Saved to {path}")

    def artifacts_saved(self, metrics_path, features_path):
        print(f"✅ Saved metrics to {metrics_path}")
        print(f"✅ Saved feature list to {features_path}")

    def model_summary(self, config, features, performance, split_scheme, interpretation):
        print(f"\n📋 Model Configuration:")
        for key, value in config.items():
            print(f"   - {key}: {value}")
        print(f"\n📊 Features Used:")
        print(f"   - Categorical (one-hot): {', '.join(features['categorical'])}")
        print(f"   - Numeric: {', '.join(features['numeric'])}")
        print(f"\n📈 Performance (Test Set):")
        for key, value in performance.items():
            print(f"   - {key}: {value:.4f}")
        print(f"\n🔍 Data Split:")
        print(f"   - Scheme: {split_scheme}")
        print(f"\n💡 Interpretation:")
        print(interpretation)

    def completion(self, figures_dir, artifacts_dir):
        print("\n" + "=" * self.width)
        print("ANALYSIS COMPLETE ✅")
        print("=" * self.width)
        print(f"\n📁 Output files:")
        print(f"   Figures: {figures_dir}/")
        print(f"   Artifacts: {artifacts_dir}/")
        print("=" * self.width)

