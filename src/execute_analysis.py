from pathlib import Path
import numpy as np
np.random.seed(42)

from logger import Logger
from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from model_pipeline import ModelPipeline
from visualizations import Visualizations


def main():
    logger = Logger()
    logger.section("TRAIN DISRUPTION PREDICTIVE ANALYTICS\nLogistic Regression - Assignment 2")

    root = Path(__file__).parent.parent
    datasets_path = root / 'datasets'
    figures_dir = root / 'figures'
    artifacts_dir = root / 'artifacts'

    loader = DataLoader(datasets_path, logger)
    df = loader.load_data()

    fe = FeatureEngineer(logger)
    df = fe.engineer(df)
    threshold = fe.compute_threshold(df)
    df = fe.create_target(df, threshold)
    train_df, val_df, test_df, split_scheme = fe.split(df)

    feature_cols = ['hour_bin', 'weekday', 'month', 'cause_group', 'station_count', 'is_engineering_work']
    X_train = train_df[feature_cols]
    y_train = train_df['is_long']
    X_val = val_df[feature_cols]
    y_val = val_df['is_long']
    X_test = test_df[feature_cols]
    y_test = test_df['is_long']

    pipeline = ModelPipeline(random_state=42, logger=logger)
    pipeline.build_preprocessor()
    best_C, model, _ = pipeline.tune(X_train, y_train, X_val, y_val)
    y_val_proba = model.predict_proba(pipeline.preprocessor.transform(X_val))[:, 1]
    decision_threshold = pipeline.select_threshold(y_val, y_val_proba)
    metrics = pipeline.evaluate(model, X_test, y_test, decision_threshold)

    viz = Visualizations(figures_dir, artifacts_dir, logger)
    viz.generate_figures(df, y_test, metrics)
    viz.save_artifacts(metrics, pipeline.cat_features, pipeline.num_features, train_df, val_df, test_df, threshold)
    viz.print_summary(metrics, best_C, decision_threshold, split_scheme, pipeline.cat_features, pipeline.num_features)

    logger.completion(figures_dir, artifacts_dir)


if __name__ == '__main__':
    main()


