import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc, brier_score_loss, classification_report, confusion_matrix, f1_score
from sklearn.calibration import calibration_curve
from logger import Logger


class ModelPipeline:
    def __init__(self, random_state=42, logger=None):
        self.random_state = random_state
        self.logger = logger or Logger()
        self.cat_features = ['hour_bin', 'weekday', 'month', 'cause_group']
        self.num_features = ['station_count', 'is_engineering_work']
        self.preprocessor = None

    def build_preprocessor(self):
        self.logger.section("BUILDING PIPELINE")
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), self.cat_features),
                ('num', 'passthrough', self.num_features),
            ],
            verbose_feature_names_out=False,
        )
        self.logger.pipeline_built(self.cat_features, self.num_features)
        return self.preprocessor

    def tune(self, X_train, y_train, X_val, y_val):
        self.logger.section("HYPERPARAMETER TUNING")
        C_values = [0.5, 1.0, 2.0]
        results = []
        self.logger.tuning_info(C_values)

        X_train_transformed = self.preprocessor.fit_transform(X_train)
        X_val_transformed = self.preprocessor.transform(X_val)

        for C in C_values:
            model = LogisticRegression(penalty='l2', C=C, class_weight='balanced', max_iter=1000, random_state=self.random_state)
            model.fit(X_train_transformed, y_train)
            y_val_proba = model.predict_proba(X_val_transformed)[:, 1]
            precision, recall, _ = precision_recall_curve(y_val, y_val_proba)
            pr_auc = auc(recall, precision)
            results.append({'C': C, 'PR_AUC': pr_auc, 'model': model, 'y_val_proba': y_val_proba})
            self.logger.tuning_result(C, pr_auc)

        best = max(results, key=lambda x: x['PR_AUC'])
        best_C = best['C']
        best_model = best['model']
        best_pr_auc = best['PR_AUC']
        self.logger.best_hyperparameter(best_C, best_pr_auc)
        return best_C, best_model, best_pr_auc

    def select_threshold(self, y_val, y_val_proba):
        self.logger.section("DECISION THRESHOLD SELECTION")
        thresholds = np.arange(0.1, 0.9, 0.01)
        f1_scores = []
        for t in thresholds:
            y_pred = (y_val_proba >= t).astype(int)
            f1 = f1_score(y_val, y_pred, pos_label=1)
            f1_scores.append(f1)
        idx = np.argmax(f1_scores)
        best_threshold = thresholds[idx]
        best_f1 = f1_scores[idx]
        self.logger.threshold_selected(best_threshold, best_f1)
        return best_threshold

    def evaluate(self, model, X_test, y_test, threshold):
        self.logger.section("MODEL EVALUATION (TEST SET)")
        X_test_transformed = self.preprocessor.transform(X_test)
        y_test_proba = model.predict_proba(X_test_transformed)[:, 1]
        y_test_pred = (y_test_proba >= threshold).astype(int)
        precision, recall, _ = precision_recall_curve(y_test, y_test_proba)
        pr_auc = auc(recall, precision)
        brier = brier_score_loss(y_test, y_test_proba)
        report = classification_report(y_test, y_test_pred, output_dict=True, target_names=['Short', 'Long'])
        cm = confusion_matrix(y_test, y_test_pred)
        precision_long = report['Long']['precision']
        recall_long = report['Long']['recall']
        f1_long = report['Long']['f1-score']
        
        metrics = {
            'PR_AUC': pr_auc,
            'Brier_Score': brier,
            'Precision_Long': precision_long,
            'Recall_Long': recall_long,
            'F1_Long': f1_long,
            'Decision_Threshold': threshold,
            'Confusion_Matrix': cm,
            'y_test_proba': y_test_proba,
            'y_test_pred': y_test_pred,
        }
        
        self.logger.test_metrics(metrics)
        self.logger.confusion_matrix(cm)
        return metrics


