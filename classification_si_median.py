from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler


def load_and_prepare_data(file_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_excel(file_path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    median_si = df["SI"].median()
    y = (df["SI"] > median_si).astype(int)
    feature_cols = [c for c in df.columns if c not in ["IC50, mM", "CC50, mM", "SI"]]
    X = df[feature_cols]
    return X, y


def build_models(random_state: int = 42) -> Dict[str, Tuple[Pipeline, Dict[str, List]]]:
    models: Dict[str, Tuple[Pipeline, Dict[str, List]]] = {}
    # Logistic regression
    log_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=500, random_state=random_state))
    ])
    log_grid = {
        'model__C': [0.1, 1.0, 10.0],
        'model__class_weight': [None, 'balanced']
    }
    models['LogisticRegression'] = (log_pipe, log_grid)
    # Random Forest
    rf_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', RandomForestClassifier(random_state=random_state))
    ])
    rf_grid = {
        'model__n_estimators': [200, 400],
        'model__max_depth': [None, 10],
        'model__min_samples_split': [2],
        'model__class_weight': [None, 'balanced']
    }
    models['RandomForest'] = (rf_pipe, rf_grid)
    # Gradient Boosting
    gb_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', GradientBoostingClassifier(random_state=random_state))
    ])
    gb_grid = {
        'model__n_estimators': [200],
        'model__learning_rate': [0.05, 0.1],
        'model__max_depth': [3, 5]
    }
    models['GradientBoosting'] = (gb_pipe, gb_grid)
    # XGBoost classifier
    xgb_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', XGBClassifier(
            random_state=random_state,
            objective='binary:logistic',
            eval_metric='logloss',
            tree_method='hist',
            n_jobs=4
        ))
    ])
    xgb_grid = {
        'model__n_estimators': [400],
        'model__max_depth': [3, 6],
        'model__learning_rate': [0.05, 0.1],
        'model__subsample': [0.8],
        'model__scale_pos_weight': [1]
    }
    models['XGBoost'] = (xgb_pipe, xgb_grid)
    return models


def evaluate_models(models: Dict[str, Tuple[Pipeline, Dict[str, List]]], X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }
    results = []
    for name, (pipe, grid) in models.items():
        print(f"\n--- Optimising {name} ---")
        gcv = GridSearchCV(
            pipe,
            grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            error_score='raise'
        )
        gcv.fit(X, y)
        print(f"Best parameters for {name}: {gcv.best_params_}")
        best = gcv.best_estimator_
        cv_scores = cross_validate(best, X, y, cv=5, scoring=scoring, n_jobs=-1)
        results.append({
            'Model': name,
            'Accuracy (mean)': cv_scores['test_accuracy'].mean(),
            'Accuracy (std)': cv_scores['test_accuracy'].std(),
            'Precision (mean)': cv_scores['test_precision'].mean(),
            'Precision (std)': cv_scores['test_precision'].std(),
            'Recall (mean)': cv_scores['test_recall'].mean(),
            'Recall (std)': cv_scores['test_recall'].std(),
            'F1 (mean)': cv_scores['test_f1'].mean(),
            'F1 (std)': cv_scores['test_f1'].std(),
            'ROC_AUC (mean)': cv_scores['test_roc_auc'].mean(),
            'ROC_AUC (std)': cv_scores['test_roc_auc'].std()
        })
    results_df = pd.DataFrame(results).sort_values(by='ROC_AUC (mean)', ascending=False)
    return results_df


def main() -> None:
    data_path = os.path.join(os.path.dirname(__file__), 'data.xlsx')
    X, y = load_and_prepare_data(data_path)
    models = build_models()
    results_df = evaluate_models(models, X, y)
    print("\n===== Cross‑validated performance summary =====")
    with pd.option_context('display.max_columns', None):
        print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    results_df.to_csv('classification_si_median_results.csv', index=False)


if __name__ == '__main__':
    main()