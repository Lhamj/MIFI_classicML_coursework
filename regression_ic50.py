from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor


def load_and_prepare_data(file_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_excel(file_path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    feature_cols = [c for c in df.columns if c not in ["IC50, mM", "CC50, mM", "SI"]]
    X = df[feature_cols]
    y = np.log10(df["IC50, mM"])
    return X, y


def build_models(random_state: int = 42) -> Dict[str, Tuple[Pipeline, Dict[str, List]]]:
    models: Dict[str, Tuple[Pipeline, Dict[str, List]]] = {}

    # Ridge regression pipeline
    ridge_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', Ridge())
    ])
    ridge_grid = {
        'model__alpha': [1.0, 10.0],
        'model__solver': ['auto']
    }
    models['Ridge'] = (ridge_pipe, ridge_grid)

    # Lasso regression pipeline
    lasso_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', Lasso(random_state=random_state, max_iter=10000))
    ])
    lasso_grid = {
        'model__alpha': [0.001, 0.01, 0.1],
        'model__selection': ['cyclic']
    }
    models['Lasso'] = (lasso_pipe, lasso_grid)

    # Random Forest Regressor pipeline
    rf_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', RandomForestRegressor(random_state=random_state))
    ])
    rf_grid = {
        'model__n_estimators': [200, 400],
        'model__max_depth': [None, 10],
        'model__min_samples_split': [2]
    }
    models['RandomForest'] = (rf_pipe, rf_grid)

    # Gradient Boosting Regressor pipeline
    gbr_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', GradientBoostingRegressor(random_state=random_state))
    ])
    gbr_grid = {
        'model__n_estimators': [200],
        'model__learning_rate': [0.05, 0.1],
        'model__max_depth': [3, 5]
    }
    models['GradientBoosting'] = (gbr_pipe, gbr_grid)

    # XGBoost Regressor pipeline
    xgb_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', XGBRegressor(
            random_state=random_state,
            objective='reg:squarederror',
            eval_metric='rmse',
            tree_method='hist',
            n_jobs=4
        ))
    ])
    xgb_grid = {
        'model__n_estimators': [400],
        'model__max_depth': [3, 6],
        'model__learning_rate': [0.05, 0.1],
        'model__subsample': [0.8]
    }
    models['XGBoost'] = (xgb_pipe, xgb_grid)

    return models


def evaluate_models(models: Dict[str, Tuple[Pipeline, Dict[str, List]]], X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    results = []
    scoring = {
        'rmse': 'neg_root_mean_squared_error',
        'mae': 'neg_mean_absolute_error',
        'r2': 'r2'
    }
    for name, (pipe, param_grid) in models.items():
        print(f"\n--- Optimising {name} ---")
        grid = GridSearchCV(
            pipe,
            param_grid,
            cv=5,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1,
            error_score='raise'
        )
        grid.fit(X, y)
        print(f"Best parameters for {name}: {grid.best_params_}")
        best_model = grid.best_estimator_
        cv_scores = cross_validate(best_model, X, y, cv=5, scoring=scoring, n_jobs=-1)
        result = {
            'Model': name,
            'RMSE (mean)': -cv_scores['test_rmse'].mean(),
            'RMSE (std)': cv_scores['test_rmse'].std(),
            'MAE (mean)': -cv_scores['test_mae'].mean(),
            'MAE (std)': cv_scores['test_mae'].std(),
            'R2 (mean)': cv_scores['test_r2'].mean(),
            'R2 (std)': cv_scores['test_r2'].std()
        }
        results.append(result)
    results_df = pd.DataFrame(results).sort_values(by='RMSE (mean)')
    return results_df


def main() -> None:
    data_path = os.path.join(os.path.dirname(__file__), 'data.xlsx')
    X, y = load_and_prepare_data(data_path)
    models = build_models()
    results_df = evaluate_models(models, X, y)
    print("\n===== Cross‑validated performance summary =====")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    results_df.to_csv('regression_ic50_results.csv', index=False)


if __name__ == '__main__':
    main()

