from IPython.display import display
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    Ridge,  
    Lasso,  
    ElasticNet,
    BayesianRidge,
    SGDClassifier
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,  
    GradientBoostingRegressor,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score,
    f1_score, 
    roc_auc_score, 
    roc_curve, 
    auc,
    mean_squared_error, 
    r2_score
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import os
import warnings

### >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
###                     общая часть реализации кода
### <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# Настройка окружения и предупреждений
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  

# Отключение предупреждений
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def print_results(results, params, models, columns, report_name):
    """
    Вывод результатов обучения моделей в виде таблицы.
    """
    rows = []  # Список для хранения строк таблицы

    for mode in results.keys():  
        for model_name in models.keys():  
            best_params = params[mode][model_name] or 'default'  
            res = results[mode][model_name]  
            # Создание строки данных
            row = {
                columns[0]: model_name,  # Название модели
                columns[1]: mode,  # Метод редукции
                columns[2]: str(best_params)  # Лучшие параметры
            }
            # Добавление метрик в строку
            for i, metric in enumerate(columns[3:], start=3):  # Начинаем с 4-го столбца (метрики)
                row[metric] = res[metric.lower().replace(" ", "_")]  # Преобразуем название метрики к формату словаря
            rows.append(row)  # Добавляем строку в список
    # Создание DataFrame из списка строк
    df = pd.DataFrame(rows, columns=columns)
    # Сортировка таблицы по модели и методу редукции
    df = df.sort_values(by=[columns[0], columns[1]], ascending=[True, True])
    df.to_csv(report_name, index=False)
    # Вывод таблицы
    display(df)

def plot_metrics(results, roc_curves, models, metrics):
    """
    Визуализация результатов: метрики и ROC-кривые.
    """
    model_names = list(models.keys())
    modes = list(results.keys())
    
    # Графики метрик
    fig, axes = plt.subplots(1, len(metrics), figsize=(23, 7)) 
    if len(metrics) == 1:
        axes = [axes]

    bar_height = 0.25

    for i, metric in enumerate(metrics):
        y_pos = np.arange(len(model_names))
        for j, mode in enumerate(modes):
            values = [results[mode][m][metric] for m in model_names]
            bars = axes[i].barh(
                y_pos + (j - 1) * bar_height, values, height=bar_height, color=['#ADD8E6', '#FF8559', '#7FFFD4'][j],
                label=mode, alpha=0.85, edgecolor='black'
            )
            for idx, bar in enumerate(bars):
                axes[i].text(
                    bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{bar.get_width():.2f}",
                    va='center', ha='left', fontsize=11, color='black'
                )
        axes[i].set_yticks(y_pos)
        axes[i].set_yticklabels([f"{name}" for name in model_names], fontsize=12)        
        axes[i].set_xlabel(metric, fontsize=13)
        axes[i].set_title(f'Сравнение по метрике: {metric}', fontsize=15, fontweight='bold')
        axes[i].xaxis.set_major_locator(mticker.MaxNLocator(5))
        axes[i].grid(axis='x', linestyle='--', alpha=0.5)
        if i == 0:
            handles = [mpatches.Patch(color=['#ADD8E6', '#FF8559', '#7FFFD4'][k], label=modes[k]) for k in range(len(modes))]
            axes[i].legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=3, frameon=False, fontsize=12)

    plt.tight_layout()
    plt.show()
    
    # ROC-кривые
    plt.figure(figsize=(20, 10))  
    
    # Добавляем общий заголовок
    plt.suptitle('Сравнение ROC-кривых для различных режимов', fontsize=16, fontweight='bold')
    
    for j, mode in enumerate(modes):
        plt.subplot(1, 3, j + 1)
        for i, name in enumerate(model_names):
            fpr, tpr, roc_auc_val = roc_curves[mode][name]
            plt.plot(fpr, tpr, label=f'{name} (AUC={roc_auc_val:.2f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', lw=1)  # Диагональная линия
        plt.title(f'{mode}', fontsize=12, fontweight='bold')
        plt.xlabel('Частота ложноположительных результатов', fontsize=12)
        plt.ylabel('Частота истинно положительных результатов', fontsize=12)
        
        # Размещаем легенду под графиком в одну колонку
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fontsize=11, ncol=1)
        
        plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout(rect=[0, 0, 1, 0.9])  
    plt.show()

### >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
###                     код связанный с решением задач классификации
### <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

def prepare_classification_data(df, target_col, reducers, test_size=0.2, random_state=42):
    """
    Подготовка данных: разделение на признаки и целевую переменную,
    применение методов снижения размерности.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    datasets = {}
    for method, reducer in reducers.items():
        X_reduced = X if reducer is None else reducer.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_reduced, y, test_size=test_size, random_state=random_state, stratify=y
        )
        datasets[method] = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }
    return datasets


def train_classifier_models(datasets, models, param_grids):
    """
    Обучение моделей с использованием GridSearchCV.
    Возвращает метрики, ROC-кривые и лучшие параметры.
    """
    results = {method: {} for method in datasets}
    roc_curves = {method: {} for method in datasets}
    best_params = {method: {} for method in datasets}

    for method, data in datasets.items():
        for model_name, model in models.items():
            grid_search = GridSearchCV(model, param_grids[model_name], cv=3, scoring='accuracy', n_jobs=-1)
            grid_search.fit(data['X_train'], data['y_train'])

            best_model = grid_search.best_estimator_
            best_params[method][model_name] = grid_search.best_params_

            # Предсказания и вероятности
            y_pred = best_model.predict(data['X_test'])
            y_proba = (
                best_model.predict_proba(data['X_test'])[:, 1]
                if hasattr(best_model, "predict_proba")
                else best_model.decision_function(data['X_test'])
            )

            # Метрики
            results[method][model_name] = {
                'accuracy': accuracy_score(data['y_test'], y_pred),
                'precision': precision_score(data['y_test'], y_pred),
                'recall': recall_score(data['y_test'], y_pred),
                'f1': f1_score(data['y_test'], y_pred),
                'roc_auc': roc_auc_score(data['y_test'], y_proba)
            }

            # ROC-кривая
            fpr, tpr, _ = roc_curve(data['y_test'], y_proba)
            roc_auc_val = auc(fpr, tpr)
            roc_curves[method][model_name] = (fpr, tpr, roc_auc_val)

    return results, roc_curves, best_params


def print_classification_results(results, params, models, columns):
    """
    Вывод результатов обучения моделей в виде таблицы 
    """
    df = pd.DataFrame(columns=columns)
    rows = []  
    for mode in results.keys():
        for model_name in models.keys():
            best_params = params[mode][model_name] or 'default'
            res = results[mode][model_name]
            row = {
                'Модель': model_name,  
                'Метод редукции': mode,  
                'Лучшие параметры': str(best_params),  
                "Accuracy": res['accuracy'],
                "Precision": res['precision'],
                "Recall": res['recall'],
                "F1": res['f1'],
                "ROC AUC": res['roc_auc']
            }
            rows.append(row)
    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    df = df.sort_values(by=['Модель', 'Метод редукции'], ascending=[True, True])
    display(df)

def train_and_compare_classifications(data_file, target_col='threshold_exceeded', test_size=0.2, random_state=42, n_components=15):
    """
    Основная функция для обучения и сравнения моделей.
    """
    df = pd.read_csv(data_file)

    
    # Определение моделей
    models = {
        'Logistic regression': LogisticRegression(max_iter=1000),
        'Decision tree': DecisionTreeClassifier(random_state=42),
        'Random forest': RandomForestClassifier(random_state=42),
        'CatBoost': CatBoostClassifier(verbose=0, random_state=42),
        'MLP': MLPClassifier(random_state=42),
        'SVC': SVC(probability=True, random_state=42),
        'SGD': SGDClassifier(random_state=42),
        'Gradient boosting': GradientBoostingClassifier(random_state=42),
        'AdaBoost': AdaBoostClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42, verbosity=0)
    }
    
    # Параметры для GridSearch
    param_grids = {
        'Logistic regression': {
            'C': [0.1, 1, 10], 
            'solver': ['liblinear', 'lbfgs']
        },
        'Decision tree': {
            'max_depth': [3, 5, 7, 10, None],
            'criterion': ['gini', 'entropy']
        },
        'Random forest': {
            'n_estimators': [50, 100],
            'max_depth': [5, 10, None],
            'max_features': ['sqrt', 'log2', None]
        },
        'CatBoost': {
            'depth': [4, 6, 8],
            'learning_rate': [0.01, 0.1],
            'iterations': [100, 500]
        },
        'MLP': {
            'hidden_layer_sizes': [(100,), (100, 50)],
            'activation': ['relu', 'tanh'],
            'max_iter': [500, 1000]
        },
        'SVC': {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear']
        },
        'SGD': {
            'loss': ['hinge', 'log_loss', 'modified_huber'],
            'penalty': ['l2', 'l1', 'elasticnet'],
            'alpha': [0.0001, 0.001, 0.01],
            'max_iter': [1000, 5000]
        },
        'Gradient boosting': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 10],
            'subsample': [0.8, 0.9, 1.0]
        },
        'AdaBoost': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 1.0]
        },
        'XGBoost': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 10],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
    }

    
    # Методы снижения размерности
    reducers = {
        'NoReduction': None,
        'PCA': PCA(n_components=n_components, random_state=random_state),
        'UMAP': UMAP(n_components=n_components, random_state=random_state)
    }
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    columns = ['Модель', 'Метод редукции', 'Лучшие параметры', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC']
    # Подготовка данных
    datasets = prepare_classification_data(df, target_col, reducers, test_size, random_state)
    # Обучение моделей
    results, roc_curves, params = train_classifier_models(datasets, models, param_grids)
    # Вывод результатов
    file_name = os.path.splitext(os.path.basename(data_file))[0]
    report_name = f'''./data/{file_name}_report.csv'''
    print_results(results, params, models, columns, report_name)
    # Визуализация метрик и ROC-кривых
    plot_metrics(results, roc_curves, models, metrics)


### >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
###                     код связанный с решением задач регрессии
### <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


def preprocess_regression_data(df, target_col, test_size=0.2, random_state=42, n_components=15):
    """
    Разделение данных и применение методов снижения размерности.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    y_bin = (y > y.median()).astype(int)
    reducers = {
        'No Reduction': None,
        'PCA': PCA(n_components=n_components, random_state=random_state),
        'UMAP': UMAP(n_components=n_components, random_state=random_state, n_jobs=1)
    }
    data_dict = {}
    for key, reducer in reducers.items():
        X_red = X if reducer is None else reducer.fit_transform(X)
        X_train, X_test, y_train, y_test, y_bin_train, y_bin_test = train_test_split(
                X_red, y, y_bin, test_size=test_size, random_state=random_state)
        data_dict[key] = {
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test,
            'y_bin_train': y_bin_train, 'y_bin_test': y_bin_test
        }
    return data_dict


def evaluate_model(model, param_grids, X_train, X_test, y_train, y_test, y_bin_test):
    """
    Обучение модели и оценка метрик.
    """
    param_grid = param_grids.get(type(model).__name__, {})
    gs = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    gs.fit(X_train, y_train)
    best_model = gs.best_estimator_
    y_pred = best_model.predict(X_test)
    y_pred_bin = (y_pred > np.median(y_train)).astype(int)

    metrics = {
        'mse': mean_squared_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'accuracy': accuracy_score(y_bin_test, y_pred_bin),
        'recall': recall_score(y_bin_test, y_pred_bin),
        'roc_auc': roc_auc_score(y_bin_test, y_pred) if len(np.unique(y_bin_test)) > 1 else np.nan
    }

    fpr, tpr, _ = roc_curve(y_bin_test, y_pred)
    roc_auc_val = auc(fpr, tpr)

    return metrics, gs.best_params_, (fpr, tpr, roc_auc_val)


def train_regression_models(data_dict, models, param_grids):
    """
    Сравнение моделей по разным методам снижения размерности.
    """
    results, params, roc_curves = {}, {}, {}

    for mode, data in data_dict.items():
        results[mode], params[mode], roc_curves[mode] = {}, {}, {}
        for name, model in models.items():
            metrics, best_params, roc_curve_data = evaluate_model(
                model, param_grids, data['X_train'], data['X_test'], data['y_train'], data['y_test'], data['y_bin_test'])
            results[mode][name] = metrics
            params[mode][name] = best_params
            roc_curves[mode][name] = roc_curve_data

    return results, params, roc_curves
    
def train_and_compare_regressions(data_file, target_col, test_size=0.2, random_state=42, n_components=15):
    """
    Главная функция для обучения и сравнения моделей.
    """
    df = pd.read_csv(data_file)
    # Определение моделей
    models = {
        "Linear": LinearRegression(),  
        "Ridge": Ridge(alpha=1.0),  
        "Lasso": Lasso(alpha=0.1),  
        "ElasticNet": ElasticNet(alpha=1.0, l1_ratio=0.5),  
        "Bayesian Ridge": BayesianRidge(),  
        "Decision Tree": DecisionTreeRegressor(random_state=42),  
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),  
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "SVR": SVR(kernel='rbf'),  
        "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5), 
        'CatBoost': CatBoostRegressor(verbose=0, random_state=random_state),
        'MLP': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=2000, random_state=random_state)
    }
    
    param_grids = {
        'Ridge': { 
            'alpha': [0.1, 1.0, 10.0], 
            'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
        },
        'Lasso': { 
            'alpha': [0.1, 1.0, 10.0], 
            'max_iter': [1000, 5000, 10000], 
            'tol': [1e-4, 1e-3, 1e-2] 
        },
        'ElasticNet': { 
            'alpha': [0.1, 1.0, 10.0], 
            'l1_ratio': [0.1, 0.5, 0.9], 
            'max_iter': [1000, 5000, 10000], 
            'tol': [1e-4, 1e-3, 1e-2]
        },
        'BayesianRidge': { 
            'alpha_1': [1e-6, 1e-5, 1e-4], 
            'alpha_2': [1e-6, 1e-5, 1e-4], 
            'lambda_1': [1e-6, 1e-5, 1e-4], 
            'lambda_2': [1e-6, 1e-5, 1e-4], 
            'max_iter': [100, 300, 500]  
        },
        'GradientBoostingRegressor': { 
            'n_estimators': [50, 100, 200], 
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 10], 
            'subsample': [0.8, 0.9, 1.0], 
            'min_samples_split': [2, 5, 10], 
            'min_samples_leaf': [1, 2, 4]
        },
        'KNeighborsRegressor': { 
            'n_neighbors': [3, 5, 7, 10], 
            'leaf_size': [10, 30, 50], 
            'p': [1, 2],
            'weights': ['uniform', 'distance'],  
            'algorithm': ['ball_tree', 'kd_tree', 'brute'] 
        },
        'LinearRegression': {
           # Линейная регрессия не имеет гиперпараметров для настройки 
        }, 
        'DecisionTreeRegressor': {
            'max_depth': [3, 5, 7, 10, None]
        },
        'RandomForestRegressor': {
            'n_estimators': [50, 100], 
            'max_depth': [5, 10, None]
        },
        'CatBoostRegressor': {
            'depth': [4, 6, 8], 
            'learning_rate': [0.01, 0.1]
        },
        'MLPRegressor': {
            'hidden_layer_sizes': [(100,), (100, 50)], 
            'max_iter': [500, 1000]
        },
        'SVR': {
            'C': [0.1, 1, 10], 
            'kernel': ['rbf', 'linear']
        }
    }

    
    metrics = ['accuracy', 'recall', 'roc_auc', 'r2']
    columns = ['Модель', 'Метод редукции', 'Лучшие параметры', 'Accuracy', 'Recall', 'ROC AUC', 'R2']
    # Препроцессинг данных
    data_dict = preprocess_regression_data(df, target_col, test_size, random_state, n_components)
    # Сравнение моделей
    results, params, roc_curves = train_regression_models(data_dict, models, param_grids)
    # Вывод результатов
    file_name = os.path.splitext(os.path.basename(data_file))[0]
    report_name = f'''./data/{file_name}_report.csv'''
    print_results(results, params, models, columns, report_name)
    # Визуализация результатов
    plot_metrics(results, roc_curves, models, metrics)