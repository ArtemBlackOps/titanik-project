from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
import numpy as np
import pandas as pd

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SklearnPipeline

# Расширенный импорт метрик
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    log_loss,
    average_precision_score,
    root_mean_squared_error,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score
)

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor


# Допустимые метрики для задач и направление их оптимизации (True = чем больше, тем лучше)
SUPPORTED_METRICS = {
    "binary": {
        "roc_auc": True,
        "pr_auc": True,        # average_precision_score
        "accuracy": True,
        "precision": True,
        "recall": True,
        "f1": True,
        "log_loss": False
    },
    "multiclass": {
        "accuracy": True,
        "f1": True,            # f1_score(average='macro')
        "precision": True,     # precision_score(average='macro')
        "recall": True,        # recall_score(average='macro')
        "log_loss": False
    },
    "regression": {
        "rmse": False,
        "mse": False,
        "mae": False,
        "mape": False,
        "r2": True
    }
}


@dataclass
class TuningConfig:
    search_engine: str = "optuna"
    n_trials: int = 10
    param_spaces: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    task: str = "binary"  # "binary", "multiclass", "regression"
    val_strategy: Optional[str] = "stratified"
    n_splits: int = 5
    val_size: float = 0.2
    random_state: int = 42
    metric: str = "auto"
    run_mode: str = "train_only"
    score_name: str = "Score_v1"  # <-- Новое поле для имени столбца со скором
    tuning_config: TuningConfig = field(default_factory=TuningConfig)
    active_models: List[str] = field(default_factory=lambda: ["lgb", "xgb"])
    models_params: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        if self.metric == "auto":
            if self.task == "binary":
                self.metric = "roc_auc"
            elif self.task == "multiclass":
                self.metric = "accuracy"
            elif self.task == "regression":
                self.metric = "rmse"

        if self.task not in SUPPORTED_METRICS:
            raise ValueError(f"Неподдерживаемая задача '{self.task}'. Выберите из {list(SUPPORTED_METRICS.keys())}")

        task_metrics = SUPPORTED_METRICS[self.task]
        if self.metric not in task_metrics:
            raise ValueError(
                f"Метрика '{self.metric}' не поддерживается для задачи '{self.task}'. "
                f"Доступные метрики: {list(task_metrics.keys())}"
            )


def calculate_metric(y_true: Any, y_pred: Any, metric_name: str, task: str) -> float:
    """Единый расчёт всех поддерживаемых метрик."""
    if metric_name == "roc_auc":
        return float(roc_auc_score(y_true, y_pred))
    elif metric_name == "pr_auc":
        return float(average_precision_score(y_true, y_pred))
    elif metric_name == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    elif metric_name == "precision":
        avg = 'macro' if task == 'multiclass' else 'binary'
        return float(precision_score(y_true, y_pred, average=avg, zero_division=0))
    elif metric_name == "recall":
        avg = 'macro' if task == 'multiclass' else 'binary'
        return float(recall_score(y_true, y_pred, average=avg, zero_division=0))
    elif metric_name == "f1":
        avg = 'macro' if task == 'multiclass' else 'binary'
        return float(f1_score(y_true, y_pred, average=avg, zero_division=0))
    elif metric_name == "log_loss":
        return float(log_loss(y_true, y_pred))
    elif metric_name == "rmse":
        return float(root_mean_squared_error(y_true, y_pred))
    elif metric_name == "mse":
        return float(mean_squared_error(y_true, y_pred))
    elif metric_name == "mae":
        return float(mean_absolute_error(y_true, y_pred))
    elif metric_name == "mape":
        return float(mean_absolute_percentage_error(y_true, y_pred))
    elif metric_name == "r2":
        return float(r2_score(y_true, y_pred))
    else:
        raise ValueError(f"Неизвестная метрика: {metric_name}")


def get_model_predictions(model: Any, X: Any, metric_name: str, task: str) -> np.ndarray:
    """
    Возвращает вероятности или классы/значения в зависимости от требований метрики.
    """
    if task == "binary":
        if metric_name in ["roc_auc", "pr_auc", "log_loss"]:
            if hasattr(model, "predict_proba"):
                return model.predict_proba(X)[:, 1]
            elif hasattr(model, "decision_function"):
                return model.decision_function(X)
        return model.predict(X)

    elif task == "multiclass":
        if metric_name == "log_loss":
            return model.predict_proba(X)
        return model.predict(X)

    else:  # regression
        return model.predict(X)


class DataPreprocessor:
    def __init__(self, numeric_cols: List[str], categorical_cols: List[str], scale_num: bool = True):
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.scale_num = scale_num
        self.pipeline: Optional[ColumnTransformer] = None

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        num_transformer = SklearnPipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler() if self.scale_num else 'passthrough')
        ])
        cat_transformer = SklearnPipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        transformers = []
        if self.numeric_cols:
            transformers.append(('num', num_transformer, self.numeric_cols))
        if self.categorical_cols:
            transformers.append(('cat', cat_transformer, self.categorical_cols))

        if not transformers:
            return X.to_numpy() if isinstance(X, pd.DataFrame) else X

        self.pipeline = ColumnTransformer(transformers=transformers, remainder='passthrough')
        return self.pipeline.fit_transform(X)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            return X.to_numpy() if isinstance(X, pd.DataFrame) else X
        return self.pipeline.transform(X)


class ModelFactory:
    @staticmethod
    def get_model(model_name: str, task: str, params: Optional[Dict[str, Any]] = None, random_state: int = 42):
        params = params.copy() if params else {}

        if task in ["binary", "multiclass"]:
            if model_name == "logreg":
                return LogisticRegression(random_state=random_state, **params)
            elif model_name == "rf":
                return RandomForestClassifier(random_state=random_state, **params)
            elif model_name == "lgb":
                return LGBMClassifier(random_state=random_state, verbose=-1, **params)
            elif model_name == "xgb":
                return XGBClassifier(random_state=random_state, **params)
            elif model_name == "cat":
                return CatBoostClassifier(random_state=random_state, verbose=0, **params)

        elif task == "regression":
            if model_name == "ridge":
                return Ridge(random_state=random_state, **params)
            elif model_name == "rf":
                return RandomForestRegressor(random_state=random_state, **params)
            elif model_name == "lgb":
                return LGBMRegressor(random_state=random_state, verbose=-1, **params)
            elif model_name == "xgb":
                return XGBRegressor(random_state=random_state, **params)
            elif model_name == "cat":
                return CatBoostRegressor(random_state=random_state, verbose=0, **params)

        raise ValueError(f"Модель '{model_name}' не поддерживается для задачи '{task}'.")


class HyperparameterTuner:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def _evaluate_params(self, model_name: str, params: Dict[str, Any], X: Any, y: Any) -> float:
        if self.config.task in ["binary", "multiclass"]:
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.config.random_state)
        else:
            cv = KFold(n_splits=3, shuffle=True, random_state=self.config.random_state)

        scores = []
        for train_idx, val_idx in cv.split(X, y):
            if isinstance(X, (pd.DataFrame, pd.Series)):
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            else:
                X_tr, X_val = X[train_idx], X[val_idx]

            if isinstance(y, (pd.DataFrame, pd.Series)):
                y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            else:
                y_tr, y_val = y[train_idx], y[val_idx]

            model = ModelFactory.get_model(model_name, self.config.task, params, self.config.random_state)
            model.fit(X_tr, y_tr)

            preds = get_model_predictions(model, X_val, self.config.metric, self.config.task)
            score = calculate_metric(y_val, preds, self.config.metric, self.config.task)
            scores.append(score)

        return float(np.mean(scores))

    def tune_model(self, model_name: str, X: Any, y: Any) -> Dict[str, Any]:
        space = self.config.tuning_config.param_spaces.get(model_name, {})
        if not space:
            print(f"[Tuner] Пространство параметров для '{model_name}' не задано. Используются параметры по умолчанию.")
            return self.config.models_params.get(model_name, {})

        print(f"[Tuner] Старт подбора для '{model_name}' ({self.config.tuning_config.n_trials} итераций)...")

        is_maximize = SUPPORTED_METRICS[self.config.task][self.config.metric]

        if HAS_OPTUNA and self.config.tuning_config.search_engine == "optuna":
            direction = "maximize" if is_maximize else "minimize"

            def objective(trial):
                trial_params = {}
                for p_name, p_bounds in space.items():
                    if isinstance(p_bounds, tuple) and len(p_bounds) == 2:
                        if isinstance(p_bounds[0], int) and isinstance(p_bounds[1], int):
                            trial_params[p_name] = trial.suggest_int(p_name, p_bounds[0], p_bounds[1])
                        elif isinstance(p_bounds[0], float) and isinstance(p_bounds[1], float):
                            trial_params[p_name] = trial.suggest_float(p_name, p_bounds[0], p_bounds[1])
                    elif isinstance(p_bounds, list):
                        trial_params[p_name] = trial.suggest_categorical(p_name, p_bounds)

                return self._evaluate_params(model_name, trial_params, X, y)

            study = optuna.create_study(direction=direction)
            study.optimize(objective, n_trials=self.config.tuning_config.n_trials)
            print(f"[Tuner] Лучшие параметры для {model_name}: {study.best_params}")
            return study.best_params
        else:
            best_score = -float('inf') if is_maximize else float('inf')
            best_params = {}
            for _ in range(self.config.tuning_config.n_trials):
                rand_params = {}
                for p_name, p_bounds in space.items():
                    if isinstance(p_bounds, tuple) and len(p_bounds) == 2:
                        if isinstance(p_bounds[0], int) and isinstance(p_bounds[1], int):
                            rand_params[p_name] = int(np.random.randint(p_bounds[0], p_bounds[1] + 1))
                        else:
                            rand_params[p_name] = float(np.random.uniform(p_bounds[0], p_bounds[1]))
                    elif isinstance(p_bounds, list):
                        rand_params[p_name] = np.random.choice(p_bounds)

                score = self._evaluate_params(model_name, rand_params, X, y)
                is_better = (score > best_score) if is_maximize else (score < best_score)
                if is_better:
                    best_score = score
                    best_params = rand_params

            print(f"[Tuner] Лучшие параметры для {model_name}: {best_params}")
            return best_params


class MLPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.preprocessor: Optional[DataPreprocessor] = None
        self.fitted_models: Dict[str, List[Any]] = {}
        self.oof_predictions: Dict[str, Any] = {}
        self.best_params: Dict[str, Dict[str, Any]] = {}

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
            cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
            self.preprocessor = DataPreprocessor(num_cols, cat_cols, scale_num=True)
            X_proc = self.preprocessor.fit_transform(X)
        else:
            X_proc = X

        y_vec = y.values if isinstance(y, (pd.Series, pd.DataFrame)) else y

        tuner = HyperparameterTuner(self.config)
        for model_name in self.config.active_models:
            if self.config.run_mode == "tune":
                self.best_params[model_name] = tuner.tune_model(model_name, X_proc, y_vec)
            else:
                self.best_params[model_name] = self.config.models_params.get(model_name, {})

        results = []

        for model_name in self.config.active_models:
            print(f"\n=================== Training: {model_name} ===================")
            params = self.best_params[model_name]
            self.fitted_models[model_name] = []

            if self.config.val_strategy is None:
                print("--> Обучение на 100% данных без валидации...")
                model = ModelFactory.get_model(model_name, self.config.task, params, self.config.random_state)
                model.fit(X_proc, y_vec)
                self.fitted_models[model_name].append(model)
                
                # Замеряем метрику на train, если нет валидации
                preds = get_model_predictions(model, X_proc, self.config.metric, self.config.task)
                score = calculate_metric(y_vec, preds, self.config.metric, self.config.task)
                score_val = round(score * 100, 2) if self.config.metric in ["accuracy", "roc_auc", "f1"] else round(score, 4)

            elif self.config.val_strategy == "holdout":
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X_proc, y_vec, test_size=self.config.val_size,
                    random_state=self.config.random_state,
                    stratify=y_vec if self.config.task in ["binary", "multiclass"] else None
                )
                model = ModelFactory.get_model(model_name, self.config.task, params, self.config.random_state)
                model.fit(X_tr, y_tr)

                preds = get_model_predictions(model, X_val, self.config.metric, self.config.task)
                score = calculate_metric(y_val, preds, self.config.metric, self.config.task)
                print(f"--> Holdout {self.config.metric.upper()}: {score:.4f}")

                self.fitted_models[model_name].append(model)
                score_val = round(score * 100, 2) if self.config.metric in ["accuracy", "roc_auc", "f1"] else round(score, 4)

            else:
                if self.config.val_strategy == "stratified" and self.config.task in ["binary", "multiclass"]:
                    cv = StratifiedKFold(n_splits=self.config.n_splits, shuffle=True, random_state=self.config.random_state)
                else:
                    cv = KFold(n_splits=self.config.n_splits, shuffle=True, random_state=self.config.random_state)

                if self.config.task == "multiclass" and self.config.metric == "log_loss":
                    n_classes = len(np.unique(y_vec))
                    oof = np.zeros((len(X_proc), n_classes))
                else:
                    oof = np.zeros(len(X_proc))

                for fold, (train_idx, val_idx) in enumerate(cv.split(X_proc, y_vec)):
                    X_tr, X_val = X_proc[train_idx], X_proc[val_idx]
                    y_tr, y_val = y_vec[train_idx], y_vec[val_idx]

                    model = ModelFactory.get_model(model_name, self.config.task, params, self.config.random_state + fold)
                    model.fit(X_tr, y_tr)

                    preds = get_model_predictions(model, X_val, self.config.metric, self.config.task)
                    oof[val_idx] = preds
                    self.fitted_models[model_name].append(model)

                self.oof_predictions[model_name] = oof
                total_score = calculate_metric(y_vec, oof, self.config.metric, self.config.task)
                print(f"--> Total OOF {self.config.metric.upper()}: {total_score:.4f}")
                score_val = round(total_score * 100, 2) if self.config.metric in ["accuracy", "roc_auc", "f1"] else round(total_score, 4)

            results.append({
                'Model': model_name,
                self.config.score_name: score_val
            })

        # Формируем и сортируем итоговый DataFrame
        is_maximize = SUPPORTED_METRICS[self.config.task][self.config.metric]
        summary_df = pd.DataFrame(results).sort_values(
            by=self.config.score_name, ascending=not is_maximize
        ).reset_index(drop=True)

        return summary_df

    def predict(self, X_test: pd.DataFrame) -> pd.DataFrame:
        if self.preprocessor is not None and isinstance(X_test, pd.DataFrame):
            X_test_proc = self.preprocessor.transform(X_test)
        else:
            X_test_proc = X_test

        predictions_df = pd.DataFrame(index=range(len(X_test))) if not isinstance(X_test, pd.DataFrame) else pd.DataFrame(index=X_test.index)

        for model_name, models in self.fitted_models.items():
            model_preds = []
            for model in models:
                p = get_model_predictions(model, X_test_proc, self.config.metric, self.config.task)
                model_preds.append(p)

            predictions_df[f"{model_name}_pred"] = np.mean(model_preds, axis=0)

        pred_cols = [c for c in predictions_df.columns if c.endswith("_pred")]
        predictions_df["ensemble_pred"] = predictions_df[pred_cols].mean(axis=1)

        return predictions_df