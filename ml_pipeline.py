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
from sklearn.metrics import roc_auc_score, f1_score, root_mean_squared_error, accuracy_score

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor


@dataclass
class TuningConfig:
    search_engine: str = "optuna"
    n_trials: int = 10
    param_spaces: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    task: str = "binary"
    val_strategy: Optional[str] = "stratified"
    n_splits: int = 5
    val_size: float = 0.2
    random_state: int = 42
    metric: str = "auto"
    run_mode: str = "train_only"
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

        raise ValueError(f"Model '{model_name}' is not supported for task '{task}'.")


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
            # Исправленная безопасная индексация для массивов и датафреймов
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

            if self.config.metric == "roc_auc":
                preds = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_val)
                scores.append(roc_auc_score(y_val, preds))
            elif self.config.metric == "rmse":
                preds = model.predict(X_val)
                scores.append(root_mean_squared_error(y_val, preds))
            else:
                preds = model.predict(X_val)
                scores.append(accuracy_score(y_val, preds))

        return float(np.mean(scores))

    def tune_model(self, model_name: str, X: Any, y: Any) -> Dict[str, Any]:
        space = self.config.tuning_config.param_spaces.get(model_name, {})
        if not space:
            print(f"[Tuner] Space for '{model_name}' not defined. Using default params.")
            return self.config.models_params.get(model_name, {})

        print(f"[Tuner] Starting tuning for '{model_name}' ({self.config.tuning_config.n_trials} trials)...")

        if HAS_OPTUNA and self.config.tuning_config.search_engine == "optuna":
            direction = "minimize" if self.config.metric == "rmse" else "maximize"
            
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
            print(f"[Tuner] Best params for {model_name}: {study.best_params}")
            return study.best_params
        else:
            best_score = float('inf') if self.config.metric == "rmse" else -float('inf')
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
                is_better = (score < best_score) if self.config.metric == "rmse" else (score > best_score)
                if is_better:
                    best_score = score
                    best_params = rand_params

            print(f"[Tuner] Best params for {model_name}: {best_params}")
            return best_params


class MLPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.preprocessor: Optional[DataPreprocessor] = None
        self.fitted_models: Dict[str, List[Any]] = {}
        self.oof_predictions: Dict[str, np.ndarray] = {}
        self.best_params: Dict[str, Dict[str, Any]] = {}

    def _get_metric_score(self, y_true, y_pred) -> float:
        if self.config.metric == "roc_auc":
            return roc_auc_score(y_true, y_pred)
        elif self.config.metric == "rmse":
            return root_mean_squared_error(y_true, y_pred)
        elif self.config.metric == "f1":
            return f1_score(y_true, y_pred, average='macro' if self.config.task == 'multiclass' else 'binary')
        else:
            return accuracy_score(y_true, y_pred)

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
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

        metrics_summary = {}

        for model_name in self.config.active_models:
            print(f"\n=================== Training: {model_name} ===================")
            params = self.best_params[model_name]
            self.fitted_models[model_name] = []

            if self.config.val_strategy is None:
                print("--> Training on 100% data without validation...")
                model = ModelFactory.get_model(model_name, self.config.task, params, self.config.random_state)
                model.fit(X_proc, y_vec)
                self.fitted_models[model_name].append(model)
                metrics_summary[model_name] = None

            elif self.config.val_strategy == "holdout":
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X_proc, y_vec, test_size=self.config.val_size, 
                    random_state=self.config.random_state, 
                    stratify=y_vec if self.config.task in ["binary", "multiclass"] else None
                )
                model = ModelFactory.get_model(model_name, self.config.task, params, self.config.random_state)
                model.fit(X_tr, y_tr)
                
                preds = model.predict_proba(X_val)[:, 1] if self.config.task == "binary" and hasattr(model, "predict_proba") else model.predict(X_val)
                score = self._get_metric_score(y_val, preds)
                print(f"--> Holdout {self.config.metric.upper()}: {score:.4f}")
                
                self.fitted_models[model_name].append(model)
                metrics_summary[model_name] = score

            else:
                if self.config.val_strategy == "stratified" and self.config.task in ["binary", "multiclass"]:
                    cv = StratifiedKFold(n_splits=self.config.n_splits, shuffle=True, random_state=self.config.random_state)
                else:
                    cv = KFold(n_splits=self.config.n_splits, shuffle=True, random_state=self.config.random_state)

                oof = np.zeros(len(X_proc))
                for fold, (train_idx, val_idx) in enumerate(cv.split(X_proc, y_vec)):
                    X_tr, X_val = X_proc[train_idx], X_proc[val_idx]
                    y_tr, y_val = y_vec[train_idx], y_vec[val_idx]

                    model = ModelFactory.get_model(model_name, self.config.task, params, self.config.random_state + fold)
                    model.fit(X_tr, y_tr)

                    if self.config.task == "binary" and hasattr(model, "predict_proba"):
                        preds = model.predict_proba(X_val)[:, 1]
                    else:
                        preds = model.predict(X_val)

                    oof[val_idx] = preds
                    self.fitted_models[model_name].append(model)

                self.oof_predictions[model_name] = oof
                total_score = self._get_metric_score(y_vec, oof)
                print(f"--> Total OOF {self.config.metric.upper()}: {total_score:.4f}")
                metrics_summary[model_name] = total_score

        return metrics_summary

    def predict(self, X_test: pd.DataFrame) -> pd.DataFrame:
        if self.preprocessor is not None and isinstance(X_test, pd.DataFrame):
            X_test_proc = self.preprocessor.transform(X_test)
        else:
            X_test_proc = X_test

        predictions_df = pd.DataFrame(index=range(len(X_test))) if not isinstance(X_test, pd.DataFrame) else pd.DataFrame(index=X_test.index)

        for model_name, models in self.fitted_models.items():
            model_preds = []
            for model in models:
                if self.config.task == "binary" and hasattr(model, "predict_proba"):
                    p = model.predict_proba(X_test_proc)[:, 1]
                else:
                    p = model.predict(X_test_proc)
                model_preds.append(p)
            
            predictions_df[f"{model_name}_pred"] = np.mean(model_preds, axis=0)

        pred_cols = [c for c in predictions_df.columns if c.endswith("_pred")]
        predictions_df["ensemble_pred"] = predictions_df[pred_cols].mean(axis=1)

        return predictions_df