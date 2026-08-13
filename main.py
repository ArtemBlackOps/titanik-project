import os
import pickle
import numpy as np
import pandas as pd
import polars as pl
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, roc_auc_score

# =====================================================================
# 1. СЛОВАРИ ГИПЕРПАРАМЕТРОВ ПО УМОЛЧАНИЮ
# =====================================================================

LINEAR_PARAMS = {
    'fit_intercept': True,
    'regularization': 'ridge',  # 'none', 'ridge', 'lasso', 'elasticnet'
    'alpha': 1.0,               # Сила регуляризации
    'l1_ratio': 0.5,            # Для elasticnet
    'max_iter': 2000,
    'tol': 1e-4,
    'random_state': 42
}

LOGISTIC_PARAMS = {
    'penalty': 'l2',            # 'l1', 'l2', 'elasticnet', None
    'C': 1.0,
    'l1_ratio': None,
    'solver': 'lbfgs',          # Автоматически сменится на saga/liblinear при L1/ElasticNet
    'max_iter': 2000,
    'class_weight': None,
    'n_jobs': -1,
    'random_state': 42
}

# =====================================================================
# 2. КЛАСС КОНФИГУРАЦИИ
# =====================================================================

class Config:
    def __init__(
        self,
        task: str = "regression",             # "regression" или "classification"
        n_splits: int = 5,
        linear_params: dict = None,
        logistic_params: dict = None,
        to_train: dict = None,
        to_inference: list = None,
        path_checkpoints: str = "./checkpoints/"
    ):
        self.task = task
        self.n_splits = n_splits
        self.path_checkpoints = path_checkpoints
        
        # Загрузка параметров по умолчанию с возможностью переопределения
        self.linear_params = linear_params if linear_params is not None else LINEAR_PARAMS.copy()
        self.logistic_params = logistic_params if logistic_params is not None else LOGISTIC_PARAMS.copy()
        
        self.to_train = to_train if to_train is not None else {
            "linear_regression": True,
            "logistic_regression": False
        }
        self.to_inference = to_inference if to_inference is not None else ["linear_regression_oof"]
        
        # Корректировка валидных параметров под выбранную логику
        self._adjust_parameters()

    def _adjust_parameters(self):
        """Автоматическая настройка логики модели на основе выставленных параметров."""
        
        # --- Корректировка для Logistic Regression ---
        if self.logistic_params.get('penalty') == 'l1' and self.logistic_params.get('solver') not in ['saga', 'liblinear']:
            self.logistic_params['solver'] = 'saga'
            
        if self.logistic_params.get('penalty') == 'elasticnet':
            self.logistic_params['solver'] = 'saga'
            if self.logistic_params.get('l1_ratio') is None:
                self.logistic_params['l1_ratio'] = 0.5


# =====================================================================
# 3. ВСПОМОГАТЕЛЬНАЯ ФАБРИКА МОДЕЛЕЙ
# =====================================================================

def build_model(model_name: str, config: Config):
    """Возвращает инициализированный объект sklearn-модели в зависимости от настроек конфига."""
    
    if model_name == "linear_regression":
        p = config.linear_params.copy()
        reg_type = p.pop('regularization', 'none').lower()
        
        if reg_type == 'ridge':
            p.pop('l1_ratio', None)
            return Ridge(**p)
        elif reg_type == 'lasso':
            p.pop('l1_ratio', None)
            return Lasso(**p)
        elif reg_type == 'elasticnet':
            return ElasticNet(**p)
        else:
            # Обычная LinearRegression без регуляризации
            return LinearRegression(
                fit_intercept=p.get('fit_intercept', True),
                n_jobs=-1
            )
            
    elif model_name == "logistic_regression":
        return LogisticRegression(**config.logistic_params)
        
    else:
        raise ValueError(f"Неизвестное имя модели: {model_name}")


# =====================================================================
# 4. ОСНОВНОЙ ПАЙПЛАЙН (FIT / PREDICT)
# =====================================================================

class MLPipeline:
    def __init__(self, config: Config, rerun: bool = False):
        self.config = config
        self.rerun = rerun
        os.makedirs(self.config.path_checkpoints, exist_ok=True)

    def _scale_data(self, X_train, X_valid, X_test=None):
        """Внутренний скейлер для предотвращения утечек данных (Data Leakage)."""
        drop_cols = [c for c in ["data_mode", "index"] if c in X_train.columns]
        X_tr = X_train.drop(columns=drop_cols, errors='ignore').copy()
        X_va = X_valid.drop(columns=drop_cols, errors='ignore').copy()

        scaler = StandardScaler()
        num_cols = X_tr.select_dtypes(include=[np.number]).columns.tolist()

        X_tr[num_cols] = scaler.fit_transform(X_tr[num_cols])
        X_va[num_cols] = scaler.transform(X_va[num_cols])

        if X_test is not None:
            X_te = X_test.drop(columns=drop_cols, errors='ignore').copy()
            X_te[num_cols] = scaler.transform(X_te[num_cols])
            return X_tr, X_va, X_te

        return X_tr, X_va

    def fit(self, df: pd.DataFrame, X: pd.DataFrame, Y: pd.Series, groups: pd.Series = None, catcols: list = None) -> dict:
        """Метод обучения моделей с генерацией OOF-предсказаний и подчетом метрик."""
        oof_results = {}

        for model_name, is_active in self.config.to_train.items():
            if not is_active:
                continue

            print(f"\n================ Обучение: {model_name} ================")
            oof_preds = np.zeros(len(df))
            feature_importances = None

            for fold in range(self.config.n_splits):
                train_idx = df[df["fold"] != fold].index
                valid_idx = df[df["fold"] == fold].index

                X_tr, X_va = self._scale_data(X.iloc[train_idx], X.iloc[valid_idx])
                Y_tr, Y_va = Y.iloc[train_idx], Y.iloc[valid_idx]

                ckpt_path = os.path.join(self.config.path_checkpoints, f"{model_name}_fold_{fold}.pickle")

                if not self.rerun:
                    model = build_model(model_name, self.config)
                    model.fit(X_tr, Y_tr)
                    with open(ckpt_path, "wb") as f:
                        pickle.dump(model, f)
                else:
                    with open(ckpt_path, "rb") as f:
                        model = pickle.load(f)

                # Предсказание OOF
                if model_name == "logistic_regression" and self.config.task == "classification":
                    preds = model.predict_proba(X_va)[:, 1]
                else:
                    preds = model.predict(X_va)

                oof_preds[valid_idx] = preds

                # Подсчет коэффициентов
                coefs = np.abs(model.coef_).ravel() if hasattr(model, "coef_") else np.zeros(X_tr.shape[1])
                feature_importances = coefs / self.config.n_splits if feature_importances is None else feature_importances + (coefs / self.config.n_splits)

            # Вывод итога по модели
            oof_results[f"{model_name}_oof"] = oof_preds
            score = np.sqrt(mean_squared_error(Y, oof_preds)) if self.config.task == "regression" else roc_auc_score(Y, oof_preds)
            print(f"--> Total OOF Metric ({self.config.task}): {round(score, 4)}")

            fi_df = pl.DataFrame({"feature": X_tr.columns.tolist(), "importance": feature_importances}).sort(by='importance', descending=True)
            print("Top-5 Important Features:\n", fi_df.head(5))

        return oof_results

    def predict(self, df_train: pd.DataFrame, X_train: pd.DataFrame, df_test: pd.DataFrame, X_test: pd.DataFrame) -> pd.DataFrame:
        """Метод применения чекпоинтов моделей к тестовой выборке."""
        sub_dict = {"index": df_test.index if "index" not in df_test.columns else df_test["index"]}

        for model_key in self.config.to_inference:
            model_name = model_key.replace("_oof", "")
            print(f"\n================ Инференс: {model_name} ================")

            test_preds = np.zeros(len(df_test))

            for fold in range(self.config.n_splits):
                train_idx = df_train[df_train["fold"] != fold].index
                _, _, X_te = self._scale_data(X_train.iloc[train_idx], X_train.iloc[train_idx], X_test)

                ckpt_path = os.path.join(self.config.path_checkpoints, f"{model_name}_fold_{fold}.pickle")
                with open(ckpt_path, "rb") as f:
                    model = pickle.load(f)

                if model_name == "logistic_regression" and self.config.task == "classification":
                    test_preds += model.predict_proba(X_te)[:, 1] / self.config.n_splits
                else:
                    test_preds += model.predict(X_te) / self.config.n_splits

            sub_dict[model_key] = test_preds

        return pd.DataFrame(sub_dict)