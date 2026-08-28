# Titanik-project
This project will analyze the Titanic dataset to identify indicators that influence passenger survival. The project will also create various machine learning and deep learning models to determine which one is most suitable for this study.

# 1. Summary

Я попробовал разные исследовал данные, применил разные обработки числовых, категориальных признаков, применил много разных алгоритмов, применил стратегию валидации через 5к-фолд

Инстайты:
1) как будет видно из разных обработок данных и добавления в них новых фичей, в основном увеличение метрик происходило как раз за счет добавления или оптимальной обработки фичей под алгоритмы
2) Доработки обработки и преобразования фичей скорее всего еще больше увеличат скор
3) Без валидации большинство моделей переобучались и давали плохой скор на тестовой выборке
4) Пайплайны в виде пай файлов помогают перебирать различные комбинации гиперпараметров
5) 

The general pipeline is presented below.

# 2. Validation + Preproccesing strategy

Стратифицированная кросс-валидация (5-Fold Stratified K-Fold): Разбиение сохраняет баланс классов целевой переменной Survived.

Строгая изоляция фолдов: Предобработка (processing_func) запускается внутри каждого фолда. Заполнение пропусков медианами, Target Encoding и StandardScaler рассчитываются строго на обучающей части фолда (fold_train) и применяются к валидационной (fold_val), что полностью исключает Data Leakage.

# 3. Data part
Первичный EDA позволил прийти к следующим выводам по построению бейзлайна:
Выкинуть столбцы: 
1) Ticket (потому что практически уникальные значения), 
2) Cabin (много пропусков, НО возможно если дополнительно их заполнить реальной инфой, то это местоположение может дать доп очки), 
3) Fare (цена билет по сути отображена в колонке Pclass, опять же потенциально можно преобразовать в доп переменную либо понадеятся что этот столбец дополнительно потом еще даст большее значение метрики)
4) Name (логически это вообще ни на что влиять не может, но опять же как доп фича преобразованная может дать еще прирост метрики)

Преобразовать: Parch, SibSp (логика в том, что как будто если у людей больше детей/родственников, то они могли забить место в шлюпке)

Оставить: 
1) Sex (тут сильная корреляция, по статистике женщин выжило больше), 
2) Pclass (по статистике выжило больше пассажиров 1 и 2 класса), 
3) Age (возможно если преобразовать в категориальный признак, то окажется что выжившие мужчины - дети/подростки), 
4) Embarked (по статистике частично повлияло на выживаемость)



# 2.1 Итерации инжиниринга фичей (Feature Engineering Pipelines)
V1 (OHE Baseline): Базовое заполнение пропусков Age общей медианой, биннинг SibSp, кодирование категорий через pd.get_dummies (One-Hot Encoding) с выравниванием колонок (align).

V2 (Target Encoding & Intersections): Введение синергетических признаков (FamilySize, IsAlone, Age*Pclass), биннинг возраста и кодирование порт приписки (Embarked) с помощью TargetEncoder.

V3 (Group Binning & Scaling): Группировка возраста по интервалам (AgeGroup), OHE для категорий и масштабирование непрерывного возраста через StandardScaler (оптимально для линейных моделей, SVM и Neural Networks/MLP).

V4 (Group Medians & Raw Numerical): Продвинутая импутация Age медианами по группам (Sex, Pkg/Pclass), прямое добавление признака Fare, сохранение числовых признаков без биннинга для деревьев решений.

# Baseline

| Model | V1_OHE_CV_Acc | V2_TargetEnc_CV_Acc | V3_Scaled_CV_Acc | V4_GroupMedians_CV_Acc |
| :--- | :---: | :---: | :---: | :---: |
| **CatBoost** | 82.38 | 81.03 | 81.59 | 84.28 |
| **Random Forest** | 81.14 | 81.03 | 80.58 | 82.04 |
| **XGBoost** | 81.14 | 80.92 | 80.81 | 82.16 |
| **LightGBM** | 80.81 | 80.69 | 82.04 | 83.28 |
| **Logistic Regression** | 80.59 | 79.80 | 80.25 | 80.59 |
| **Logistic Regression (L2)** | 80.36 | 79.80 | 80.25 | 76.32 |
| **Logistic Regression (ElasticNet)** | 80.36 | 79.80 | 80.25 | 76.21 |
| **Logistic Regression (L1)** | 80.03 | 79.58 | 80.14 | 76.21 |
| **Linear SVC** | 80.02 | 79.58 | 79.46 | 80.47 |
| **Naive Bayes** | 77.11 | 77.78 | 79.46 | 79.01 |
| **Advanced MLP (4-layer + Reg)** | 82.60 | 82.94 | 82.04 | 82.94 |
| **Simple MLP (2-layer)** | 82.49 | 80.70 | 82.60 | 80.59 |





# 4. Modeling
4.1. Approach
From the very beginning to the end of the competition, Catboost showed the best results for me, while other models were worse, so I decided to use a two-stage approach to build the ensemble.

I considered this problem as a regression task and trained all my models to optimize RMSE metric. I tried to solve this task as a classification one and optimizing QWK, but got worse results. Anyways, since we have discrete labels in training data, it's possible to found a way of rounding predictions that would led to a perfomance boost. I did some experiments with post-processing, where I was looking for some thresholds for rounding my predictions, however the results were unstable so I decided not to use it in my final solution.

I also tried different reguralizations to achieve more balanced feature importances but it didnt' lead to any improvements as well.

4.2. Catboost
The best single model. The parameters are adjusted manually.

catboost_params = {

        depth': 6
        learning_rate': 0.1
        l2_leaf_reg': 7
        iterations': 300
        verbose=0
        random_state=42

}
4.3. Two-staged models
LGBM, XGB and DNN models were consistently worse in this competition, so I decided to use catboost OOF predictions as another feature for these models.

First, I needed to collect those OOF predictions propeply. Using simple outputs of 5 folds model split didn't work and caused leakage so I used a Nested CV Catboost model (5*5 folds) for this. The reason to do so is described well in this discussion and this notebook by @martinapreusse.
Then I trained Catboost OOF and LGBM OOF models. Paramters for catboost are the same as above, parameters for LGBM model presented below, I didn't tune them a lot and there is a huge room for improvement here.
lgbm_params = {
    'objective': 'regression',
    'min_child_samples': 24,
    'num_iterations': 30000,
    'learning_rate': 0.01,
    'extra_trees': True,
    'reg_lambda': 0.8,
    'reg_alpha': 0.1,
    'num_leaves': 64,
    'metric': 'rmse',
    'device': 'cpu',
    'max_depth': 9,
    'max_bin': 128,
    'verbose': -1,
    'seed': 42
}
Next I trained another Catboost model using OOF predictions, but this time not as a feature but as a baseline initialization. An example can be found here.
Finally, I trained a DNN model. I tried a lot of different architecture variations, but ended up with a classic MLP with modifications:
    1. Embedding layer (128) for categorical features.
    2. Quantile transformer for numerical features.
    3. OOF feature as it is.
    4. Concat -> MLP (Dropout=0.9, Hardswish, input -> 2048 -> 1024 -> 512 -> 256 -> 128 -> 1).
    5. Use MADGRAD optimizer and RMSE loss for optimization.
4.4. Other models
Besides that I tried to train another meta-models that would predict number of games played, corner cases and draws, however it didn't lead to any significant improvements. XGBoost models weren't effective as well.

4.5. Ensembling
After testing various ensembling approaches I just ended up choosing weighted ensemble with positive weights selected by the scipy minimize function based on CV score.


# 6. Final results


| Model | CV | TestScore |
| :--- | :---: | :---: |
|solo cat| CV: 84.62| final score: 0.77272
|CatBoost Ensemble with OOF| CV: 83.50| final score: 0.78229
|CatBoost Ensemble with OOF & Optuna Tuning| CV: 85.41| final score: 0.77511
|XGB Ensemble with OOF & Optuna Tuning| CV: 84.40| final score: 0.77033
|simple_mlp_v1| CV: 82.49| final score: 0.77751
|advanced_mlp_v4| CV: 82.94| final score: 0.77033



# 7. What can also be beneficial and what is not






