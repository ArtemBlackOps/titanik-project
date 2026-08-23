import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from category_encoders import TargetEncoder

from sklearn.preprocessing import StandardScaler

def process_and_align_datasets(train_df, test_df):
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']
    
    # 1. Создаем явные копии срезов
    X_tr = train_df[features].copy()
    X_te = test_df[features].copy()
    
    # 2. Заполняем пропуски в Age (медиану считаем по TRAIN!)
    age_median = X_tr['Age'].median()
    X_tr['Age'] = X_tr['Age'].fillna(age_median)
    X_te['Age'] = X_te['Age'].fillna(age_median)
    
    # 3. Биннинг SibSp
    bins = [-1, 0, 2, np.inf]
    labels = ['Alone', 'Few_persons', 'Family']
    X_tr['SibSp_new'] = pd.cut(X_tr['SibSp'], bins=bins, labels=labels, include_lowest=True)
    X_te['SibSp_new'] = pd.cut(X_te['SibSp'], bins=bins, labels=labels, include_lowest=True)
    
    # 4. Маппинг категориальных фичей
    sex_map = {'female': 1, 'male': 0}
    pclass_map = {1: 'Upper', 2: 'Middle', 3: 'Lower'}
    embarked_map = {'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'}
    
    for df in [X_tr, X_te]:
        df['Sex'] = df['Sex'].map(sex_map)
        df['Pclass'] = df['Pclass'].map(pclass_map)
        df['Embarked'] = df['Embarked'].map(embarked_map)
    
    # 5. Применяем pd.get_dummies
    cols_to_use = ['Pclass', 'Sex', 'SibSp_new', 'Parch', 'Embarked', 'Age']
    X_tr_encoded = pd.get_dummies(X_tr[cols_to_use], drop_first=True)
    X_te_encoded = pd.get_dummies(X_te[cols_to_use], drop_first=True)
    
    # 6. Выравниваем колонки test по train (если в test не оказалось какой-то категории)
    X_tr_encoded, X_te_encoded = X_tr_encoded.align(X_te_encoded, join='left', axis=1, fill_value=0)
    
    # 7. Приводим bool к int (1 и 0)
    bool_cols_tr = X_tr_encoded.select_dtypes(include='bool').columns
    X_tr_encoded[bool_cols_tr] = X_tr_encoded[bool_cols_tr].astype(int)
    
    bool_cols_te = X_te_encoded.select_dtypes(include='bool').columns
    X_te_encoded[bool_cols_te] = X_te_encoded[bool_cols_te].astype(int)
    
    return X_tr_encoded, X_te_encoded

def process_and_align_datasets_v2(train, test):
    # 1. Расчет медианы возраста строго по Train
    age_median = train['Age'].median()
    
    # ==========================================
    # 1. ОБРАБОТКА TRAIN DATASET
    # ==========================================
    train_data = train[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']].copy()
    
    train_data['Age'] = train_data['Age'].fillna(age_median).astype(int)
    train_data['Sex'] = train_data['Sex'].map({'female': 1, 'male': 0})
    train_data['Embarked'] = train_data['Embarked'].map(
        {'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'}
    )
    
    train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1
    train_data['IsAlone'] = (train_data['FamilySize'] == 1).astype(int)
    
    train_data['Age'] = pd.cut(
        train_data['Age'],
        bins=[-float('inf'), 16, 32, 48, 64, float('inf')],
        labels=[0, 1, 2, 3, 4],
    ).astype(int)
    
    # Target Encoding: Обучение и трансформацию делаем на Train
    encoder = TargetEncoder(cols=['Embarked'])
    train_data['Embarked_encoded'] = encoder.fit_transform(
        train_data['Embarked'], train_data['Survived']
    ).round(2)
    
    train_data['Age*Pclass'] = train_data['Age'] * train_data['Pclass']
    train_data = train_data.drop(columns=['Embarked'])
    
    # ==========================================
    # 2. ОБРАБОТКА TEST DATASET
    # ==========================================
    test_data = test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']].copy()
    
    test_data['Age'] = test_data['Age'].fillna(age_median).astype(int)
    test_data['Sex'] = test_data['Sex'].map({'female': 1, 'male': 0})
    test_data['Embarked'] = test_data['Embarked'].map(
        {'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'}
    )
    
    test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1
    test_data['IsAlone'] = (test_data['FamilySize'] == 1).astype(int)
    
    test_data['Age'] = pd.cut(
        test_data['Age'],
        bins=[-float('inf'), 16, 32, 48, 64, float('inf')],
        labels=[0, 1, 2, 3, 4],
    ).astype(int)
    
    # Target Encoding: Используем трансформер, обученный на Train
    test_data['Embarked_encoded'] = encoder.transform(
        test_data['Embarked']
    ).round(2)
    
    test_data['Age*Pclass'] = test_data['Age'] * test_data['Pclass']
    test_data = test_data.drop(columns=['Embarked'])
    
    # Формируем финальные выборки
    X_train = train_data.drop(columns=['Survived', 'Parch', 'Age'])
    Y_train = train_data['Survived']
    X_test = test_data.drop(columns=['Parch', 'Age'])
    
    return X_train, Y_train, X_test

def process_and_align_datasets_v3(train_df: pd.DataFrame, test_df: pd.DataFrame):
    # 1. Расчет медианы возраста строго по Train
    age_median = train_df['Age'].median()
    
    # Списки колонок
    raw_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']
    
    # 2. Копирование и заполнение пропусков
    X_tr = train_df[raw_cols].copy()
    X_te = test_df[raw_cols].copy()
    
    X_tr['Age'] = X_tr['Age'].fillna(age_median).astype(int)
    X_te['Age'] = X_te['Age'].fillna(age_median).astype(int)
    
    # 3. Маппинг категорий
    sex_map = {'female': 1, 'male': 0}
    pclass_map = {1: 'Upper', 2: 'Middle', 3: 'Lower'}
    embarked_map = {'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'}
    
    bins = [0, 12, 18, 35, 60, 100]
    
    for df in [X_tr, X_te]:
        df['Sex'] = df['Sex'].map(sex_map)
        df['Pclass'] = df['Pclass'].map(pclass_map)
        df['Embarked'] = df['Embarked'].map(embarked_map)
        
        # Признаки семьи и возраста
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=False, right=False)

    # 4. One-Hot Encoding
    features = ['Pclass', 'Sex', 'Parch', 'Embarked', 'FamilySize', 'IsAlone', 'AgeGroup']
    
    X_tr_encoded = pd.get_dummies(X_tr[features], dtype=int, drop_first=True)
    X_te_encoded = pd.get_dummies(X_te[features], dtype=int, drop_first=True)
    
    # Выравнивание структуры столбцов OHE между Train и Test
    X_tr_encoded, X_te_encoded = X_tr_encoded.align(X_te_encoded, join='left', axis=1, fill_value=0)
    
    # 5. Масштабирование возраста (Age)
    scaler = StandardScaler()
    X_tr_encoded['Age_scaled'] = scaler.fit_transform(X_tr[['Age']])
    X_te_encoded['Age_scaled'] = scaler.transform(X_te[['Age']])
    
    return X_tr_encoded, X_te_encoded

def evaluate_baseline_models(X_train, Y_train, X_test, score_name: str = 'Score'):
    # 1. Задаем словарь всех проверяемых моделей
    models_dict = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Support Vector Machines': SVC(random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=3),
        'Naive Bayes': GaussianNB(),
        'Perceptron': Perceptron(random_state=42),
        'Linear SVC': LinearSVC(random_state=42, max_iter=2000),
        'Stochastic Gradient Decent': SGDClassifier(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
    }
    
    results = []
    
    # 2. Обучаем каждую модель в цикле и собираем результаты
    for name, model in models_dict.items():
        model.fit(X_train, Y_train)
        
        # Инференс на тесте
        Y_pred = model.predict(X_test)
        
        # Расчет метрики на train
        acc_score = round(model.score(X_train, Y_train) * 100, 2)
        
        # Имя ключа формируется динамически из переданной переменной
        results.append({
            'Model': name,
            score_name: acc_score
        })
    
    # 3. Формируем DataFrame и сортируем по кастомному имени колонки
    models_df = pd.DataFrame(results)
    sorted_models = models_df.sort_values(by=score_name, ascending=False).reset_index(drop=True)
    
    return sorted_models












