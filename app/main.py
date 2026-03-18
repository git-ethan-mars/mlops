import os

import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost
from evidently import Report, Dataset, DataDefinition
from evidently.presets import DataDriftPreset
from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate, false_positive_rate
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from prometheus_client import Gauge
from prometheus_fastapi_instrumentator import Instrumentator
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder

app = FastAPI()

MODEL_F1_SCORE = Gauge('model_f1_score', 'Latest model F1 score on test set')
MODEL_ACCURACY_SCORE = Gauge('model_accuracy_score', 'Latest model ACCURACY score on test set')
MODEL_AUC_SCORE = Gauge('model_AUC_score', 'Latest model ROC-AUC score on test set')

Instrumentator().instrument(app).expose(app, endpoint='/metrics')


def apply_drift(X_data):
    X_drifted = X_data.copy()

    drift_values = np.random.randint(-1, 8, size=len(X_drifted))
    X_drifted['age'] = X_drifted['age'] + drift_values
    X_drifted['age'] = X_drifted['age'].clip(lower=0)

    X_drifted['hours-per-week'] = X_drifted['hours-per-week'] * 0.73

    mask = (
            (X_drifted['race'] == 'White') &
            (np.random.rand(len(X_drifted)) < 0.13)
    )
    X_drifted.loc[mask, 'race'] = 'Black'

    mask = (
            X_drifted['marital-status'].isin(['Married-civ-spouse', 'Never-married']) &
            (np.random.rand(len(X_drifted)) < 0.17)
    )
    X_drifted.loc[mask, 'marital-status'] = 'Divorced'

    mask = np.random.rand(len(X_drifted)) < 0.73
    X_drifted.loc[mask, 'sex'] = X_drifted.loc[mask, 'sex'].apply(
        lambda x: 'Female' if x == 'Male' else 'Male'
    )

    return X_drifted


@app.post("/api/train")
async def train_model(n_trials: int = Query(5, gt=0, le=1000)):
    try:
        loaded_artifacts = joblib.load("models/artifacts.pkl")

        X_train = loaded_artifacts.get("X_train_drifted", loaded_artifacts["X_train"])
        X_test = loaded_artifacts.get("X_test_drifted", loaded_artifacts["X_test"])
        y_train = loaded_artifacts["y_train"]
        y_test = loaded_artifacts["y_test"]
    except (FileNotFoundError, KeyError):
        df = pd.read_csv('data/adult.csv')
        df['income_fix'] = df['income'].apply(lambda v: v.replace('.', ''))
        df['workclass'] = df['workclass'].fillna(df['workclass'].mode()[0])
        df['occupation'] = df['occupation'].fillna(df['occupation'].mode()[0])
        df['native-country'] = df['native-country'].fillna(df['native-country'].mode()[0])
        filtered_df = df[(df['age'] < 75) & (25 < df['hours-per-week']) & (df['hours-per-week'] < 55)]

        X = filtered_df.drop(['income', 'income_fix'], axis=1)
        y = filtered_df['income_fix'].apply(lambda val: 1 if val == '>50K' else 0)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

    numeric_columns = X_train.select_dtypes(include=['number']).columns
    categorical_columns = X_train.select_dtypes(exclude=['number']).columns

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(
        handle_unknown='ignore',
        sparse_output=False,
        drop='first'
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_columns),
            ('cat', categorical_transformer, categorical_columns)
        ]
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'objective': 'binary:logistic',
            'random_state': 42,
            'tree_method': 'hist',
            'n_jobs': -1
        }

        classifier = xgboost.XGBClassifier(**params)
        scores = cross_val_score(
            classifier,
            X_train_processed,
            y_train,
            scoring='f1',
            cv=5,
            n_jobs=-1
        )
        return scores.mean()

    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.HyperbandPruner()
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params

    best_model = xgboost.XGBClassifier(
        **best_params,
        objective='binary:logistic',
        random_state=42,
        tree_method='hist',
        n_jobs=-1
    )
    best_model.fit(X_train_processed, y_train)

    y_preds = best_model.predict(X_test_processed)

    test_f1 = f1_score(y_test, y_preds)
    test_accuracy = accuracy_score(y_test, y_preds)
    test_auc = roc_auc_score(y_test, best_model.predict_proba(X_test_processed)[:, 1])

    MODEL_F1_SCORE.set(test_f1)
    MODEL_ACCURACY_SCORE.set(test_accuracy)
    MODEL_AUC_SCORE.set(test_auc)

    os.makedirs("models", exist_ok=True)
    artifacts = {
        "model": best_model,
        "preprocessor": preprocessor,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test
    }

    joblib.dump(artifacts, "models/artifacts.pkl")

    return {
        "status": "success",
        "best_params": best_params,
        "cv_best_score": study.best_value,
        "test_metrics": {
            "f1": test_f1,
            "accuracy": test_accuracy,
            "auc": test_auc
        },
        "n_trials": n_trials
    }


@app.get("/api/drift")
async def get_data_drift():
    try:
        loaded_artifacts = joblib.load("models/artifacts.pkl")
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Model is not trained.")

    model = loaded_artifacts.get("model")
    preprocessor = loaded_artifacts.get("preprocessor")
    x_reference = loaded_artifacts.get("X_test")
    x_current = loaded_artifacts.get("X_test_drifted")

    if model is None or preprocessor is None or x_reference is None:
        raise HTTPException(status_code=400, detail="Model is not trained.")
    if x_current is None:
        x_current = x_reference

    numeric_columns = x_reference.select_dtypes(include=['number']).columns
    categorical_columns = x_reference.select_dtypes(exclude=['number']).columns

    eval_data = Dataset.from_pandas(x_reference, data_definition=DataDefinition(
        numerical_columns=list(numeric_columns),
        categorical_columns=list(categorical_columns)
    ))

    eval_data_drift = Dataset.from_pandas(x_current, data_definition=DataDefinition(
        numerical_columns=list(numeric_columns),
        categorical_columns=list(categorical_columns)
    ))

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=eval_data, current_data=eval_data_drift)

    html_content = snapshot.get_html_str(as_iframe=False)

    return HTMLResponse(content=html_content, status_code=200)


@app.post("/api/drift/simulate")
async def simulate_data_drift():
    try:
        loaded_artifacts = joblib.load("models/artifacts.pkl")
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Model is not trained.")

    model = loaded_artifacts.get("model")
    preprocessor = loaded_artifacts.get("preprocessor")
    X_train = loaded_artifacts.get("X_train")
    X_test = loaded_artifacts.get("X_test")
    y_test = loaded_artifacts.get("y_test")

    if model is None or X_train is None or X_test is None or y_test is None or preprocessor is None:
        raise HTTPException(status_code=400, detail="Model is not trained.")

    X_train_drifted = apply_drift(X_train)
    X_test_drifted = apply_drift(X_test)
    X_test_processed = preprocessor.transform(X_test_drifted)

    y_preds = model.predict(X_test_processed)
    test_f1 = f1_score(y_test, y_preds)
    test_accuracy = accuracy_score(y_test, y_preds)
    test_auc = roc_auc_score(y_test, model.predict_proba(X_test_processed)[:, 1])

    MODEL_F1_SCORE.set(test_f1)
    MODEL_ACCURACY_SCORE.set(test_accuracy)
    MODEL_AUC_SCORE.set(test_auc)

    loaded_artifacts["X_train_drifted"] = X_train_drifted
    loaded_artifacts["X_test_drifted"] = X_test_drifted
    joblib.dump(loaded_artifacts, "models/artifacts.pkl")

    return {
        "status": "success",
        "test_metrics": {
            "f1": test_f1,
            "accuracy": test_accuracy,
            "auc": test_auc
        }
    }


@app.get("/api/fairness")
async def get_fairness():
    try:
        loaded_artifacts = joblib.load("models/artifacts.pkl")
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Model is not trained.")

    model = loaded_artifacts.get("model")
    preprocessor = loaded_artifacts.get("preprocessor")
    x_reference = loaded_artifacts.get("X_test")
    y_reference = loaded_artifacts.get("y_test")
    x_current = loaded_artifacts.get("X_test_drifted")

    if model is None or preprocessor is None or x_reference is None or y_reference is None:
        raise HTTPException(status_code=400, detail="Model is not trained.")
    if x_current is None:
        x_current = x_reference

    X_test_processed = preprocessor.transform(x_current)
    y_preds = model.predict(X_test_processed)

    metric_frame = MetricFrame(
        metrics={
            'selection_rate': selection_rate,
            'TRP': true_positive_rate,
            'FPR': false_positive_rate
        },
        y_true=y_reference,
        y_pred=y_preds,
        sensitive_features=x_reference['sex']
    )

    dp_diff = demographic_parity_difference(
        y_true=y_reference,
        y_pred=y_preds,
        sensitive_features=x_reference['sex']
    )

    eo_diff = equalized_odds_difference(
        y_true=y_reference,
        y_pred=y_preds,
        sensitive_features=x_reference['sex']
    )

    return {
        "status": "success",
        "metric_frame_by_group": metric_frame.by_group.to_dict(),
        "metric_frame_overall": metric_frame.overall.to_dict(),
        "demographic_parity_difference": dp_diff,
        "equalized_odds_difference": eo_diff
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
