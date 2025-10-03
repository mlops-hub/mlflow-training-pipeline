from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
import pandas as pd
import joblib
import os

PROJECT_ROOT = os.getcwd()

def tuning(base_model, X_train, X_test, y_train, y_test):
    # set parameters
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['lbfgs', 'saga'],
        'penalty': ['l2'],
        'max_iter': [1000]
    }

    # set cv
    strat_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

    # use gridserachcv for tuning model
    grid = GridSearchCV(base_model, param_grid=param_grid, cv=strat_cv, scoring='accuracy')

    # train gridsearchcv model
    grid.fit(X_train, y_train)

    # get best model and save in models/
    tuned_model = grid.best_estimator_
    output_dir = os.path.join(PROJECT_ROOT, "models")
    joblib.dump(tuned_model, f"{output_dir}/best_model.pkl")

    print(f'best params: {grid.best_params_}')
    print(f'best cv scores: {grid.best_score_ * 100}')
    print(f'best model: {tuned_model}')

    # log other paramaeter values in logs/
    results = pd.DataFrame(grid.cv_results_)
    print(results.head(3))

    grid_results = results[['param_C', 'param_solver', 'mean_test_score', 'std_test_score']]

    log_path = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(log_path, exist_ok=True)

    grid_results.to_csv(f'{log_path}/gridsearch_logs.csv', index=False)

    # predict the output with tuned_model
    y_pred_ht = tuned_model.predict(X_test)

    # tuned model evaluation
    accuracy_ht = accuracy_score(y_test, y_pred_ht)
    print('accuracy: ', accuracy_ht)

    # get trin/test score
    tuned_train_score = tuned_model.score(X_train, y_train) 
    tuned_test_score =  tuned_model.score(X_test, y_test)

    print('tuned train score: ', tuned_train_score)
    print('tuned test score: ', tuned_test_score)

    # Compare CV scores of base model and tuned model
    base_cv_scores = cross_val_score(base_model, X_train, y_train, cv=strat_cv)
    tuned_cv_scores = cross_val_score(tuned_model, X_train, y_train, cv=strat_cv)

    print("=== MODEL COMPARISON ===")
    print(f"Base Model CV:     {base_cv_scores.mean():.4f} (+/- {base_cv_scores.std():.4f})")
    print(f"Tuned Model CV:    {tuned_cv_scores.mean():.4f} (+/- {tuned_cv_scores.std():.4f})")

    return tuned_model, grid.best_params_, accuracy_ht