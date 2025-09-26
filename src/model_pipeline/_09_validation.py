import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold, cross_val_score


def validation(model, X_train, y_train):
    # cross vallidation: stratified k-fold validation
    strat_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=strat_cv, scoring='accuracy')
    
    cv_score_avg = cv_scores.mean() * 100
    print(f"Strat cv score: {cv_score_avg}")

    return cv_score_avg

