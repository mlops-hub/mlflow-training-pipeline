from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import os

PROJECT_ROOT = os.getcwd()

def training(X_train, y_train):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression())
    ])
    
    # model.fit(X_train, y_train)
    pipeline.fit(X_train, y_train)
    print(pipeline)

    return pipeline
