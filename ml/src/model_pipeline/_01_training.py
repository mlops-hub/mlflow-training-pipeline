from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def training(X_train, y_train):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression())
    ])
    
    pipeline.fit(X_train, y_train)
    print(pipeline)

    return pipeline
