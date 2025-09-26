from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import joblib

preprocess_path = "../../datasets/preprocess"

def evaluation(model, X_train, y_train, X_test, y_test):
    # predict
    y_pred = model.predict(X_test)

    # evaluation
    accuracy = accuracy_score(y_test, y_pred) # 100%
    print('accuracy: ', accuracy * 100)

    cr = classification_report(y_test, y_pred)
    # print("calssification report: ", cr)

    cm = confusion_matrix(y_test, y_pred)
    # print("confusion matrix report: ", cm)

    # train/test scores
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    print('train score %: ', train_score * 100)
    print('test score %: ', test_score * 100)

    return accuracy

    # accuracy:  87.5
    # train score %:  98.94736842105263
    # test score %:  87.5
