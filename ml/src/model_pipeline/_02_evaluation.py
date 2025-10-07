from sklearn.metrics import accuracy_score

def evaluation(model, X_train, y_train, X_test, y_test):
    # predict
    y_pred = model.predict(X_test)

    # evaluation
    accuracy = accuracy_score(y_test, y_pred) # 100%
    print('accuracy: ', accuracy * 100)

    # train/test scores
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    print('train score %: ', train_score * 100)
    print('test score %: ', test_score * 100)

    return accuracy