from sklearn.metrics import accuracy_score, mean_squared_error


def evaluate_model(model, X_test, y_test, problem_type):
    """
    Evaluate model based on problem type
    """
    y_pred = model.predict(X_test)

    if problem_type == "classification":
        score = accuracy_score(y_test, y_pred)
    else:
        score = mean_squared_error(y_test, y_pred, squared=False)

    return score
