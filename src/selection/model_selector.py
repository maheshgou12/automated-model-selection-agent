def select_best_model(results, problem_type):
    """
    Select best model based on problem type
    """
    if problem_type == "classification":
        best_model_name = max(results, key=results.get)
    else:
        best_model_name = min(results, key=results.get)

    return best_model_name
