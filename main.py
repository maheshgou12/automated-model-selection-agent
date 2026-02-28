from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.pipeline import detect_problem_type
from src.models.classification_models import get_classification_models
from src.models.regression_models import get_regression_models
from src.evaluation.evaluator import evaluate_model
from src.selection.model_selector import select_best_model
import joblib

# 1. Load data
df = load_data("data/dataset.csv")
print("Rows & Columns:", df.shape)

# 2. Preprocess
X_train, X_test, y_train, y_test, preprocessor = preprocess_data(
    df, target_column="species"
)



# 3. Detect problem type
print("Target dtype:", y_train.dtype)
print("Target unique values:", y_train.unique())

problem_type = detect_problem_type(y_train)
print("Detected Problem Type:", problem_type)


# 4. Get models
if problem_type == "classification":
    models = get_classification_models()
else:
    models = get_regression_models()

# 5. Train & evaluate
results = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    score = evaluate_model(model, X_test, y_test, problem_type)
    results[model_name] = score

# 6. Select best model
best_model_name = select_best_model(results, problem_type)
best_model = models[best_model_name]

# 7. Save best model
joblib.dump(
    {
        "model": best_model,
        "preprocessor": preprocessor
    },
    "models/best_model.pkl"
)


# 8. Print results
print("\nModel Performance:")
for model_name, score in results.items():
    print(f"{model_name}: {score}")

print(f"\n🏆 Best Model Selected: {best_model_name}")
print("Best model saved to models/best_model.pkl")







