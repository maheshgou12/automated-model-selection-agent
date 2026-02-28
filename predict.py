import joblib
import pandas as pd

# 1. Load saved pipeline
pipeline = joblib.load("models/best_model.pkl")

model = pipeline["model"]
preprocessor = pipeline["preprocessor"]

# 2. Create sample input (MATCH feature columns exactly)
sample = pd.DataFrame([{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}])

# 3. Preprocess sample
sample_transformed = preprocessor.transform(sample)

# 4. Predict
prediction = model.predict(sample_transformed)

print("Prediction:", prediction)
