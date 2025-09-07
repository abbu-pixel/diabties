# model.py
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# URL for the Pima Indians Diabetes dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

# Define the column names for the dataset
column_names = [
    "pregnancies",
    "glucose",
    "blood_pressure",
    "skin_thickness",
    "insulin",
    "bmi",
    "diabetes_pedigree_function",
    "age",
    "outcome",
]

# Load the dataset directly from the URL using pandas
try:
    diabetes_df = pd.read_csv(url, header=None, names=column_names)
except Exception as e:
    print(f"Error loading data from URL: {e}")
    # Exit if data cannot be loaded
    exit()

# Drop the 'pregnancies' column as requested
diabetes_df = diabetes_df.drop("pregnancies", axis=1)

# Separate features (X) and target (y)
X = diabetes_df.drop("outcome", axis=1)
y = diabetes_df["outcome"]

# Create a pipeline to first scale the data and then apply the classifier
pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(random_state=42)),
    ]
)

# Define the parameter grid for the grid search
param_grid = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [10, 20, 30, None],
    "classifier__min_samples_split": [2, 5, 10],
}

# Perform the grid search with cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X, y)

# Get the best estimator from the grid search
best_model = grid_search.best_estimator_

print("Best model parameters found: ", grid_search.best_params_)
print(f"Model accuracy on the training set: {best_model.score(X, y):.4f}")

# Save the best trained model to a pickle file
with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("model.pkl has been created with the best performing model.")
