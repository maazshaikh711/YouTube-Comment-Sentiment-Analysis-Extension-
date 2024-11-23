import pytest
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from load_model_test import setup_mlflow_tracking, load_model_from_registry

@pytest.mark.parametrize("model_name, vectorizer_name, stage, holdout_data_path", [
    ("sentiment_analysislgbm_model", "sentiment_analysisvectorizer", "staging", "data/interim/test_processed.csv"),
])
def test_model_performance(model_name, vectorizer_name, stage, holdout_data_path):
    """
    Performance test for the latest model and vectorizer from MLflow registry, with feature transformation and joining.
    """
    try:
        # Set up environment and MLflow tracking URI
        setup_mlflow_tracking()
        
        # Load model and vectorizer from MLflow registry
        model, vectorizer = load_model_from_registry(model_name, vectorizer_name, stage)

        # Load holdout test data
        holdout_data = pd.read_csv(holdout_data_path)
        X_holdout_raw = holdout_data["comment"]  
        y_holdout = holdout_data["category"]   

        # Drop the 'comment' and 'category' columns to retain additional features
        additional_features = holdout_data.drop(columns=["comment", "category"])

        # Handle NaN values in the text data
        X_holdout_raw = X_holdout_raw.fillna("")

        # Apply vectorizer transformation
        X_transformed = vectorizer.transform(X_holdout_raw)
        X_transformed_df = pd.DataFrame(X_transformed.toarray(), columns=vectorizer.get_feature_names_out())

        # Combine vectorized text data with additional features
        X_final = pd.concat([X_transformed_df, additional_features.reset_index(drop=True)], axis=1)

        # Predict using the model
        y_pred = model.predict(X_final)

        # Calculate performance metrics
        accuracy = accuracy_score(y_holdout, y_pred)
        precision = precision_score(y_holdout, y_pred, average="weighted", zero_division=1)
        recall = recall_score(y_holdout, y_pred, average="weighted", zero_division=1)
        f1 = f1_score(y_holdout, y_pred, average="weighted", zero_division=1)

        # Define thresholds for the performance metrics
        expected_accuracy = 0.70
        expected_precision = 0.70
        expected_recall = 0.70
        expected_f1 = 0.70

        # Assert that the metrics meet the thresholds
        assert accuracy >= expected_accuracy, f"Accuracy {accuracy:.2f} is below threshold {expected_accuracy}"
        assert precision >= expected_precision, f"Precision {precision:.2f} is below threshold {expected_precision}"
        assert recall >= expected_recall, f"Recall {recall:.2f} is below threshold {expected_recall}"
        assert f1 >= expected_f1, f"F1 Score {f1:.2f} is below threshold {expected_f1}"

        print(f"Performance test passed for model '{model_name}' at stage '{stage}'")

    except Exception as e:
        pytest.fail(f"Model performance test failed: {e}")