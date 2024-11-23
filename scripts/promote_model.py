from mlflow.tracking import MlflowClient
from load_model_test import setup_mlflow_tracking

def promote_to_production(model_name: str, vectorizer_name: str):
    """
    Promote the latest model and vectorizer from Staging to Production.

    Args:
        model_name (str): Name of the model in the MLflow registry.
        vectorizer_name (str): Name of the vectorizer in the MLflow registry.
    """
    try:
        # Set up MLflow tracking
        setup_mlflow_tracking()

        # Initialize MLflow client
        client = MlflowClient()

        # Get the latest staging versions for model and vectorizer
        latest_model_staging = client.get_latest_versions(model_name, stages=["Staging"])
        latest_vectorizer_staging = client.get_latest_versions(vectorizer_name, stages=["Staging"])

        if not latest_model_staging or not latest_vectorizer_staging:
            raise ValueError(f"Staging versions not found for model '{model_name}' or vectorizer '{vectorizer_name}'.")

        latest_model_staging_version = latest_model_staging[0].version
        latest_vectorizer_staging_version = latest_vectorizer_staging[0].version

        # Archive existing production models and vectorizers
        prod_model_versions = client.get_latest_versions(model_name, stages=["Production"])
        prod_vectorizer_versions = client.get_latest_versions(vectorizer_name, stages=["Production"])

        for version in prod_model_versions:
            client.transition_model_version_stage(
                name=model_name,
                version=version.version,
                stage="Archived"
            )
            print(f"Archived model version {version.version} from 'Production'.")

        for version in prod_vectorizer_versions:
            client.transition_model_version_stage(
                name=vectorizer_name,
                version=version.version,
                stage="Archived"
            )
            print(f"Archived vectorizer version {version.version} from 'Production'.")

        # Promote the latest staging versions to production
        client.transition_model_version_stage(
            name=model_name,
            version=latest_model_staging_version,
            stage="Production"
        )
        print(f"Promoted model version {latest_model_staging_version} to 'Production'.")

        client.transition_model_version_stage(
            name=vectorizer_name,
            version=latest_vectorizer_staging_version,
            stage="Production"
        )
        print(f"Promoted vectorizer version {latest_vectorizer_staging_version} to 'Production'.")

    except Exception as e:
        print(f"Error promoting model/vectorizer: {e}")

if __name__ == "__main__":
    # Model and vectorizer names
    promote_to_production("sentiment_analysislgbm_model", "sentiment_analysisvectorizer")