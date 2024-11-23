import os
import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import pytest
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

def setup_mlflow_tracking():
    """
    Set up MLflow tracking URI and authentication using DAGSHub token.
    """
    if not os.getenv("GITHUB_ACTIONS"):
        load_dotenv()  # Load environment variables from .env file if not in GitHub Actions
    
    dagshub_token = os.getenv("DAGSHUB_PAT")
    if not dagshub_token:
        raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
    mlflow.set_tracking_uri("https://dagshub.com/dakshvandanarathi/YT-Sentiment-Analyser.mlflow")
    print("MLflow tracking URI is set to:", mlflow.get_tracking_uri())

def load_model_from_registry(model_name: str, vectorizer_name: str, stage: str):
    """
    Load the latest version of the model and its vectorizer from MLflow Model Registry.
    
    Parameters:
    - model_name (str): The name of the model to load.
    - vectorizer_name (str): The name of the vectorizer to load.
    - stage (str): The stage (e.g., "staging") to load the model from.

    Returns:
    - tuple: A tuple containing the loaded model and vectorizer.
    """
    # Initialize the MLflow client
    client = MlflowClient()

    # Load the latest model version from the registry
    try:
        latest_model_version = client.get_latest_versions(model_name, stages=[stage])[0].version
        model_uri = f"models:/{model_name}/{latest_model_version}"
        model = mlflow.pyfunc.load_model(model_uri)
    except IndexError:
        raise ValueError(f"No version found for model '{model_name}' in stage '{stage}'")
    except Exception as e:
        raise RuntimeError(f"Failed to load model '{model_name}': {str(e)}")

    # Load the latest vectorizer version from the registry
    try:
        latest_vectorizer_version = client.get_latest_versions(vectorizer_name, stages=[stage])[0].version
        vectorizer_uri = f"models:/{vectorizer_name}/{latest_vectorizer_version}"
        vectorizer = mlflow.sklearn.load_model(vectorizer_uri)
    except IndexError:
        raise ValueError(f"No version found for vectorizer '{vectorizer_name}' in stage '{stage}'")
    except Exception as e:
        raise RuntimeError(f"Failed to load vectorizer '{vectorizer_name}': {str(e)}")

    return model, vectorizer

@pytest.mark.parametrize("model_name, vectorizer_name, stage", [
    ("sentiment_analysislgbm_model", "sentiment_analysisvectorizer", "staging"),
])
def test_load_latest_staging_model(model_name, vectorizer_name, stage):
    """
    Test function to load the latest model and vectorizer from the MLflow registry.
    """
    try:
        # Setup environment and MLflow tracking URI
        setup_mlflow_tracking()
        
        # Load model and vectorizer from MLflow registry
        model, vectorizer = load_model_from_registry(model_name, vectorizer_name, stage)

        # Ensure the model and vectorizer are loaded successfully
        assert model is not None, f"Failed to load model '{model_name}'"
        assert vectorizer is not None, f"Failed to load vectorizer '{vectorizer_name}'"
        
        print(f"Model '{model_name}' and vectorizer '{vectorizer_name}' loaded successfully from '{stage}' stage.")
    
    except Exception as e:
        pytest.fail(f"Loading failed with error: {e}")