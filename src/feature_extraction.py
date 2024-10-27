import numpy as np
import pandas as pd
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
from utils import Logger, load_params, load_data, save_data, save_model


# Initialize logger
logger = Logger('feature_extraction', logging.INFO)

def apply_tfidf(
    train_data: pd.DataFrame, max_features: int, ngram_range: tuple
) -> pd.DataFrame:
    """
    This function applies TF-IDF transformation to the comment column of a given DataFrame containing text data.
    It uses the TfidfVectorizer from scikit-learn to perform the transformation.

    Parameters
    ----------
    train_data : pd.DataFrame
        The DataFrame containing the text data.
    max_features : int
        The maximum number of features to keep.
    ngram_range : tuple
        The range of ngrams to consider.

    Returns
    -------
    pd.DataFrame
        The transformed data with TF-IDF features and the original columns.
    """
    try:
        train_data.dropna(inplace=True)
        
        # Perform TF-IDF transformation
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        train_vec = vectorizer.fit_transform(train_data['comment']).toarray()
        train_combined = np.hstack([train_vec, train_data[['word_count', 'char_count', 'avg_word_length', 'category']].values])

        if train_vec.shape[1] + 4 != train_combined.shape[1]:
            raise ValueError('The number of features in the transformed data does not match the expected number')

        train_tfidf = pd.DataFrame(train_combined, columns=list(vectorizer.get_feature_names_out()) + ['word_count', 'char_count', 'avg_word_length', 'category'])

        logger.info(f"TF-IDF transformation complete. Training data shape: {train_tfidf.shape}")

        # Save the vectorizer in the root directory
        vectorizer_path = Path(__file__).parent.parent / 'models'
        save_model(logger, vectorizer, vectorizer_path, 'tfidf_vectorizer')
        return train_tfidf
    except Exception as e:
        logger.error(f'Error during TF-IDF transformation: {str(e)}')
        raise


def main() -> None:
    """
    Execute the feature extraction pipeline.

    The pipeline involves loading the preprocessed data, applying TF-IDF transformation,
    and saving the transformed data to a given directory.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    try:
        logger.info("Starting feature extraction pipeline...")

        # Load the parameters
        params = load_params(logger)['feature_extraction']

        # Load the data
        data_path = Path(__file__).parent.parent / 'data'
        train_data = load_data(logger, data_path / 'interim' / 'train_processed.csv')
        
        # Apply TF-IDF transformation
        train_tfidf = apply_tfidf(train_data, params['max_features'], tuple(params['ngram_range']))

        # Create data/processed directory
        data_path = data_path / 'processed'
        data_path.mkdir(parents=True, exist_ok=True)

        # Save the transformed data
        save_data(logger, train_tfidf, data_path, 'train_tfidf.csv')
        logger.info("Feature extraction pipeline completed successfully.")
    except Exception as e:
        logger.error(f'Error during feature extraction:{str(e)}')


if __name__ == '__main__':
    main()