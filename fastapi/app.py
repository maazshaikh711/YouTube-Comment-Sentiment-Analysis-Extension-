import mlflow
from mlflow.tracking import MlflowClient
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.dates as mdates
import os
from dotenv import load_dotenv

app = FastAPI()

# CORS middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (you can restrict to specific domains)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Pydantic models for request validation
class Comment(BaseModel):
    text: str
    timestamp: str
    authorId: str

class CommentsInput(BaseModel):
    comments: list[Comment]


class SentimentData(BaseModel):
    sentiment_data: list


class CommentResponse(BaseModel):
    comment: str
    sentiment: str
    timestamp: str


# Define the preprocessing function
def preprocess_comment(comment: str) -> tuple[str, int, int, float]:
    """
    Apply preprocessing transformations to a comment and calculate features.

    Parameters
    ----------
    comment : str
        The comment to preprocess.

    Returns
    -------
    tuple[str, int, int, float]
        A tuple containing the preprocessed comment, word count, character count,
        and average word length.
    """
    try:
        # Convert to lowercase
        comment = comment.lower()

        # Remove trailing and leading whitespaces
        comment = comment.strip()

        # Remove newline characters
        comment = re.sub(r'\n', ' ', comment)

        # Remove stopwords but retain important ones for sentiment analysis
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        # Calculate features
        word_count = len(comment.split())
        char_count = len(comment)
        avg_word_length = char_count / (word_count + 1)  # +1 to avoid division by zero

        # Remove non-alphanumeric characters, except punctuation
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment, word_count, char_count, avg_word_length
    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        return comment, 0, 0, 0  # Return default values in case of error
    

# Function to clean and limit comments
def clean_and_limit_comments(comments: list[Comment], max_tokens: int = 2000) -> str:
    """
    Clean and limit the comments to a maximum token count.

    :param comments: List of Comment objects to be processed.
    :param max_tokens: Maximum token count allowed in the output.
    :return: A single string containing the cleaned and limited comments without \n.
    """
    # Extract and clean the text from each comment
    cleaned_comments = []
    for comment in comments:
        sanitized_comment = re.sub(r"[^\S\r\n]+", " ", comment.text)  # Remove extra spaces
        sanitized_comment = sanitized_comment.replace("\\", "").strip()  # Remove backslashes
        cleaned_comments.append(sanitized_comment)

    # Format comments into a numbered list
    formatted_comments = [f"{i + 1}. {comment}" for i, comment in enumerate(cleaned_comments)]

    # Join comments into a single string
    combined_comments = "\n".join(formatted_comments)

    # Remove \n characters
    combined_comments = combined_comments.replace("\n", " ")

    # Token count estimation (adjust if necessary)
    # Assume 1 token ≈ 4 characters (change as needed based on your LLM's tokenizer).
    estimated_tokens = len(combined_comments) // 4

    if estimated_tokens <= max_tokens:
        return combined_comments

    # If token count exceeds the limit, truncate comments
    truncated_comments = []
    current_tokens = 0

    for comment in formatted_comments:
        comment_tokens = len(comment) // 4
        if current_tokens + comment_tokens > max_tokens:
            break
        truncated_comments.append(comment)
        current_tokens += comment_tokens

    truncated_combined_comments = " ".join(truncated_comments).replace("\n", " ")
    return truncated_combined_comments


# Load the model and vectorizer from the model registry
def load_model_from_registry(model_name: str, vectorizer_name: str):
    """
    Load the latest version of a model and its vectorizer from the MLflow Model Registry.

    Parameters
    ----------
    model_name : str
        The name of the model to load.
    vectorizer_name : str
        The name of the vectorizer to load.

    Returns
    -------
    tuple
        A tuple containing the loaded model and vectorizer.
    """
    try:
        # Set the MLflow tracking URI to DagsHub or your desired MLflow server
        # Set up DagsHub credentials for MLflow tracking
        # Load .env file only if not in GitHub Actions
        if not os.getenv("GITHUB_ACTIONS"):
            load_dotenv()

        dagshub_token = os.getenv("DAGSHUB_PAT")
        if not dagshub_token:
            raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
        mlflow.set_tracking_uri("https://dagshub.com/dakshvandanarathi/YT-Sentiment-Analyser.mlflow")
        
        # Initialize the MLflow client
        client = MlflowClient()
        
        # Get the latest version of the model
        latest_model_version = client.get_latest_versions(model_name, stages=["production"])[0].version
        model_uri = f"models:/{model_name}/{latest_model_version}"
        model = mlflow.pyfunc.load_model(model_uri)
        
        # Get the latest version of the vectorizer
        latest_vectorizer_version = client.get_latest_versions(vectorizer_name, stages=["production"])[0].version
        vectorizer_uri = f"models:/{vectorizer_name}/{latest_vectorizer_version}"
        vectorizer = mlflow.sklearn.load_model(vectorizer_uri)
        
        # Return the loaded model and vectorizer
        return model, vectorizer
    except Exception as e:
        print(f"Error loading model or vectorizer: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading model or vectorizer: {e}")

# Initialize the model and vectorizer
model, vectorizer = load_model_from_registry("sentiment_analysislgbm_model", "sentiment_analysisvectorizer")


@app.get("/")
async def home() -> dict[str, str]:
    """
    The root endpoint of the API.

    Returns:
    - dict[str, str]: A dictionary containing a welcome message.
    """
    return {"message": "Welcome to the FastAPI app"}


@app.post("/predict_with_timestamps", response_model=None)
async def predict_with_timestamps(data: CommentsInput) -> JSONResponse:
    """
    Predicts the sentiment of each comment in the provided list of comments.

    Args:
    - data (CommentsInput): A JSON object containing a list of comments.

    Returns:
    - A JSONResponse object with a list of dictionaries containing the original comment, predicted sentiment, and timestamp.
    """
    comments_data = data.comments

    if not comments_data:
        raise HTTPException(status_code=400, detail="No comments provided")

    try:
        comments = [item.text for item in comments_data]
        timestamps = [item.timestamp for item in comments_data]

        # Preprocess each comment before vectorizing
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        
        # Convert the results into a DataFrame for further processing
        processed_df = pd.DataFrame(preprocessed_comments, columns=["preprocessed_comment", "word_count", "char_count", "avg_word_length"])
        
        # Transform comments using the vectorizer
        test_tfidf = vectorizer.transform(processed_df['preprocessed_comment'].values).toarray()
        transformed_comments = np.hstack([test_tfidf, processed_df[['word_count', 'char_count', 'avg_word_length']].values])
        
        # Make predictions
        predictions = model.predict(transformed_comments).tolist()  # Convert to list

        # Create a mapping from 0, 1, 2 to -1, 0, 1
        prediction_mapping = {0: -1, 1: 0, 2: 1}

        # Map the predictions
        predictions = [prediction_mapping[pred] for pred in predictions]
        
        # Convert predictions to strings for consistency
        predictions = [str(pred) for pred in predictions]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    # Return the response with original comments, predicted sentiments, and timestamps
    response = [{"comment": comment, "sentiment": sentiment, "timestamp": timestamp} 
                for comment, sentiment, timestamp in zip(comments, predictions, timestamps)]
    return JSONResponse(content=response)


@app.post("/generate_wordcloud")
async def generate_wordcloud(data: CommentsInput):
    """
    Generate a word cloud image from the provided comments.

    This endpoint takes a list of comments as input, preprocesses the comments,
    and generates a word cloud image based on the preprocessed text. The word
    cloud is returned as a PNG image.

    Parameters
    ----------
    data : CommentsInput
        A Pydantic model containing a list of Comment objects.

    Returns
    -------
    StreamingResponse
        A streaming response containing the generated word cloud image in PNG format.

    Raises
    ------
    HTTPException
        If no comments are provided or if an error occurs during word cloud generation.
    """
    comments_data = data.comments
    try:
        comments = [item.text for item in comments_data]

        if not comments:
            raise HTTPException(status_code=400, detail="No comments provided")

        # Preprocess comments
        preprocessed_comments = [preprocess_comment(comment)[0] for comment in comments]

        # Combine all comments into a single string
        text = ' '.join(preprocessed_comments)

        # Generate the word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='black',
            colormap='Reds',
            stopwords=set(stopwords.words('english')),
            collocations=False
        ).generate(text)

        # Save the word cloud to a BytesIO object
        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)

        # Return the image as a response
        return StreamingResponse(img_io, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Word cloud generation failed: {str(e)}")


@app.post("/generate_trend_graph")
async def generate_trend_graph(sentiment_data: SentimentData):
    """
    Generate a sentiment trend graph over time.

    Takes a list of sentiment data points in the form of a list of dictionaries, each with a 'timestamp' and 'sentiment' key.
    The sentiment values are expected to be either -1, 0, or 1, and the timestamps should be in any format that can be parsed by pandas.

    The function will return a PNG image of the sentiment trend graph as a response.

    :param sentiment_data: A list of dictionaries, each containing a 'timestamp' and 'sentiment' key.
    :return: A PNG image of the sentiment trend graph.
    """
    
    try:
        if not sentiment_data.sentiment_data:
            raise HTTPException(status_code=400, detail="No sentiment data provided")

        # Convert sentiment_data to DataFrame
        df = pd.DataFrame(sentiment_data.sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Set the timestamp as the index
        df.set_index('timestamp', inplace=True)

        # Ensure the 'sentiment' column is numeric
        df['sentiment'] = df['sentiment'].astype(int)

        # Resample the data over monthly intervals and count sentiments
        monthly_counts = df.resample('ME')['sentiment'].value_counts().unstack(fill_value=0)

        # Calculate total counts per month
        monthly_totals = monthly_counts.sum(axis=1)

        # Calculate percentages
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        # Ensure all sentiment columns are present
        for sentiment_value in [-1, 0, 1]:
            if sentiment_value not in monthly_percentages.columns:
                monthly_percentages[sentiment_value] = 0.0

        # Plotting the sentiment trend graph
        plt.figure(figsize=(10, 6))
        for sentiment_value in [-1, 0, 1]:
            plt.plot(monthly_percentages.index, monthly_percentages[sentiment_value], label=f"Sentiment {sentiment_value}")

        # Formatting the graph
        plt.xlabel("Month")
        plt.ylabel("Sentiment Percentage")
        plt.title("Sentiment Trend Over Time")
        plt.legend()

        # Format the x-axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

        # Rotate and format x-axis labels
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Save the graph to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG')
        img_io.seek(0)
        plt.close()

        # Return the graph as an image
        return StreamingResponse(img_io, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate sentiment trend graph: {str(e)}")


# API route for cleaning and limiting comments
@app.post("/process_comments")
async def process_comments(input: CommentsInput) -> dict[str, str]:
    """
    API route for cleaning and limiting comments.

    This API takes a list of comments in the request body and returns a single string
    containing the cleaned and limited comments.

    The comments are cleaned by removing extra spaces, backslashes, and newline characters,
    and then formatted into a numbered list. The list is then joined into a single string
    without any newline characters.

    The comments are limited to a maximum of 2000 tokens. If the comments exceed this limit,
    they are truncated to fit within the limit.

    :param input: The list of comments to process
    :type input: CommentsInput
    :return: A dictionary containing the processed comments
    :rtype: dict[str, str]
    """
    try:
        # Process the comments and limit to 2000 tokens
        processed_comments: str = clean_and_limit_comments(input.comments, max_tokens=2000)
        return {"processed_comments": processed_comments}
    except Exception as e:
        return {"error": str(e)}
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
