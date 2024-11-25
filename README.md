# Sentiment Analysis

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>
<a target="_blank" href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3.10.15%20-3776AB?logo=python&" />
</a>
<a target="_blank" href="https://numpy.org/">
    <img src="https://img.shields.io/badge/Numpy-Scientific%20Computing-013243?logo=numpy" />
</a>
<a target="_blank" href="https://pandas.pydata.org/">
    <img src="https://img.shields.io/badge/Pandas-Data%20Manipulation-150458?logo=pandas" />
</a>
<a target="_blank" href="https://scikit-learn.org/">
    <img src="https://img.shields.io/badge/scikit--learn-Machine%20Learning-F7931E?logo=scikit-learn" />
</a>
<a target="_blank" href="https://matplotlib.org/">
    <img src="https://img.shields.io/badge/Matplotlib-Visualization-EE4C2C?logo=matplotlib" />
</a>
<a target="_blank" href="https://seaborn.pydata.org/">
    <img src="https://img.shields.io/badge/Seaborn-Visualization-FF5D47?logo=seaborn" />
</a>
<a target="_blank" href="https://lightgbm.readthedocs.io/">
    <img src="https://img.shields.io/badge/LightGBM-Machine%20Learning%20Framework-71B34D?logo=lightgbm" />
</a>
<a target="_blank" href="https://optuna.org/">
    <img src="https://img.shields.io/badge/Optuna-Optimization%20Framework-4C8C29?logo=optuna" />
<a target="_blank" href="https://docs.pytest.org/">
    <img src="https://img.shields.io/badge/pytest-Testing Framework-0A9EDC?logo=pytest" />
</a>
<a target="_blank" href="https://fastapi.tiangolo.com/">
    <img src="https://img.shields.io/badge/FastAPI-Web%20Framework-009688?logo=fastapi" />
</a>
<a target="_blank" href="https://www.docker.com/">
    <img src="https://img.shields.io/badge/Docker-Platform-2496ED?logo=docker" />
</a>
<a target="_blank" href="https://dvc.org/">
    <img src="https://img.shields.io/badge/DVC-Data%20Version%20Control-45B8D8?logo=dvc" />
</a>
<a target="_blank" href="https://mlflow.org/">
    <img src="https://img.shields.io/badge/MLflow-Machine%20Learning-00A9E0?logo=mlflow" />
</a>
<a target="_blank" href="https://dagshub.com/">
    <img src="https://img.shields.io/badge/Dagshub-ML%20Operations-3F8CFF?logo=dagshub" />
</a>
<a target="_blank" href="https://github.com/features/actions">
    <img src="https://img.shields.io/badge/GitHub%20Actions-Continuous%20Integration-2088FF?logo=github-actions" />
</a>
<a target="_blank" href="https://aws.amazon.com/ecr/">
    <img src="https://img.shields.io/badge/AWS%20ECR-Container%20Registry-232F3E?logo=amazon-aws" />
</a>
<a target="_blank" href="https://aws.amazon.com/ec2/">
    <img src="https://img.shields.io/badge/AWS%20EC2-Cloud%20Computing-FF9900?logo=amazon-aws" />
</a>

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)

## Project Overview

This sentiment analysis project aims to classify YouTube comments into positive, neutral, or negative sentiments using machine learning techniques. The key components of the project include:

- **Model Training**: The project uses **LightGBM** as the base model for its high performance. Various feature extraction methods like **TF-IDF** and resampling techniques are employed to handle class imbalances and optimize model performance.

- **Backend**: A **FastAPI** application serves the trained model for real-time predictions, exposing API endpoints to interact with the model.

- **MLflow & DVC**: **MLflow** is used for model tracking and versioning, while **DVC** is utilized for data and model version control, ensuring reproducibility and efficient pipeline management.

Additionally, this repository focuses on the **backend** of the project. For the **frontend**, a browser extension/plugin has been created to interact with the model. You can explore the [frontend repository here](https://github.com/DakshRathi/YT-Chrome-Plugin-Frontend).


## Project Structure

The project follows a well-defined structure that organizes code, data, models, and configurations in separate directories:

```plaintext
└── 📁sentiment_analysis
    └── 📁.github
        └── 📁workflows
            └── ⚙️ cicd.yaml
    └── 📁deploy
        └── 📁scripts
            └── download_env.sh
            └── install_dependencies.sh
            └── start_docker.sh
    └── 📁data
        └── 📁interim
            └── test_processed.csv
            └── train_processed.csv
        └── 📁processed
            └── train_target.csv
            └── train_tfidf.csv
        └── 📁raw
            └── test.csv
            └── train.csv
        └── 📁visualizations
            └── 🖼️confusion_matrix_Test_Data.png
    └── 📁fastapi
        └── app.py
        └── 📜requirements.txt
    └── 📁models
        └── 💾lgbm_model.joblib
        └── 💾tfidf_vectorizer.joblib
    └── 📁notebooks
        └── preprocessing_eda.ipynb
        └── exp_1_baseline_model.ipynb
        └── exp_2_bow_tfidf_word2vec.ipynb
        └── exp_3_handling_imbalanced_data.ipynb
        └── exp_4_tuning_ml_algo.ipynb
        └── exp_4_tuning_ml_algo_2.ipynb
        └── exp_5_lightGBM_final.ipynb
    └── 📁scripts
        └── load_model_test.py
        └── performance_test.py
        └── promote_model.py
        └── fastapi_test.py
    └── 📁src
        └── data_ingestion.py
        └── data_preprocessing.py
        └── feature_extraction.py
        └── model_building.py
        └── model_evaluation.py
        └── register_model.py
        └── utils.py
    └── .dvcignore
    └── .env
    └── .gitignore
    └── 🐳Dockerfile
    └── 📝appspec.yml
    └── dvc.lock
    └── 📝dvc.yaml
    └── experiment_info.json
    └── Makefile
    └── 📝params.yaml
    └── pyproject.toml
    └── README.md
    └── 📜requirements.txt
```

### 📁 Notebooks Folder

#### **1. preprocessing_eda.ipynb**  
This notebook focuses on **Exploratory Data Analysis (EDA)** and **preprocessing** steps. It includes:  
- **Data loading**: Imported and explored the Reddit sentiment dataset.  
- **Cleaning and preprocessing**: Removed missing values, duplicates, URLs, and non-English characters; converted text to lowercase; and applied lemmatization.  
- **Feature engineering**: Added columns for word count, character count, and punctuation count.  
- **Visualization**: Analyzed class distribution, word counts, stop words for each sentiment category.



#### **2. exp_1_baseline_model.ipynb**  
This notebook builds a baseline **Random Forest model** for sentiment analysis. Key steps:  
- **Dataset preparation**: Loaded preprocessed data (`preprocessed_data.csv`). Split data into training (80%) and testing (20%) sets using stratified sampling.  
- **Feature extraction**: Vectorized comments using **Bag of Words (CountVectorizer)** with a max feature size of 10,000. Combined vectorized features with original dataset columns.  
- **Model training**: Used a **Random Forest Classifier** (`n_estimators=200`, `max_depth=15`) as a baseline model. Trained on vectorized and combined features.  
- **Evaluation**: Calculated accuracy and detailed metrics for each class using **classification report**.  Visualized results with a **confusion matrix**.  
- **Logging with MLflow**: Logged experiment parameters, metrics, model artifacts, and dataset details to MLflow. Saved confusion matrix as a plot artifact.  
- **Accuracy**: Achieved baseline accuracy and recorded detailed class-wise metrics.  


#### **3. exp_2_bow_tfidf_word2vec.ipynb**  
This notebook compares **BoW**, **TF-IDF**, and **Word2Vec** vectorization techniques for sentiment analysis, while optimizing hyperparameters using **Optuna**.  

- **Dataset preparation**: Dataset preparation included loading preprocessed data (`preprocessed_data.csv`), dropping missing comments, and retaining features like word count, character count, and average word length.  
- **Vectorization**: Implemented vectorization techniques including **CountVectorizer** (BoW), **TfidfVectorizer**, and **Word2Vec**; Optuna tuned hyperparameters such as `vectorizer_type` (BoW, TF-IDF, Word2Vec), `ngram_range` for BoW and TF-IDF, `max_features` for BoW and TF-IDF, and `vector_size` for Word2Vec.  
- **Model training**: Trained a **Random Forest Classifier** (`n_estimators=200`, `max_depth=15`) using combined vectorized features with additional features for training and testing.  
- **Hyperparameter optimization**: Ran **200 trials** with Optuna to optimize hyperparameters and logged each trial's results to **MLflow**, including metrics, parameters, and confusion matrix plots.  
- **Evaluation**: Achieved the best accuracy and identified the optimal vectorization method and hyperparameters, with the highest accuracy and best parameters displayed.


#### **4. exp_3_handling_imbalanced_data.ipynb**  
This notebook explores various strategies for handling imbalanced datasets in sentiment analysis using **TF-IDF** vectorization and resampling techniques.  

- Loaded preprocessed data (`preprocessed_data.csv`) and extracted features like word count, character count, and average word length. Split the dataset into training and testing sets and vectorized the text data using **TF-IDF** with hyperparameters (`max_features=1006`, `ngram_range=(1, 2)`), combining vectorized text with additional features.  
- Experimented with multiple resampling techniques: **Random Undersampling**, **Tomek Links**, **Centroid Clustering**, **NearMiss Undersampling**, **Random Oversampling**, **SMOTE**, **ADASYN**, **Borderline SMOTE**, **SMOTETomek**, **SMOTEENN**, and **class_weight='balanced'**.  
- Trained a **Random Forest Classifier** (`n_estimators=200`, `max_depth=15`) on resampled data for each technique and evaluated performance using **accuracy score**, **classification report**, and **confusion matrix**. Logged all parameters, metrics, and confusion matrix plots to **MLflow** with appropriate tags for each technique.  
- Generated confusion matrices for visual analysis of predictions, logged artifacts for all runs, and identified the impact of resampling techniques on model performance. Experiment concluded with all results logged and accuracy improvements analyzed.  

#### **5. exp_4_tuning_ml_algo.ipynb & exp_4_tuning_ml_algo_2.ipynb**

This notebook is focused on hyperparameter tuning and improving sentiment analysis performance:
- **Optuna Optimization**: Implements Optuna for multi-objective hyperparameter tuning across various models, including Random Forest, Logistic Regression, Naive Bayes, SVM, XGBoost and LightGBM.
- **Resampling Techniques**: Applies ADASYN for handling class imbalance in the dataset.
- **Experiment Tracking**: Uses MLflow with Dagshub integration for logging parameters, metrics, and artifacts like confusion matrices.
- **Model Evaluation**: Evaluates models on accuracy, F1-score, and classification metrics while tracking their performance across trials.

#### **6. exp_5_lightGBM_final.ipynb**

This notebook demonstrates detailed hyperparameter tuning and evaluation of a LightGBM model for sentiment analysis on YouTube comments:
- **Model Optimization**: Optuna optimizes hyperparameters like n_estimators, max_depth, and learning_rate over `150 trials`.
- **Evaluation**: The model is evaluated using accuracy, F1 score, and a confusion matrix.
- **Prediction Function**: A function predicts sentiment for new comments, returning the sentiment and its confidence.



### 📁 SRC Folder
Contains all the source code and scripts for different stages of the ML pipeline. The project uses **DVC** (Data Version Control) for tracking data and model versions, ensuring that experiments are reproducible and for building pipeline. The trained models are also registered in **MLflow Model Registry** for version management and easier deployment.
- **data_ingestion.py**: Handles the process of loading and collecting data.
- **data_preprocessing.py**: Performs necessary preprocessing tasks such as text cleaning, handling missing values, etc.
- **feature_extraction.py**: Implements feature extraction technique `TF-IDF`.
- **model_building.py**: Contains code for defining, training, and tuning `LightGBM` models.
- **model_evaluation.py**: Provides functions for evaluating the model’s performance using metrics like accuracy, precision, recall, etc.
- **register_model.py**: Registers the trained model into MLflow Model Registry for version control and easy tracking.
- **utils.py**: Contains helper functions for tasks that are used across the pipeline (e.g., logging, exception handling, loading, etc.).

### 📝 dvc.yaml
Defines the DVC pipeline, describing how data and models flow through the different stages of the project.
### 📄 dvc.lock
Locks the versions of the dependencies and ensures reproducibility across environments.

### 📝 params.yaml
This file contains the hyperparameters and settings for different stages of the machine learning pipeline. It includes data ingestion, feature extraction, and model building parameters.

### 📁 Scripts Directory

This directory contains various Python scripts used for testing, evaluating, and promoting machine learning models, as well as for testing the FastAPI backend using `pytest`.

- **1. load_model_test.py**: Tests the loading vectorizer machine learning model from model registry.
   
- **2. performance_test.py**: This script is typically used for assessing the accuracy, precision, recall and F1 score metrics for model evaluation.
   
- **3. promote_model.py**: Handles the model promotion process. This script is used to transition the model from staging phase to production. It ensures that the best-performing model is deployed.
   
- **4. fastapi_test.py**: Tests the FastAPI endpoints used for making predictions. This script ensures that the backend API is functional, and the model predictions are returned correctly.
   
Each of these scripts is critical for testing and deploying models and ensuring the API works correctly for real-time predictions.


### 📁 FastAPI Directory

This directory contains the FastAPI application and its dependencies used for serving machine learning model predictions via a backend API.

- **1. app.py**: The main FastAPI application file that defines the API endpoints. It handles incoming HTTP requests, loads the trained machine learning model, and serves predictions based on user inputs. The FastAPI app is used to expose model functionality to external services or users.

- **2. requirements.txt**: A text file that lists the dependencies required to run the FastAPI application. It includes libraries and tools such as FastAPI, Uvicorn, and any other necessary dependencies for the model serving functionality.
   
This directory is key for deploying the machine learning model as a service and interacting with it through HTTP requests.

### 🐳 Dockerfile
This `Dockerfile` defines a multi-stage build for creating a Docker image to deploy the FastAPI application with the machine learning model.Stages in the Dockerfile are:

- **1. Builder Stage**
   - **Base Image**: `python:3.10-slim`
   - **Purpose**: Sets up the environment for building the application by installing the necessary dependencies and libraries.

- **2. Runtime Image**
   - **Base Image**: `python:3.10-slim`
   - **Purpose**: The final image used to run the application, which is based on a slimmed-down Python image.

The use of a multi-stage build optimizes the Docker image by separating the build and runtime environments. The builder stage contains all the tools and dependencies needed to install libraries, while the runtime image only includes the essential libraries and application code, making the final image smaller and more efficient.

### ⚙️ **CICD Pipeline**

This GitHub Actions pipeline automates the deployment and testing of a machine learning model as a FastAPI service with integration to AWS services such as S3, ECR, and CodeDeploy.

---

#### 🔍 **Pipeline Steps**

1. **Checkout Code**
   : Pulls the latest code from the GitHub repository using `actions/checkout@v3`.

2. **Set up Python**
   : Configures Python 3.10 using `actions/setup-python@v2` for compatibility with the required dependencies.

3. **Cache pip Dependencies**
   : Caches Python package installations for faster subsequent runs using `actions/cache@v3`.

4. **Install Dependencies**
   : Installs all the required libraries listed in `requirements.txt`.

#### 🔄 **Data Versioning and Processing**

5. **Run Pipeline**
   : Executes the DVC pipeline (`dvc repro`) to preprocess and prepare data, with AWS credentials and DAGsHub authentication passed as environment variables.

6. **Push DVC-tracked Data**
   : Uploads processed data and outputs tracked by DVC to a remote storage (AWS S3).


#### 📝 **Code Management**

7. **Configure Git**
   : Sets up Git user details for commits by GitHub Actions.

8. **Add and Commit Changes**
   : Stages and commits changes, such as updated DVC outputs, to the repository.

9. **Push Changes**
   : Pushes committed changes back to the repository.

#### 🧪 **Testing and Validation**

10. **Run Model Loading Test**
    : Verifies the model loading script with `pytest` to ensure the model loads correctly.

11. **Run Model Performance Test**
    : Validates the model’s performance by executing a performance test script.

12. **Run FastAPI Tests**
    : Tests the FastAPI endpoints for functionality using `pytest`.

#### 🚀 **Deployment Steps**

13. **Start FastAPI App**
    : Launches the FastAPI application in the background using `nohup`.

14. **Stop FastAPI App**
    : Ensures the running FastAPI application is terminated after testing.

15. **Login to AWS ECR**
    : Authenticates Docker with AWS Elastic Container Registry (ECR) for image uploads.

16. **Build Docker Image**
    : Builds a Docker image named `yt-plugin` for the FastAPI application.

17. **Tag Docker Image**
    : Tags the built Docker image for ECR.

18. **Push Docker Image to AWS ECR**
    : Uploads the Docker image to AWS ECR for containerized deployment.


#### 📦 **Packaging for Deployment**

19. **Zip Files for Deployment**
    : Compresses deployment files (`appspec.yml` and necessary scripts) into a `deployment.zip`.

20. **Upload ZIP and Secret to S3** : Uploads the deployment package and environment variables to an S3 bucket.


#### 📤 **AWS CodeDeploy Integration**

21. **Deploy to AWS CodeDeploy** : Initiates deployment through AWS CodeDeploy, specifying the application, deployment group, and configuration settings.

--- 

This pipeline ensures seamless integration and deployment by leveraging GitHub Actions, DVC, AWS services, and Docker, automating every step from preprocessing to deployment, while running rigorous tests at each stage.



### 📁 **Deploy Directory**

The `deploy` directory includes `scripts` folder which contains all necessary configuration files for deploying the application via AWS CodeDeploy.


1. **download_env.sh** :
    - Downloads environment variables (`dagshub.env`) from an S3 bucket.

2. **install_dependencies.sh**  
   - Installs Docker, AWS CLI, and other necessary utilities.  
   - Configures Docker to run without `sudo` and enables it as a service.

3. **start_docker.sh**  
   - Loads environment variables and verifies them.  
   - Logs in to AWS ECR and pulls the latest Docker image.  
   - Stops and removes any existing container before starting a new one.  
   - Cleans up sensitive files after deployment.


### 📝 **appspec.yml**

Defines deployment lifecycle hooks:

- **BeforeInstall**: Runs `install_dependencies.sh` and `download_env.sh` to set up dependencies and environment variables.  
- **ApplicationStart**: Executes `start_docker.sh` to pull and run the Docker container.

--- 

This setup ensures a smooth and automated deployment process, adhering to security and efficiency standards.