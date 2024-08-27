# Bank Term Deposit Predictions

## Project Overview
This project aims to predict customer subscription to a term deposit using machine learning techniques. The dataset consists of various features from a direct marketing campaign by a Portuguese banking institution.


## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [MLflow Integration](#mlflow-integration)
- [Contributing](#contributing)
- [License](#license)

## Installation
To get started, you'll need to install the project dependencies. Make sure you have Python installed. Then, create a virtual environment and install the required packages using:

pip install -r requirements.txt

## Usage

## **Data Preprocessing**

**In this section**, we cover the steps required to preprocess the data before feeding it into the machine learning models.

**Steps include:**

1. **Loading Data**: Load the training and test datasets.
2. **Handling Missing Values**: Address any missing values in the dataset.
3. **Encoding Categorical Variables**: Convert categorical variables into numerical format.
4. **Feature Scaling**: Normalize or standardize features if needed.
5. **Splitting Data**: Split the data into training and validation sets.

For detailed code implementation, refer to the `data_preprocessing.py` script

## **Model Training**

**In this section**, we detail the process for training machine learning models on the preprocessed data.

**Steps include:**

1. **Model Selection**: Choose and initialize the machine learning model(s) to be used.
2. **Training the Model**: Fit the model on the training data.
3. **Hyperparameter Tuning**: Adjust the model's hyperparameters to optimize performance.
4. **Saving the Model**: Save the trained model for future use or evaluation.
5. **Logging with MLflow**: Log the model and metrics using MLflow for tracking and comparison.

For detailed code implementation, refer to the `model_training.py` script.

## Data
The dataset used in this project is from a direct marketing campaign by a Portuguese bank. It includes information about customer demographics, previous marketing campaigns, and the outcome of the campaign.

train.csv: Training dataset
test.csv: Testing dataset

## Model Training
The model training script uses a Random Forest Classifier to predict customer subscription to a term deposit. Hyperparameters like n_estimators and max_depth are adjustable.

## Evaluation
The model's performance is evaluated using metrics such as accuracy, precision, recall, and F1-score. The results are logged and can be reviewed using MLflow.

## MLflow Integration
MLflow is used for tracking experiments, logging parameters, metrics, and model artifacts. The logs can be reviewed using the MLflow UI.

## Contributing
Contributions are welcome! Please follow the guidelines below:

Fork the repository.
Create a new branch for your feature or fix.
Submit a pull request with a clear description of the changes.