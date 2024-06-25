# Parkinson's Disease Prediction Project

## Introduction

This project aims to predict whether a person has Parkinson's disease based on various biomedical voice measurements. The analysis covers the entire pipeline from data loading and preprocessing to model training, evaluation, and making predictions on new data.

## Necessary Libraries

The following libraries are used in this project:

- `numpy`: for numerical operations.
- `pandas`: for data manipulation and analysis.
- `scikit-learn`: for machine learning models and evaluation metrics.

## Dataset

The dataset used in this project contains biomedical voice measurements from 31 people, 23 with Parkinson's disease. The main features include measurements such as average vocal fundamental frequency, variation in fundamental frequency, and others.

### Dataset Description

- `name`: Name of the subject.
- `status`: Health status (1 = Parkinson's, 0 = healthy).
- `MDVP:Fo(Hz)`, `MDVP:Fhi(Hz)`, `MDVP:Flo(Hz)`: Various measures of vocal fundamental frequency.
- Other columns: Various measures of variation in amplitude, period, and other properties of the voice.

## Data Loading and Preprocessing

The dataset is loaded and the first few rows are displayed to understand its structure. Initial data cleaning and preparation steps include checking dataset information and describing its statistics. The data is then split into training and testing sets to evaluate the performance of the model.

### Splitting Data into Training and Testing Sets

The features and labels are separated, and the data is split into training and testing sets with an 80-20 split.

### Standardizing Data

Standardization is performed to ensure that all features contribute equally to the model performance. This involves scaling the training data and applying the same transformation to the testing data.

## Model Training

A Support Vector Machine (SVM) classifier with a linear kernel is used for training. The model is trained using the standardized training data.

## Model Evaluation

### Evaluating the Model

The model's performance is evaluated on both training and testing data. The accuracy score is used as the evaluation metric. Both the training and testing accuracies are found to be 87.18%.

### Analysis of Results

Both the training and testing accuracies being 87.18% indicates that the model performs consistently on both training and testing data. This suggests that the model is neither overfitting nor underfitting, which is a desirable property.

## Making Predictions

### Building a Predictive Model

A predictive model is built using new input data. The input data is standardized using the previously fitted scaler, and the model is used to make a prediction. The output indicates whether the person is healthy or has Parkinson's disease.

### Analysis of New Predictions

The model can correctly predict new data based on the provided measurements by standardizing the input data before making a prediction. The output indicates whether the person is healthy or has Parkinson's disease.

## Conclusion

The SVM model with a linear kernel trained in this project has shown to be effective in predicting Parkinson's disease with an accuracy of 87.18%. To further improve the model, other data preprocessing techniques, machine learning algorithms, and cross-validation could be explored.

## Future Work

Future work could involve exploring more advanced machine learning algorithms, such as Random Forests or Gradient Boosting Machines. Additionally, feature engineering techniques and hyperparameter tuning could be employed to enhance the model's performance. Cross-validation should be used to obtain a more robust estimate of the model's performance.

## Repository Structure

- `parkinson_dataset.csv`: The dataset used in this project.
- `Parkinson_Project.ipynb`: Jupyter notebook containing the code and analysis.
- `README.md`: This file, providing an overview and description of the project.


