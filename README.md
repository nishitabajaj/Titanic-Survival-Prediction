# Titanic Survival Prediction

This project demonstrates the use of machine learning to predict the survival of passengers aboard the Titanic based on various features such as age, sex, fare, class, and other factors. The goal is to create a predictive model that can help us determine whether a passenger survived or not.

## Project Overview

The dataset used in this project is from the famous Titanic dataset, which contains information about passengers on the Titanic, such as their age, sex, fare, class, and whether they survived. This project aims to predict the survival of a passenger using machine learning techniques.

## Features:
- **Age**: The age of the passenger.
- **Fare**: The fare paid by the passenger.
- **Sex**: The gender of the passenger (male or female).
- **Embarked**: The port where the passenger boarded the Titanic.
- **Survived**: The target variable (0 = did not survive, 1 = survived).

## Steps in the Process

1. **Data Preprocessing**:
   - Missing values are handled (mean imputation for age, median for fare).
   - Irrelevant columns (e.g., `Cabin`, `PassengerId`, `Name`, `Ticket`) are dropped.
   - Categorical variables like `Sex` and `Embarked` are encoded using one-hot encoding.

2. **Model Training**:
   - A Random Forest Classifier is trained to predict survival.
   - The dataset is split into training and testing sets using an 80-20 split.

3. **Model Evaluation**:
   - The model is evaluated using various metrics, including:
     - **Accuracy**
     - **Precision**
     - **Recall**
     - **F1 Score**

4. **Model Deployment**:
   - The trained model is saved using `pickle` for future use.
   - The model can be loaded and used to make predictions on new data.

## Files in this Repository:
- `titanic_survival_model.py`: Python code for training and evaluating the model.
- `random_forest_model.pkl`: Saved Random Forest model.
- `README.md`: This file providing details about the project.

## Requirements:
The following libraries are required to run this project:
- `pandas`
- `numpy`
- `scikit-learn`

You can install the dependencies using the following command:

```bash
pip install -r requirements.txt
