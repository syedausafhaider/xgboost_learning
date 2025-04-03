# XGBoost Titanic Survival Prediction

This repository contains a complete implementation of an XGBoost model to predict survival on the Titanic dataset. The code is structured to be beginner-friendly while covering both basic and advanced concepts of XGBoost.

## Overview

I used the Titanic dataset from Kaggle to build a binary classification model that predicts whether a passenger survived or not. The dataset includes features like age, gender, fare, and cabin class, which are used to train the model. The implementation follows these steps:

## Step-by-Step Explanation

### Dataset Preparation
- Loaded the Titanic dataset and prepared the features (`X`) and target variable (`y`).
- Dropped irrelevant columns like `Name`, `Ticket`, and `Cabin` to simplify the dataset.
- Encoded categorical variables (`Sex` and `Embarked`) into numeric values using `LabelEncoder`.

### Data Splitting
- Split the dataset into training and validation sets to evaluate the model's performance on unseen data.

### DMatrix Conversion
- Converted the training and validation datasets into DMatrix format, optimized for XGBoost computations.

### Model Training
- Defined the hyperparameters for the XGBoost model, including:
  - Objective function: `binary:logistic`
  - Evaluation metric: `logloss`
  - Learning rate (`eta`)
- Trained the model using the training data while monitoring its performance on the validation set.
- Implemented a custom callback that inherits from `xgb.callback.TrainingCallback` to track training and validation metrics.

### Prediction and Evaluation
- Made predictions on the validation set.
- Evaluated the model using:
  - Accuracy
  - Confusion matrix
  - Precision
  - Recall
  - F1-score
  - ROC AUC score
- Visualized results using:
  - ROC curve
  - Confusion matrix heatmap
  - Feature importance plot

### Learning Curve
- Created a learning curve to visualize how the training and validation loss change over boosting rounds, helping diagnose overfitting or underfitting.

### Feature Importance
- Plotted the feature importance graph to understand which features contribute most to the model's predictions.

### Confusion Matrix Heatmap
- Visualized the confusion matrix as a heatmap to better understand classification results, including true positives, false positives, true negatives, and false negatives.

## Key Features

- **Custom Callback:** Implemented a custom callback to track training and validation metrics, ensuring compatibility with older XGBoost versions.
- **Graphs and Visualizations:** Included multiple visualizations (learning curve, ROC curve, feature importance, confusion matrix heatmap) for deeper model insights.
- **Evaluation Metrics:** Used various evaluation metrics to assess model performance comprehensively.
- **Modular Code:** Wrote the code in a modular way, making it easy to follow and adapt for other projects.

## How to Run the Program

### Install the required libraries:
```bash
pip install xgboost scikit-learn matplotlib seaborn pandas
```

### Download the Titanic dataset:
- Download the dataset from [Kaggle](https://www.kaggle.com/c/titanic/data) and save it as `train.csv` in the same directory as the script.

### Run the script:
```bash
python titanic_xgboost.py
```

### View the output:
- The console will display evaluation metrics like accuracy, precision, recall, and F1-score.
- Graphs will appear in separate windows or be saved as files, depending on your environment.

## Conclusion

This project demonstrates how to use XGBoost for binary classification tasks. By following this code, you can learn how to preprocess data, train an XGBoost model, evaluate its performance, and interpret results using visualizations. Feel free to experiment with different hyperparameters or datasets to enhance your understanding!

If you have any questions or suggestions, feel free to reach out. Happy coding! 😊

