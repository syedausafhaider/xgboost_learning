# XGBoost Titanic Survival Prediction with Custom Objective

This repository demonstrates how I built an XGBoost model to predict survival on the Titanic dataset. I focused on handling missing values, encoding categorical variables, training the model with a custom logistic loss function, and visualizing feature importance. Below, I explain each step in detail.

---

## Overview

I used the Titanic dataset to predict whether a passenger survived using XGBoost. The goal was to:

- Preprocess the data (handle missing values and encode categorical variables)
- Train a model with a custom objective function for binary classification
- Evaluate the model and visualize feature importance

---

## Step-by-Step Explanation

### 1. Loading the Dataset
- Loaded the Titanic dataset using `pandas.read_csv`
- Used features like `Age`, `Sex`, `Fare`, and `Embarked` to predict the `Survived` target

### 2. Handling Missing Values
- Identified missing values using `data.isnull().sum()`
- Filled numeric columns (`Age`, `Fare`) with column mean
- Filled categorical column (`Embarked`) with the most frequent value ('S')

### 3. Encoding Categorical Variables
- Encoded `Sex` and `Embarked` using `LabelEncoder`
- Ensured all features were numeric for XGBoost

### 4. Splitting the Dataset
- Used `train_test_split` to split the dataset into 80% training and 20% validation

### 5. Converting Data to DMatrix
- Converted training and validation sets to `xgb.DMatrix` format for optimized computation

### 6. Defining a Custom Objective Function
- Defined a custom logistic loss function:
  ```python
  def custom_logistic_obj(y_pred, y_true):
      y_true = y_true.get_label()
      preds = 1 / (1 + np.exp(-y_pred))
      grad = preds - y_true
      hess = preds * (1 - preds)
      return grad, hess
  ```

### 7. Training the Model
- Trained model with parameters:
  - `max_depth=6`
  - `eta=0.1`
- Used `early_stopping_rounds=10` to avoid overfitting

### 8. Making Predictions
- Applied sigmoid to raw outputs to get probabilities
- Used threshold of `0.5` to classify passengers

### 9. Evaluating Performance
- Used `accuracy_score` to evaluate model performance on the validation set

### 10. Visualizing Feature Importance
- Plotted feature importance with `xgb.plot_importance`
- Used the `gain` metric to determine importance

---

## How to Run the Program

### Install Dependencies
```bash
pip install xgboost scikit-learn pandas matplotlib
```

### Download the Dataset
- Download Titanic dataset from Kaggle
- Save it as `train.csv` in the working directory

### Run the Script
```bash
python titanic_xgboost.py
```

### View Output
- Console shows validation set accuracy
- A feature importance plot will be displayed

---

## Key Features of the Code

- **Custom Objective Function:** Defined a logistic loss function manually
- **Missing Value Handling:** Mean for numeric, mode for categorical
- **Feature Importance:** Visualized with `plot_importance`
- **Early Stopping:** Prevented overfitting via validation loss tracking

---

## Example Output
```text
Accuracy: 0.82
```

### Feature Importance Plot
> *(Example placeholder: Shows top features like Sex, Fare, and Age)*

---

## Conclusion

This project demonstrates how to implement a custom objective function in XGBoost while handling real-world data challenges like missing values and categorical variables. You can:

- Learn preprocessing for XGBoost
- Understand the mechanics of gradient boosting
- Interpret model predictions with visualizations

Feel free to experiment with other hyperparameters or enhancements. If you have questions, feel free to reach out. Happy coding! 😊

