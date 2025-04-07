# XGBoost Titanic Survival Prediction with Learning Rate Decay

This project demonstrates how I applied **learning rate decay** in an XGBoost model to predict survival on the Titanic dataset. The learning rate decreases **exponentially** over boosting rounds, helping the model make faster progress early and finer adjustments later. This improves training stability and convergence.

---

## Overview

Using the Titanic dataset, I built an XGBoost classifier with:

- Missing value handling
- Categorical variable encoding
- A custom learning rate decay schedule
- Evaluation on a validation set
- A plot showing how the learning rate changes over time

---

## Step-by-Step Breakdown

### 1. Loading the Dataset
- Used `pandas.read_csv` to load `train.csv`.
- Features used: `Age`, `Sex`, `Fare`, and `Embarked`.
- Target variable: `Survived`.

### 2. Handling Missing Values
- Replaced missing `Age` and `Fare` values with their mean.
- Replaced missing `Embarked` values with the most frequent value ("S").

### 3. Encoding Categorical Variables
- Used `LabelEncoder` to convert `Sex` and `Embarked` into numeric format.
- XGBoost requires all input data to be numeric.

### 4. Splitting the Dataset
- Used `train_test_split` to split the data into:
  - 80% training set
  - 20% validation set

### 5. Converting to DMatrix
- Converted training and validation sets into XGBoost’s DMatrix format for optimized performance.

### 6. Learning Rate Decay

Implemented an exponential decay schedule:

```python
def lr_decay(iteration, base_lr=0.1, decay_rate=0.99):
    return base_lr * (decay_rate ** iteration)
```

- Starts with a learning rate of `0.1`
- Reduces by `1%` per round using `xgb.callback.LearningRateScheduler`

### 7. Training the Model

Key parameters:
- `max_depth=6`: Controls tree complexity
- `eta=0.1`: Initial learning rate (overridden by decay schedule)
- `early_stopping_rounds=10`: Stops training when validation score plateaus

### 8. Making Predictions

- Generated predictions on the validation set
- Applied sigmoid threshold of 0.5 to convert probabilities to binary labels

### 9. Evaluating Performance

Used `accuracy_score` from scikit-learn to compute validation accuracy.

**Example output:**
```
Accuracy: 0.82
```

### 10. Visualizing Learning Rate Decay

Plotted the learning rate across 100 boosting rounds using matplotlib:

```python
plt.plot([lr_decay(i) for i in range(100)])
plt.title("Learning Rate Decay")
plt.xlabel("Boosting Round")
plt.ylabel("Learning Rate")
plt.grid()
```

---

## How to Run the Program

### Install Dependencies
```bash
pip install xgboost scikit-learn pandas matplotlib
```

### Download Dataset
- Get the Titanic dataset from [Kaggle](https://www.kaggle.com/c/titanic) and save it as `train.csv`.

### Run the Script
```bash
python titanic_xgboost.py
```

### View Output
- Console prints accuracy
- Plot displays the learning rate decay over time

---

## Key Features

- **Learning Rate Decay**: Smoothly reduces learning rate per iteration
- **Early Stopping**: Prevents overfitting when validation loss plateaus
- **DMatrix Format**: Efficient handling of input data for XGBoost
- **Exponential Decay Formula**: `learning_rate = 0.1 * 0.99^iteration`

---

## Example Output

**Console Output**
```
Accuracy: 0.82
```

**Learning Rate Decay Plot**
_A plot that shows learning rate decreasing from 0.1 to lower values over 100 rounds._

---

## Conclusion

This project demonstrates how learning rate decay can improve XGBoost model convergence. By starting with a high learning rate and gradually lowering it:

- The model learns faster early on
- It fine-tunes more carefully later

This technique works well for both small and large datasets. Feel free to experiment with different `decay_rate` or `initial_lr` values to further improve performance.

---

Got questions or want to optimize it further? Let’s connect. Happy coding! 😊

