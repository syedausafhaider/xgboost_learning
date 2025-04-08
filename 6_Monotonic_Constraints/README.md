# Save the entire markdown content into a single .md file and provide it for download

readme_md_path = "/mnt/data/README.md"

with open(readme_md_path, "w") as f:
    f.write("""# XGBoost Titanic Survival Prediction with Monotonic Constraints

This repository demonstrates how I enforced **monotonic constraints** in XGBoost to align the model with domain knowledge. For example, I ensured that higher Fare values have a positive relationship with survival probability. Below, I explain each step in detail.

---

## 🧠 Overview

I used the Titanic dataset to predict survival while incorporating domain knowledge into the model. Monotonic constraints ensure that the model’s predictions follow logical assumptions (e.g., wealthier passengers likely had better survival chances). This improves interpretability and aligns the model with real-world reasoning.

---

## 🛠️ Step-by-Step Explanation

### 1. Loading the Dataset
- Loaded the Titanic dataset from a CSV file using `pandas.read_csv`.
- The dataset includes features like `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, and `Embarked`.

### 2. Handling Missing Values
- Filled missing values in numeric columns (`Age`, `Fare`) with the mean.
- Filled missing values in the categorical column (`Embarked`) with the most frequent value (`'S'`).

### 3. Encoding Categorical Variables
- Encoded `Sex` and `Embarked` into numeric values using `LabelEncoder`.
- Ensures compatibility with XGBoost, which requires numeric input.

### 4. Splitting the Dataset
- Split the data into training (80%) and validation (20%) sets to evaluate generalization.

### 5. Converting Data to DMatrix
- Converted the datasets into XGBoost’s optimized `DMatrix` format for faster computations.

### 6. Defining Monotonic Constraints
- Enforced a **positive monotonic constraint** on the `Fare` feature (index 5 in the DMatrix).
- Constraint tuple: `(0, 0, 0, 0, 0, 1, 0)` — corresponds to:
