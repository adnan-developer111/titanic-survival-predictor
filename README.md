# Titanic Survival Prediction

A complete machine learning project that predicts passenger survival on the Titanic 
using classification algorithms. Built as part of my AI Engineering learning path.

## Project Overview

The goal of this project is to predict whether a passenger survived the Titanic 
disaster based on features such as age, sex, passenger class, and fare. 
This project covers the full ML workflow: data exploration, cleaning, 
feature engineering, model comparison, and evaluation.

## Dataset

- **Source**: [Titanic Dataset](https://github.com/datasciencedojo/datasets/blob/master/titanic.csv)
- **Size**: 891 passengers, 12 features
- **Target variable**: `Survived` (0 = Did not survive, 1 = Survived)

## Key Findings from Exploration

- Overall survival rate: ~38%
- Female survival rate: ~74% vs Male survival rate: ~19%
- Strong correlation between passenger class and survival

## Project Structure
titanic-survival-predictor/

├── titanic.ipynb       # Main notebook: full workflow

├── titanic.csv         # Dataset

├── requirements.txt    # Dependencies

└── README.md           # Project documentation

## Data Preprocessing

- **Missing values**: 
  - Age (177 missing) → filled with median age
  - Cabin (687 missing, 77%) → dropped entirely
  - Embarked (2 missing) → filled with mode
- **Encoding**: 
  - Sex → binary encoded (male=0, female=1)
  - Embarked → one-hot encoded

## Models Compared

| Model | Accuracy |
|-------|----------|
| Logistic Regression | 81% |
| Decision Tree | 79% |
| Random Forest | 79% |
| **XGBoost** | **82%** |

XGBoost achieved the highest accuracy and was selected as the final model.

## Why XGBoost?

XGBoost (Extreme Gradient Boosting) outperformed other models on this dataset 
because it builds an ensemble of decision trees sequentially, where each tree 
corrects the errors of the previous one. This makes it particularly strong on 
structured/tabular data with mixed feature types like this dataset.

## Results

- **Final Model**: XGBoost Classifier
- **Test Accuracy**: 82%
- The model correctly identifies key survival patterns: 
  gender and passenger class are the strongest predictors.

## Custom Prediction Example

The model can predict survival for any passenger:

```python
new_person = pd.DataFrame({
    "Pclass": [1],
    "Sex": [1],      # 1 = female
    "Age": [29],
    "SibSp": [0],
    "Parch": [0],
    "Fare": [211.0]
})
# Output: Survived — Probability: 91%
```

## Installation & Usage

```bash
git clone https://github.com/YOUR_USERNAME/titanic-survival-predictor.git
cd titanic-survival-predictor
pip install -r requirements.txt
jupyter notebook titanic.ipynb
```

## Skills Demonstrated

- Data exploration and visualization (Pandas, Matplotlib)
- Data cleaning and feature engineering
- Multiple ML model training and comparison
- Model evaluation (accuracy, prediction probability)
- Python, scikit-learn, XGBoost, Jupyter Notebook

## Author

Adnan Zahedi — 15-year-old aspiring AI Engineer based in Budapest, Hungary.
Building toward a career in Machine learning and AI systems.