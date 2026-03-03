```markdown
# House Price Prediction with Linear Regression

A beginner machine learning project that walks through predicting house prices
using a simple linear regression model. Built as a learning exercise to understand
the full ML pipeline from raw data to model evaluation.

Dataset: [Housing Prices Dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset/data?select=Housing.csv) from Kaggle (545 rows, 13 features)

---

## What this notebook covers

Starting from raw data and ending with a trained model, the notebook goes through:

- Exploratory data analysis and statistical summaries
- Correlation analysis and feature relationships
- Outlier detection using IQR (Tukey fences)
- Log transformation to handle skewed price distribution
- Preprocessing pipeline with RobustScaler and OneHotEncoder
- Linear Regression training and evaluation
- Cross validation with custom scorers for real price scale metrics
- Residual analysis and assumption checks
- Baseline comparison against a naive median predictor

---

## Results

| Metric | Score |
|---|---|
| R-squared | 0.69 |
| RMSE | $1,121,659 |
| MAE | $813,290 |
| Naive baseline RMSE | ~$1,870,440 |

The model outperforms the naive median baseline by roughly 38% on MAE,
explaining about 69% of house price variance from 12 input features.

---

## Stack

```
Python 3.12
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
```

---

## How to run it

**Locally**

1. Clone the repo
2. Place `Housing.csv` inside a `Datasets/` folder one level above the notebook
3. Install dependencies
4. Run all cells top to bottom

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

**On Kaggle**

Upload the notebook and add the Housing Prices Dataset from Kaggle directly.
The notebook includes a path detection cell that handles both environments automatically.

---

## Project structure

```
house-price/
│
├── notebooks/
│   └── house-price.ipynb       # main notebook
│
├── Datasets/
│   └── Housing.csv             # dataset (not included, download from Kaggle)
│
└── README.md
```

---

## Key decisions and why

**Log transform on price**
The price distribution is right-skewed. Applying `np.log1p()` before training
brings it closer to normal, reduces outlier influence, and improved R² by about
0.025 compared to training on raw prices.

**RobustScaler over StandardScaler**
The dataset has luxury house outliers that StandardScaler is sensitive to.
RobustScaler uses median and IQR internally, so those extreme values have
minimal effect on how the other houses get scaled.

**OneHotEncoder with drop='first'**
Categorical features like furnishing status have no natural numeric order,
so label encoding would introduce false rank relationships. OneHotEncoder
treats each category independently. The `drop='first'` argument removes
one column per category to avoid the dummy variable trap.

**Kept outliers rather than removing them**
The 15 houses above Tukey's upper fence are consistent internally (large area,
multiple bathrooms, good amenities). Removing real data points to clean up
metrics is a form of bias, so they were kept and handled through the log transform instead.

---

## Limitations

The model leaves about 31% of price variance unexplained. Most of that gap
comes down to missing features rather than model choice. Location, house age,
condition, and neighbourhood quality are all known price drivers that are not
in this dataset. No algorithm change will recover information that was never collected.

---

## What to try next

- Ridge or Lasso regression to handle mild multicollinearity between features
- Random Forest to capture non-linear relationships
- Gradient Boosting (XGBoost or LightGBM) for better handling of the expensive
  house segment where the linear model consistently under-predicts
- A richer dataset with location and condition data

---

## References

- Dataset: Yasser H, Kaggle Housing Prices Dataset
- Coefficient visualization approach: [DataCamp sklearn linear regression tutorial](https://www.datacamp.com/tutorial/sklearn-linear-regression)
- Encoding strategies: [GeeksforGeeks ML Label Encoding](https://www.geeksforgeeks.org/machine-learning/ml-label-encoding-of-datasets-in-python/)
```
