import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.resolve()))
from nb_common import md, code, save, SETUP_SNIPPET

cells = []

cells.append(md('''# Credit Risk Intelligence Platform

## Business Problem

- Financial institutions must decide whether an applicant is likely to repay a loan.  
  A wrong approval can lead to financial losses.

- Goal:  
  Predict whether a customer will experience payment difficulties.

- Target Variable:  
  TARGET = 0 → No payment difficulties  
  TARGET = 1 → Payment difficulties / default  

## Project Objectives

1. Understand customer characteristics
2. Identify important risk factors
3. Build a predictive model
4. Explain predictions using SHAP

> **Local version notes:** this notebook originally ran on Databricks using PySpark
> (`spark.read.csv`, `display()`, etc.) and read files from a Databricks Volume.
> It has been rewritten to use **pandas** and read CSVs from a local folder, so it
> runs in a normal local Jupyter environment. The logic/analysis is unchanged.'''))

cells.append(md("## Setup"))
cells.append(code(SETUP_SNIPPET))

cells.append(md("## Loading Data"))
cells.append(code('''
import pandas as pd
import numpy as np
import gc

# To prevent MemoryError, we only load application_train.csv fully.
# The other files are checked memory-efficiently.
app = pd.read_csv(DATA_DIR / "application_train.csv", engine='pyarrow').sample(frac=0.3, random_state=42)
'''))

cells.append(md('''# Dataset Overview

Before building models we must understand:

- Number of observations
- Number of features
- Data types
- Missing values
- Class imbalance

This helps determine the preprocessing strategy.'''))

cells.append(code('''
# Get shapes memory-efficiently to avoid OOM
overview_data = []
for name, filename in [
    ("Application", "application_train.csv"),
    ("Bureau", "bureau.csv"),
    ("Previous", "previous_application.csv"),
    ("Installments", "installments_payments.csv")
]:
    # Read first row to get columns
    cols_df = pd.read_csv(DATA_DIR / filename, nrows=1)
    cols = len(cols_df.columns)
    # Read one column to get rows
    rows_df = pd.read_csv(DATA_DIR / filename, usecols=[cols_df.columns[0]])
    rows = len(rows_df)
    overview_data.append((name, rows, cols))
    del cols_df, rows_df
    gc.collect()

for name, rows, cols in overview_data:
    print("\\n", name)
    print("Rows:", rows)
    print("Columns:", cols)
'''))

cells.append(md('''## Target Distribution

### Class Imbalance Analysis

Many credit-risk datasets are imbalanced.  
Most customers repay loans.  
If default cases are rare, accuracy becomes misleading.  
We therefore inspect the TARGET distribution.'''))

cells.append(code('''
target_dist = app["TARGET"].value_counts().rename_axis("TARGET").reset_index(name="count")
total = len(app)
target_dist["percentage"] = (target_dist["count"] * 100 / total).round(2)

display(target_dist)
'''))

cells.append(md('''### Observations
- Approximately 92% of customers do not default.
- Only about 8% default/face payment difficulties.
- This indicates severe class imbalance.
- Metrics such as ROC-AUC, Recall and F1 Score will be more useful than Accuracy.'''))

cells.append(md('''## Missing Values
Missing values can reduce model performance.

Identifying:
- Features with low missingness
- Features with moderate missingness
- Features with excessive missingness

will guide imputation and feature selection.'''))

cells.append(code('''
rows = len(app)

null_counts = app.isnull().sum()
missing_df = pd.DataFrame({
    "column": null_counts.index,
    "null_count": null_counts.values,
})
missing_df["null_pct"] = (missing_df["null_count"] * 100 / rows).round(2)

display(missing_df.sort_values("null_pct", ascending=False))
'''))

cells.append(md('''## Numerical Feature Analysis

Income, Credit Amount and Annuity are expected to be strong predictors.

We examine their distributions and summary statistics.'''))

cells.append(code('''
display(app[["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY"]].describe())
'''))

cells.append(md('''### Employment Duration Anomaly

The value 365243 appears frequently in DAYS_EMPLOYED.

This is not a real employment duration.

According to dataset documentation, it represents missing information.

We verify its frequency.'''))

cells.append(code('''
(app["DAYS_EMPLOYED"] == 365243).sum()
'''))

cells.append(code('''
app.groupby(app["DAYS_EMPLOYED"] == 365243).size()
'''))

cells.append(md('''## Correlation with Default
### External Risk Scores

The dataset contains external credit risk indicators.  
These are expected to be among the strongest predictors.'''))

cells.append(code('''
# Databricks' display() on a raw (non-aggregated) dataframe shows a sample of rows.
# We do the same locally with .head().
display(app[["TARGET", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].head(20))
'''))

cells.append(md('''## Univariate Analysis

Univariate analysis studies one variable at a time.

Goals:
- Understand distribution
- Detect skewness
- Detect outliers
- Understand central tendency

Metrics:
- Mean
- Median
- Mode
- Variance
- Standard Deviation'''))

cells.append(code('''
numeric_cols = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "DAYS_BIRTH"
]
'''))

cells.append(code('''
# mean
print(app[numeric_cols].mean())
'''))

cells.append(code('''
# median
medians = app[numeric_cols].median()
print(medians)
'''))

cells.append(code('''
# variance
print(app[numeric_cols].var())
'''))

cells.append(code('''
# standard deviation
print(app[numeric_cols].std())
'''))

cells.append(code('''
# mode
display(
    app["CODE_GENDER"].value_counts().rename_axis("CODE_GENDER").reset_index(name="count")
)
'''))

cells.append(code('''
pdf = app[["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY"]].sample(frac=0.1, random_state=42)

import matplotlib.pyplot as plt
%matplotlib inline

pdf["AMT_INCOME_TOTAL"].hist()
plt.title("Income Distribution")
plt.show()
'''))

cells.append(md('''Income distribution is heavily right-skewed.

Most applicants earn moderate incomes, while a small number earn exceptionally high incomes.

This suggests that a log transformation may be useful.'''))

cells.append(md("## Bivariate analysis"))

cells.append(code('''
# target vs income
display(app.groupby("TARGET")["AMT_INCOME_TOTAL"].mean())
'''))

cells.append(code('''
# target vs credit amount
display(app.groupby("TARGET")["AMT_CREDIT"].mean())
'''))

cells.append(code('''
# target vs gender
display(app.groupby(["CODE_GENDER", "TARGET"]).size().rename("count").reset_index())
'''))

cells.append(md("Correlation Analysis"))

cells.append(code('''
for c in [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3"
]:
    print(c, app[c].corr(app["TARGET"]))
'''))

cells.append(md('''EXT_SOURCE_2 & EXT_SOURCE_3 show a strong negative correlation with TARGET.  
Higher external credit scores are associated with lower default risk.'''))

cells.append(code('''
sample = app[["AMT_INCOME_TOTAL", "AMT_CREDIT", "TARGET"]].sample(frac=0.05, random_state=42)
'''))

cells.append(code('''
import matplotlib.pyplot as plt

plt.scatter(
    sample["AMT_INCOME_TOTAL"],
    sample["AMT_CREDIT"],
    alpha=0.3
)

plt.xlabel("Income")
plt.ylabel("Credit")

plt.show()
'''))

cells.append(md("## Multivariate Analysis"))

cells.append(code('''
cols = [
    "TARGET",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "EXT_SOURCE_1",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED"
]
'''))

cells.append(code('''
corr_df = app[cols].sample(frac=0.3, random_state=42)

corr_df.corr()
'''))

cells.append(code('''
import seaborn as sns

sns.heatmap(
    corr_df.corr(),
    annot=True
)
'''))

cells.append(md('''## Correlation Analysis Findings

1. EXT_SOURCE_1, EXT_SOURCE_2 and EXT_SOURCE_3 exhibit the strongest
relationship with TARGET and are expected to be important predictors.

2. AMT_CREDIT and AMT_ANNUITY show strong positive correlation (0.77),
indicating larger loans generally require larger installment payments.

3. DAYS_BIRTH and DAYS_EMPLOYED show strong association, suggesting
older applicants typically possess longer employment histories.

4. Most individual features show relatively weak correlation with TARGET,
indicating that default prediction is likely driven by complex
interactions among multiple variables.

5. Feature engineering using ratios such as Credit-to-Income and
Annuity-to-Income may improve predictive power.'''))

cells.append(code('''
# Grouping Analysis
display(
    app.groupby("NAME_INCOME_TYPE").agg(
        **{"avg(AMT_INCOME_TOTAL)": ("AMT_INCOME_TOTAL", "mean")},
        **{"avg(TARGET)": ("TARGET", "mean")},
    )
)
'''))

cells.append(code('''
# pivot table
pivot = pd.crosstab(app["NAME_EDUCATION_TYPE"], app["TARGET"])

display(pivot)
'''))

cells.append(md('''Applicants with higher education levels appear to have lower default rates.

Education may therefore be an important predictive feature.'''))

cells.append(md('''EDA Conclusions

1. Dataset is highly imbalanced (~8% defaults).
2. Income and credit variables are strongly right-skewed.
3. DAYS_EMPLOYED contains anomaly values (365243).
4. External credit scores are among the strongest predictors.
5. Several variables contain substantial missingness.
6. Education level and income type show differences in default rates.
7. Feature engineering using credit-to-income ratios is likely to improve predictive power.  

These findings guide the cleaning and feature engineering stage.'''))

save(cells, "01_BusinessUnderstanding_EDA.ipynb", "01")
