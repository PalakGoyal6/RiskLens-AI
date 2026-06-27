# Home Credit Default Risk Dataset

This folder should contain the raw CSV files for the project. Due to size limitations, the dataset is not committed to this repository.

## Download Instructions

1. **Source:** Download the dataset from the Kaggle competition:
   [Home Credit Default Risk Dataset](https://www.kaggle.com/competitions/home-credit-default-risk/data)
   
2. **Files required:**
   Place the following CSV files directly inside this `data/` folder before running the notebooks:
   - `application_train.csv` (main customer application training data)
   - `application_test.csv` (main customer application test data)
   - `bureau.csv` (history of customer credit reported by other financial institutions)
   - `bureau_balance.csv` (monthly balance history of customer credit in bureau)
   - `previous_application.csv` (history of customer applications for home credit)
   - `installments_payments.csv` (repayment history for home credit loans)
   - `POS_CASH_balance.csv` (monthly balance history of point-of-sale and cash loans)
   - `credit_card_balance.csv` (monthly balance history of credit cards)
   - `HomeCredit_columns_description.csv` (descriptions of dataset columns)

3. **Output Subdirectory:**
   The `data/processed/` folder will be automatically created to hold intermediate compiled Parquet files (e.g. `model_dataset.parquet`) when you run the pipeline.
