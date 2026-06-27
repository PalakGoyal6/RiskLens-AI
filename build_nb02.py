import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.resolve()))
from nb_common import md, code, save, SETUP_SNIPPET

cells = []

cells.append(md('''# Data Cleaning, Feature Engineering & Feature Selection

This notebook performs data cleaning, advanced feature engineering, and feature selection for the Home Credit Default Risk dataset.

## Workflow:
1. **Data Cleaning:** Handle anomalies (e.g., `DAYS_EMPLOYED` anomaly) and impute missing values.
2. **Advanced Aggregations:** Summarize historical datasets (`bureau_balance`, `bureau`, `previous_application`, `installments_payments`, `POS_CASH_balance`, `credit_card_balance`) to the customer level.
3. **Feature Engineering:** Build financial ratios and interactions (e.g., external credit scores mean/multiplications, credit-to-income ratios).
4. **Feature Selection:** Apply 4 feature selection methods (Correlation, XGBoost feature importance, Recursive Feature Elimination, Lasso regularization) and keep features selected by at least 2 methods.
5. **Dimensionality Reduction:** Analyze PCA and LDA projection (though we preserve original features for explainability).
6. **Persistence:** Save the final dataset as `model_dataset.parquet` and the feature set as `final_features.json`.

> **Local version notes:** Combined from the original Notebook 02 (basic cleaning) and Notebook 04 (advanced aggregations). Runs entirely in pandas to process raw CSV datasets memory-efficiently using chunks.'''))

cells.append(md("## Setup"))
cells.append(code(SETUP_SNIPPET))

cells.append(md("## 1. Load application_train and Perform Basic Cleaning"))
cells.append(code('''
import pandas as pd
import numpy as np
import gc

app = pd.read_csv(DATA_DIR / "application_train.csv", low_memory=False)
print(f"Loaded application_train: {app.shape}")

# 1. Fix DAYS_EMPLOYED anomaly
app["DAYS_EMPLOYED_ANOM"] = (app["DAYS_EMPLOYED"] == 365243).astype(int)
app["DAYS_EMPLOYED"] = app["DAYS_EMPLOYED"].where(app["DAYS_EMPLOYED"] != 365243, np.nan)

# 2. Impute main financial numeric columns with median
numeric_cols = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "DAYS_EMPLOYED"]
for c in numeric_cols:
    median = app[c].median()
    app[c] = app[c].fillna(median)
'''))

cells.append(md("## 2. Advanced Feature Aggregations"))

# Chunks reading helper function
cells.append(code('''
def read_csv_in_chunks(file_path, usecols, chunksize=300000):
    header = pd.read_csv(file_path, nrows=0).columns.tolist()
    cols_to_load = [c for c in usecols if c in header]
    return pd.read_csv(file_path, usecols=cols_to_load, chunksize=chunksize, low_memory=False)
'''))

cells.append(md("### POS_CASH Balance Aggregations"))
cells.append(code('''
pos_agg_chunks = []
contract_chunks = []

for chunk in read_csv_in_chunks(DATA_DIR / "POS_CASH_balance.csv", usecols=["SK_ID_CURR", "SK_ID_PREV", "SK_DPD", "SK_DPD_DEF", "NAME_CONTRACT_STATUS", "MONTHS_BALANCE"]):
    chunk["_dpd_flag"] = (chunk["SK_DPD"] > 0).astype(int)
    chunk["XY_dpd"] = chunk["MONTHS_BALANCE"] * chunk["SK_DPD"]
    chunk["X2_months"] = chunk["MONTHS_BALANCE"] ** 2
    
    pos_agg_chunk = chunk.groupby("SK_ID_CURR").agg(
        POS_COUNT=("SK_ID_CURR", "size"),
        POS_SUM_DPD=("SK_DPD", "sum"),
        POS_MAX_DPD=("SK_DPD", "max"),
        POS_SUM_DPD_DEF=("SK_DPD_DEF", "sum"),
        POS_MAX_DPD_DEF=("SK_DPD_DEF", "max"),
        POS_SUM_DPD_FLAG=("_dpd_flag", "sum"),
        sum_X=("MONTHS_BALANCE", "sum"),
        sum_Y=("SK_DPD", "sum"),
        sum_XY=("XY_dpd", "sum"),
        sum_X2=("X2_months", "sum")
    ).reset_index()
    
    pos_recent_chunk = (
        chunk[chunk["MONTHS_BALANCE"] >= -6]
        .groupby("SK_ID_CURR")["SK_DPD"]
        .agg(sum_recent_dpd="sum", count_recent_dpd="count")
        .reset_index()
    )
    pos_agg_chunk = pos_agg_chunk.merge(pos_recent_chunk, on="SK_ID_CURR", how="left")
    pos_agg_chunks.append(pos_agg_chunk)
    
    chunk["is_completed"] = (chunk["NAME_CONTRACT_STATUS"] == "Completed").astype(int)
    contract_chunk = chunk.groupby("SK_ID_PREV").agg(
        SK_ID_CURR=("SK_ID_CURR", "first"),
        max_months=("MONTHS_BALANCE", "max"),
        is_completed=("is_completed", "max")
    ).reset_index()
    contract_chunks.append(contract_chunk)
    
    del chunk, pos_agg_chunk, pos_recent_chunk, contract_chunk
    gc.collect()

pos_agg = pd.concat(pos_agg_chunks, ignore_index=True)
del pos_agg_chunks
gc.collect()

pos_agg = pos_agg.groupby("SK_ID_CURR").agg(
    POS_COUNT=("POS_COUNT", "sum"),
    POS_SUM_DPD=("POS_SUM_DPD", "sum"),
    POS_MAX_DPD=("POS_MAX_DPD", "max"),
    POS_SUM_DPD_DEF=("POS_SUM_DPD_DEF", "sum"),
    POS_MAX_DPD_DEF=("POS_MAX_DPD_DEF", "max"),
    POS_SUM_DPD_FLAG=("POS_SUM_DPD_FLAG", "sum"),
    sum_recent_dpd=("sum_recent_dpd", "sum"),
    count_recent_dpd=("count_recent_dpd", "sum"),
    sum_X=("sum_X", "sum"),
    sum_Y=("sum_Y", "sum"),
    sum_XY=("sum_XY", "sum"),
    sum_X2=("sum_X2", "sum")
).reset_index()

pos_agg["POS_AVG_DPD"] = pos_agg["POS_SUM_DPD"] / (pos_agg["POS_COUNT"] + 1e-9)
pos_agg["POS_AVG_DPD_DEF"] = pos_agg["POS_SUM_DPD_DEF"] / (pos_agg["POS_COUNT"] + 1e-9)
pos_agg["POS_DPD_RATIO"] = pos_agg["POS_SUM_DPD_FLAG"] / (pos_agg["POS_COUNT"] + 1e-9)
pos_agg["POS_RECENT_AVG_DPD"] = pos_agg["sum_recent_dpd"] / (pos_agg["count_recent_dpd"] + 1e-9)
pos_agg["POS_DPD_TREND"] = pos_agg["POS_RECENT_AVG_DPD"] - pos_agg["POS_AVG_DPD"]

pos_agg["mean_X"] = pos_agg["sum_X"] / pos_agg["POS_COUNT"]
pos_agg["mean_Y"] = pos_agg["sum_Y"] / pos_agg["POS_COUNT"]
pos_agg["mean_XY"] = pos_agg["sum_XY"] / pos_agg["POS_COUNT"]
pos_agg["mean_X2"] = pos_agg["sum_X2"] / pos_agg["POS_COUNT"]
pos_agg["POS_DPD_SLOPE"] = (pos_agg["mean_XY"] - pos_agg["mean_X"] * pos_agg["mean_Y"]) / (pos_agg["mean_X2"] - pos_agg["mean_X"]**2 + 1e-9)
pos_agg["POS_DPD_SLOPE"] = np.where(pos_agg["POS_COUNT"] < 2, 0, pos_agg["POS_DPD_SLOPE"])

pos_agg = pos_agg.drop(columns=["POS_SUM_DPD", "POS_SUM_DPD_DEF", "POS_SUM_DPD_FLAG", "sum_recent_dpd", "count_recent_dpd", "sum_X", "sum_Y", "sum_XY", "sum_X2", "mean_X", "mean_Y", "mean_XY", "mean_X2"])

contract_combined = pd.concat(contract_chunks, ignore_index=True)
del contract_chunks
gc.collect()

contract_combined = contract_combined.groupby("SK_ID_PREV").agg(
    SK_ID_CURR=("SK_ID_CURR", "first"),
    max_months=("max_months", "max"),
    is_completed=("is_completed", "max")
).reset_index()

contract_combined["is_recent"] = (contract_combined["max_months"] >= -12).astype(int)
contract_combined["is_older"] = (contract_combined["max_months"] < -12).astype(int)
contract_combined["completed_recent"] = contract_combined["is_completed"] * contract_combined["is_recent"]
contract_combined["completed_older"] = contract_combined["is_completed"] * contract_combined["is_older"]

pos_comp_df = contract_combined.groupby("SK_ID_CURR").agg(
    RECENT_COMPLETED=("completed_recent", "sum"),
    RECENT_TOTAL=("is_recent", "sum"),
    OLDER_COMPLETED=("completed_older", "sum"),
    OLDER_TOTAL=("is_older", "sum")
).reset_index()
pos_comp_df["POS_RECENT_COMPLETION_RATE"] = pos_comp_df["RECENT_COMPLETED"] / (pos_comp_df["RECENT_TOTAL"] + 1e-9)
pos_comp_df["POS_OLDER_COMPLETION_RATE"] = pos_comp_df["OLDER_COMPLETED"] / (pos_comp_df["OLDER_TOTAL"] + 1e-9)
pos_comp_df["POS_COMPLETION_RATE_DIFF"] = pos_comp_df["POS_RECENT_COMPLETION_RATE"] - pos_comp_df["POS_OLDER_COMPLETION_RATE"]

pos_agg = pos_agg.merge(pos_comp_df[["SK_ID_CURR", "POS_COMPLETION_RATE_DIFF"]], on="SK_ID_CURR", how="left")

del contract_combined, pos_comp_df
gc.collect()
print(f"POS aggregations complete: {pos_agg.shape}")
'''))

cells.append(md("### Credit Card Balance Aggregations"))
cells.append(code('''
cc_agg_chunks = []

for chunk in read_csv_in_chunks(DATA_DIR / "credit_card_balance.csv", usecols=["SK_ID_CURR", "AMT_CREDIT_LIMIT_ACTUAL", "AMT_BALANCE", "SK_DPD", "AMT_DRAWINGS_CURRENT", "AMT_INST_MIN_REGULARITY", "AMT_PAYMENT_CURRENT", "MONTHS_BALANCE"]):
    chunk["CC_UTILIZATION"] = np.where(chunk["AMT_CREDIT_LIMIT_ACTUAL"] > 0, chunk["AMT_BALANCE"] / chunk["AMT_CREDIT_LIMIT_ACTUAL"], np.nan)
    chunk["_over"] = (chunk["AMT_BALANCE"] > chunk["AMT_CREDIT_LIMIT_ACTUAL"]).astype(int)
    chunk["_ratio"] = np.where(chunk["AMT_INST_MIN_REGULARITY"] > 0, chunk["AMT_PAYMENT_CURRENT"] / chunk["AMT_INST_MIN_REGULARITY"], np.nan)
    
    cc_agg_chunk = chunk.groupby("SK_ID_CURR").agg(
        CC_COUNT=("SK_ID_CURR", "size"),
        CC_SUM_UTILIZATION=("CC_UTILIZATION", "sum"),
        CC_COUNT_UTILIZATION=("CC_UTILIZATION", "count"),
        CC_MAX_UTILIZATION=("CC_UTILIZATION", "max"),
        CC_SUM_DPD=("SK_DPD", "sum"),
        CC_MAX_DPD=("SK_DPD", "max"),
        CC_SUM_DRAWINGS=("AMT_DRAWINGS_CURRENT", "sum"),
        CC_SUM_OVER_LIMIT=("_over", "sum"),
        CC_SUM_PAY_RATIO=("_ratio", "sum"),
        CC_COUNT_PAY_RATIO=("_ratio", "count"),
        CC_DRAWINGS_HIST_SUM=("AMT_DRAWINGS_CURRENT", "sum"),
        CC_DRAWINGS_HIST_COUNT=("AMT_DRAWINGS_CURRENT", "count")
    ).reset_index()
    
    cc_recent_chunk = chunk[chunk["MONTHS_BALANCE"] >= -6].copy()
    if not cc_recent_chunk.empty:
        cc_recent_chunk["XY_util"] = cc_recent_chunk["MONTHS_BALANCE"] * cc_recent_chunk["CC_UTILIZATION"]
        cc_recent_chunk["X2_months"] = cc_recent_chunk["MONTHS_BALANCE"] ** 2
        
        cc_recent_agg_chunk = cc_recent_chunk.groupby("SK_ID_CURR").agg(
            sum_X=("MONTHS_BALANCE", "sum"),
            sum_Y=("CC_UTILIZATION", "sum"),
            sum_XY=("XY_util", "sum"),
            sum_X2=("X2_months", "sum"),
            count_cc=("CC_UTILIZATION", "count"),
            CC_DRAWINGS_RECENT_SUM=("AMT_DRAWINGS_CURRENT", "sum"),
            CC_DRAWINGS_RECENT_COUNT=("AMT_DRAWINGS_CURRENT", "count")
        ).reset_index()
        cc_agg_chunk = cc_agg_chunk.merge(cc_recent_agg_chunk, on="SK_ID_CURR", how="left")
    else:
        for c in ["sum_X", "sum_Y", "sum_XY", "sum_X2", "count_cc", "CC_DRAWINGS_RECENT_SUM", "CC_DRAWINGS_RECENT_COUNT"]:
            cc_agg_chunk[c] = np.nan
            
    cc_agg_chunks.append(cc_agg_chunk)
    del chunk, cc_agg_chunk
    if 'cc_recent_chunk' in locals():
        del cc_recent_chunk, cc_recent_agg_chunk
    gc.collect()

cc_agg = pd.concat(cc_agg_chunks, ignore_index=True)
del cc_agg_chunks
gc.collect()

cc_agg = cc_agg.groupby("SK_ID_CURR").agg(
    CC_COUNT=("CC_COUNT", "sum"),
    CC_SUM_UTILIZATION=("CC_SUM_UTILIZATION", "sum"),
    CC_COUNT_UTILIZATION=("CC_COUNT_UTILIZATION", "sum"),
    CC_MAX_UTILIZATION=("CC_MAX_UTILIZATION", "max"),
    CC_SUM_DPD=("CC_SUM_DPD", "sum"),
    CC_MAX_DPD=("CC_MAX_DPD", "max"),
    CC_SUM_DRAWINGS=("CC_SUM_DRAWINGS", "sum"),
    CC_SUM_OVER_LIMIT=("CC_SUM_OVER_LIMIT", "sum"),
    CC_SUM_PAY_RATIO=("CC_SUM_PAY_RATIO", "sum"),
    CC_COUNT_PAY_RATIO=("CC_COUNT_PAY_RATIO", "sum"),
    CC_DRAWINGS_HIST_SUM=("CC_DRAWINGS_HIST_SUM", "sum"),
    CC_DRAWINGS_HIST_COUNT=("CC_DRAWINGS_HIST_COUNT", "sum"),
    sum_X=("sum_X", "sum"),
    sum_Y=("sum_Y", "sum"),
    sum_XY=("sum_XY", "sum"),
    sum_X2=("sum_X2", "sum"),
    count_cc=("count_cc", "sum"),
    CC_DRAWINGS_RECENT_SUM=("CC_DRAWINGS_RECENT_SUM", "sum"),
    CC_DRAWINGS_RECENT_COUNT=("CC_DRAWINGS_RECENT_COUNT", "sum")
).reset_index()

cc_agg["CC_AVG_UTILIZATION"] = cc_agg["CC_SUM_UTILIZATION"] / (cc_agg["CC_COUNT_UTILIZATION"] + 1e-9)
cc_agg["CC_AVG_DPD"] = cc_agg["CC_SUM_DPD"] / (cc_agg["CC_COUNT"] + 1e-9)
cc_agg["CC_AVG_DRAWINGS"] = cc_agg["CC_SUM_DRAWINGS"] / (cc_agg["CC_COUNT"] + 1e-9)
cc_agg["CC_OVER_LIMIT_RATIO"] = cc_agg["CC_SUM_OVER_LIMIT"] / (cc_agg["CC_COUNT"] + 1e-9)
cc_agg["CC_PAYMENT_TO_MIN_RATIO"] = cc_agg["CC_SUM_PAY_RATIO"] / (cc_agg["CC_COUNT_PAY_RATIO"] + 1e-9)

cc_agg["mean_X"] = cc_agg["sum_X"] / (cc_agg["count_cc"] + 1e-9)
cc_agg["mean_Y"] = cc_agg["sum_Y"] / (cc_agg["count_cc"] + 1e-9)
cc_agg["mean_XY"] = cc_agg["sum_XY"] / (cc_agg["count_cc"] + 1e-9)
cc_agg["mean_X2"] = cc_agg["sum_X2"] / (cc_agg["count_cc"] + 1e-9)

cc_agg["CC_UTILIZATION_TREND_6M"] = (cc_agg["mean_XY"] - cc_agg["mean_X"] * cc_agg["mean_Y"]) / (cc_agg["mean_X2"] - cc_agg["mean_X"]**2 + 1e-9)
cc_agg["CC_UTILIZATION_TREND_6M"] = np.where(cc_agg["count_cc"] < 2, 0, cc_agg["CC_UTILIZATION_TREND_6M"])

cc_agg["CC_DRAWINGS_HIST_AVG"] = cc_agg["CC_DRAWINGS_HIST_SUM"] / (cc_agg["CC_DRAWINGS_HIST_COUNT"] + 1e-9)
cc_agg["CC_DRAWINGS_RECENT_AVG"] = cc_agg["CC_DRAWINGS_RECENT_SUM"] / (cc_agg["CC_DRAWINGS_RECENT_COUNT"] + 1e-9)
cc_agg["CC_DRAWINGS_DIFF_6M"] = cc_agg["CC_DRAWINGS_RECENT_AVG"] - cc_agg["CC_DRAWINGS_HIST_AVG"]

cc_agg = cc_agg.drop(columns=["CC_SUM_UTILIZATION", "CC_COUNT_UTILIZATION", "CC_SUM_DPD", "CC_SUM_DRAWINGS", "CC_SUM_OVER_LIMIT", "CC_SUM_PAY_RATIO", "CC_COUNT_PAY_RATIO", "CC_DRAWINGS_HIST_SUM", "CC_DRAWINGS_HIST_COUNT", "sum_X", "sum_Y", "sum_XY", "sum_X2", "count_cc", "CC_DRAWINGS_RECENT_SUM", "CC_DRAWINGS_RECENT_COUNT", "mean_X", "mean_Y", "mean_XY", "mean_X2", "CC_DRAWINGS_HIST_AVG", "CC_DRAWINGS_RECENT_AVG"])
gc.collect()
print(f"Credit Card aggregations complete: {cc_agg.shape}")
'''))

cells.append(md("### Bureau & Bureau Balance Aggregations"))
cells.append(code('''
bb_agg_chunks = []
for chunk in read_csv_in_chunks(DATA_DIR / "bureau_balance.csv", usecols=["SK_ID_BUREAU", "STATUS", "MONTHS_BALANCE"]):
    chunk["BB_DPD_FLAG"] = chunk["STATUS"].isin(["1", "2", "3", "4", "5"]).astype(int)
    
    bb_agg_chunk = chunk.groupby("SK_ID_BUREAU").agg(
        BB_MONTHS_COUNT=("SK_ID_BUREAU", "size"),
        BB_SUM_DPD=("BB_DPD_FLAG", "sum"),
        BB_EVER_DPD=("BB_DPD_FLAG", "max"),
    ).reset_index()
    
    bb_recent_chunk = chunk[chunk["MONTHS_BALANCE"] >= -12]
    if not bb_recent_chunk.empty:
        bb_recent_agg_chunk = bb_recent_chunk.groupby("SK_ID_BUREAU").agg(
            BB_RECENT_SUM_DPD=("BB_DPD_FLAG", "sum"),
            BB_RECENT_COUNT=("BB_DPD_FLAG", "count")
        ).reset_index()
        bb_agg_chunk = bb_agg_chunk.merge(bb_recent_agg_chunk, on="SK_ID_BUREAU", how="left")
    else:
        bb_agg_chunk["BB_RECENT_SUM_DPD"] = np.nan
        bb_agg_chunk["BB_RECENT_COUNT"] = np.nan
        
    bb_agg_chunks.append(bb_agg_chunk)
    del chunk, bb_agg_chunk
    if 'bb_recent_chunk' in locals():
        del bb_recent_chunk
        if 'bb_recent_agg_chunk' in locals():
            del bb_recent_agg_chunk
    gc.collect()

bb_agg = pd.concat(bb_agg_chunks, ignore_index=True)
del bb_agg_chunks
gc.collect()

bb_agg = bb_agg.groupby("SK_ID_BUREAU").agg(
    BB_MONTHS_COUNT=("BB_MONTHS_COUNT", "sum"),
    BB_SUM_DPD=("BB_SUM_DPD", "sum"),
    BB_EVER_DPD=("BB_EVER_DPD", "max"),
    BB_RECENT_SUM_DPD=("BB_RECENT_SUM_DPD", "sum"),
    BB_RECENT_COUNT=("BB_RECENT_COUNT", "sum")
).reset_index()

bb_agg["BB_DPD_RATIO"] = bb_agg["BB_SUM_DPD"] / (bb_agg["BB_MONTHS_COUNT"] + 1e-9)
bb_agg["BB_RECENT_DPD_RATIO"] = bb_agg["BB_RECENT_SUM_DPD"] / (bb_agg["BB_RECENT_COUNT"] + 1e-9)
bb_agg = bb_agg.drop(columns=["BB_SUM_DPD", "BB_RECENT_SUM_DPD", "BB_RECENT_COUNT"])
gc.collect()

bureau_agg2_chunks = []
for chunk in read_csv_in_chunks(DATA_DIR / "bureau.csv", usecols=["SK_ID_CURR", "SK_ID_BUREAU", "AMT_CREDIT_SUM", "CREDIT_DAY_OVERDUE", "AMT_CREDIT_SUM_DEBT", "CREDIT_ACTIVE", "DAYS_CREDIT"]):
    chunk_bb = chunk.merge(bb_agg, on="SK_ID_BUREAU", how="left")
    
    chunk_bb["weight"] = np.exp(chunk_bb["DAYS_CREDIT"] / 365.0)
    chunk_bb["weighted_overdue"] = chunk_bb["CREDIT_DAY_OVERDUE"] * chunk_bb["weight"]
    chunk_bb["XY"] = chunk_bb["DAYS_CREDIT"] * chunk_bb["AMT_CREDIT_SUM_DEBT"]
    chunk_bb["X2"] = chunk_bb["DAYS_CREDIT"] ** 2
    chunk_bb["IS_ACTIVE"] = (chunk_bb["CREDIT_ACTIVE"] == "Active").astype(int)
    chunk_bb["IS_CLOSED"] = (chunk_bb["CREDIT_ACTIVE"] == "Closed").astype(int)
    
    chunk_agg = chunk_bb.groupby("SK_ID_CURR").agg(
        BUREAU_LOAN_COUNT=("SK_ID_CURR", "size"),
        BUREAU_SUM_CREDIT=("AMT_CREDIT_SUM", "sum"),
        BUREAU_COUNT_CREDIT=("AMT_CREDIT_SUM", "count"),
        BUREAU_MAX_OVERDUE=("CREDIT_DAY_OVERDUE", "max"),
        BUREAU_SUM_DEBT=("AMT_CREDIT_SUM_DEBT", "sum"),
        BUREAU_SUM_DPD_RATIO=("BB_DPD_RATIO", "sum"),
        BUREAU_COUNT_DPD_RATIO=("BB_DPD_RATIO", "count"),
        BUREAU_EVER_DPD=("BB_EVER_DPD", "max"),
        BUREAU_SUM_RECENT_DPD_RATIO=("BB_RECENT_DPD_RATIO", "sum"),
        BUREAU_COUNT_RECENT_DPD_RATIO=("BB_RECENT_DPD_RATIO", "count"),
        sum_weighted_overdue=("weighted_overdue", "sum"),
        sum_weight=("weight", "sum"),
        sum_X=("DAYS_CREDIT", "sum"),
        sum_Y=("AMT_CREDIT_SUM_DEBT", "sum"),
        sum_XY=("XY", "sum"),
        sum_X2=("X2", "sum"),
        ACTIVE_COUNT=("IS_ACTIVE", "sum"),
        CLOSED_COUNT=("IS_CLOSED", "sum")
    ).reset_index()
    
    bureau_agg2_chunks.append(chunk_agg)
    del chunk, chunk_bb, chunk_agg
    gc.collect()

bureau_agg2 = pd.concat(bureau_agg2_chunks, ignore_index=True)
del bureau_agg2_chunks
gc.collect()

bureau_agg2 = bureau_agg2.groupby("SK_ID_CURR").agg(
    BUREAU_LOAN_COUNT=("BUREAU_LOAN_COUNT", "sum"),
    BUREAU_SUM_CREDIT=("BUREAU_SUM_CREDIT", "sum"),
    BUREAU_COUNT_CREDIT=("BUREAU_COUNT_CREDIT", "sum"),
    BUREAU_MAX_OVERDUE=("BUREAU_MAX_OVERDUE", "max"),
    BUREAU_SUM_DEBT=("BUREAU_SUM_DEBT", "sum"),
    BUREAU_SUM_DPD_RATIO=("BUREAU_SUM_DPD_RATIO", "sum"),
    BUREAU_COUNT_DPD_RATIO=("BUREAU_COUNT_DPD_RATIO", "sum"),
    BUREAU_EVER_DPD=("BUREAU_EVER_DPD", "max"),
    BUREAU_SUM_RECENT_DPD_RATIO=("BUREAU_SUM_RECENT_DPD_RATIO", "sum"),
    BUREAU_COUNT_RECENT_DPD_RATIO=("BUREAU_COUNT_RECENT_DPD_RATIO", "sum"),
    sum_weighted_overdue=("sum_weighted_overdue", "sum"),
    sum_weight=("sum_weight", "sum"),
    sum_X=("sum_X", "sum"),
    sum_Y=("sum_Y", "sum"),
    sum_XY=("sum_XY", "sum"),
    sum_X2=("sum_X2", "sum"),
    ACTIVE_COUNT=("ACTIVE_COUNT", "sum"),
    CLOSED_COUNT=("CLOSED_COUNT", "sum")
).reset_index()

bureau_agg2["BUREAU_AVG_CREDIT"] = bureau_agg2["BUREAU_SUM_CREDIT"] / (bureau_agg2["BUREAU_COUNT_CREDIT"] + 1e-9)
bureau_agg2["BUREAU_AVG_DEBT"] = bureau_agg2["BUREAU_SUM_DEBT"] / (bureau_agg2["BUREAU_LOAN_COUNT"] + 1e-9)
bureau_agg2["BUREAU_AVG_DPD_RATIO"] = bureau_agg2["BUREAU_SUM_DPD_RATIO"] / (bureau_agg2["BUREAU_COUNT_DPD_RATIO"] + 1e-9)
bureau_agg2["BUREAU_RECENT_DPD_RATIO"] = bureau_agg2["BUREAU_SUM_RECENT_DPD_RATIO"] / (bureau_agg2["BUREAU_COUNT_RECENT_DPD_RATIO"] + 1e-9)
bureau_agg2["BUREAU_ACTIVE_COUNT"] = bureau_agg2["ACTIVE_COUNT"]
bureau_agg2["BUREAU_DEBT_CREDIT_RATIO"] = bureau_agg2["BUREAU_SUM_DEBT"] / (bureau_agg2["BUREAU_SUM_CREDIT"] + 1)
bureau_agg2["BUREAU_DPD_TREND"] = bureau_agg2["BUREAU_RECENT_DPD_RATIO"] - bureau_agg2["BUREAU_AVG_DPD_RATIO"]
bureau_agg2["BUREAU_RECENCY_WEIGHTED_OVERDUE"] = bureau_agg2["sum_weighted_overdue"] / (bureau_agg2["sum_weight"] + 1e-9)

bureau_agg2["mean_X"] = bureau_agg2["sum_X"] / bureau_agg2["BUREAU_LOAN_COUNT"]
bureau_agg2["mean_Y"] = bureau_agg2["sum_Y"] / bureau_agg2["BUREAU_LOAN_COUNT"]
bureau_agg2["mean_XY"] = bureau_agg2["sum_XY"] / bureau_agg2["BUREAU_LOAN_COUNT"]
bureau_agg2["mean_X2"] = bureau_agg2["sum_X2"] / bureau_agg2["BUREAU_LOAN_COUNT"]
bureau_agg2["BUREAU_DEBT_DAYS_SLOPE"] = (bureau_agg2["mean_XY"] - bureau_agg2["mean_X"] * bureau_agg2["mean_Y"]) / (bureau_agg2["mean_X2"] - bureau_agg2["mean_X"]**2 + 1e-9)
bureau_agg2["BUREAU_DEBT_DAYS_SLOPE"] = np.where(bureau_agg2["BUREAU_LOAN_COUNT"] < 2, 0, bureau_agg2["BUREAU_DEBT_DAYS_SLOPE"])
bureau_agg2["BUREAU_ACTIVE_CLOSED_RATIO"] = bureau_agg2["ACTIVE_COUNT"] / (bureau_agg2["CLOSED_COUNT"] + 1)

bureau_agg2 = bureau_agg2.drop(columns=["BUREAU_SUM_CREDIT", "BUREAU_COUNT_CREDIT", "BUREAU_SUM_DEBT", "BUREAU_SUM_DPD_RATIO", "BUREAU_COUNT_DPD_RATIO", "BUREAU_SUM_RECENT_DPD_RATIO", "BUREAU_COUNT_RECENT_DPD_RATIO", "sum_weighted_overdue", "sum_weight", "sum_X", "sum_Y", "sum_XY", "sum_X2", "ACTIVE_COUNT", "CLOSED_COUNT", "mean_X", "mean_Y", "mean_XY", "mean_X2"])
del bb_agg
gc.collect()
print(f"Bureau aggregations complete: {bureau_agg2.shape}")
'''))

cells.append(md("### Installments Payments Aggregations"))
cells.append(code('''
inst_agg2_chunks = []
inst_recent_chunks = []

for chunk in read_csv_in_chunks(DATA_DIR / "installments_payments.csv", usecols=["SK_ID_CURR", "DAYS_ENTRY_PAYMENT", "DAYS_INSTALMENT", "AMT_PAYMENT", "AMT_INSTALMENT"]):
    chunk["PAYMENT_DELAY"] = chunk["DAYS_ENTRY_PAYMENT"] - chunk["DAYS_INSTALMENT"]
    chunk["PAYMENT_DIFF"] = chunk["AMT_PAYMENT"] - chunk["AMT_INSTALMENT"]
    chunk["IS_LATE"] = np.where(chunk["DAYS_ENTRY_PAYMENT"].isna() | (chunk["PAYMENT_DELAY"] > 0), 1, 0)
    chunk["IS_UNDERPAID"] = (chunk["PAYMENT_DIFF"] < 0).astype(int)
    
    chunk["LATE_3M"] = np.where((chunk["DAYS_INSTALMENT"] >= -90) & (chunk["IS_LATE"] == 1), 1, 0)
    chunk["LATE_6M"] = np.where((chunk["DAYS_INSTALMENT"] >= -180) & (chunk["IS_LATE"] == 1), 1, 0)
    chunk["LATE_12M"] = np.where((chunk["DAYS_INSTALMENT"] >= -365) & (chunk["IS_LATE"] == 1), 1, 0)
    
    chunk["IN_3M"] = (chunk["DAYS_INSTALMENT"] >= -90).astype(int)
    chunk["IN_6M"] = (chunk["DAYS_INSTALMENT"] >= -180).astype(int)
    chunk["IN_12M"] = (chunk["DAYS_INSTALMENT"] >= -365).astype(int)
    
    chunk["PAYMENT_RATIO"] = chunk["AMT_PAYMENT"] / (chunk["AMT_INSTALMENT"] + 1e-9)
    chunk["XY_pay_ratio"] = chunk["DAYS_INSTALMENT"] * chunk["PAYMENT_RATIO"]
    chunk["X2_days"] = chunk["DAYS_INSTALMENT"] ** 2
    
    chunk["PAYMENT_DELAY_CLEAN"] = chunk["PAYMENT_DELAY"].fillna(0)
    chunk["XY_delay"] = chunk["DAYS_INSTALMENT"] * chunk["PAYMENT_DELAY_CLEAN"]
    
    chunk_agg = chunk.groupby("SK_ID_CURR").agg(
        count_inst=("DAYS_INSTALMENT", "size"),
        sum_delay=("PAYMENT_DELAY", "sum"),
        count_delay=("PAYMENT_DELAY", "count"),
        max_delay=("PAYMENT_DELAY", "max"),
        sum_late=("IS_LATE", "sum"),
        sum_diff=("PAYMENT_DIFF", "sum"),
        sum_underpaid=("IS_UNDERPAID", "sum"),
        sum_late_3m=("LATE_3M", "sum"),
        sum_in_3m=("IN_3M", "sum"),
        sum_late_6m=("LATE_6M", "sum"),
        sum_in_6m=("IN_6M", "sum"),
        sum_late_12m=("LATE_12M", "sum"),
        sum_in_12m=("IN_12M", "sum"),
        sum_X=("DAYS_INSTALMENT", "sum"),
        sum_Y_ratio=("PAYMENT_RATIO", "sum"),
        sum_XY_ratio=("XY_pay_ratio", "sum"),
        sum_X2=("X2_days", "sum"),
        sum_Y_delay=("PAYMENT_DELAY_CLEAN", "sum"),
        sum_XY_delay=("XY_delay", "sum")
    ).reset_index()
    
    inst_agg2_chunks.append(chunk_agg)
    
    rec_chunk = chunk[["SK_ID_CURR", "DAYS_INSTALMENT", "IS_LATE", "PAYMENT_DELAY"]].copy()
    rec_chunk = rec_chunk.sort_values(["SK_ID_CURR", "DAYS_INSTALMENT"], ascending=[True, False])
    rec_chunk = rec_chunk.groupby("SK_ID_CURR").head(5)
    inst_recent_chunks.append(rec_chunk)
    
    del chunk, chunk_agg, rec_chunk
    gc.collect()

inst_agg2 = pd.concat(inst_agg2_chunks, ignore_index=True)
del inst_agg2_chunks
gc.collect()

inst_agg2 = inst_agg2.groupby("SK_ID_CURR").agg(
    count_inst=("count_inst", "sum"),
    sum_delay=("sum_delay", "sum"),
    count_delay=("count_delay", "sum"),
    max_delay=("max_delay", "max"),
    sum_late=("sum_late", "sum"),
    sum_diff=("sum_diff", "sum"),
    sum_underpaid=("sum_underpaid", "sum"),
    sum_late_3m=("sum_late_3m", "sum"),
    sum_in_3m=("sum_in_3m", "sum"),
    sum_late_6m=("sum_late_6m", "sum"),
    sum_in_6m=("sum_in_6m", "sum"),
    sum_late_12m=("sum_late_12m", "sum"),
    sum_in_12m=("sum_in_12m", "sum"),
    sum_X=("sum_X", "sum"),
    sum_Y_ratio=("sum_Y_ratio", "sum"),
    sum_XY_ratio=("sum_XY_ratio", "sum"),
    sum_X2=("sum_X2", "sum"),
    sum_Y_delay=("sum_Y_delay", "sum"),
    sum_XY_delay=("sum_XY_delay", "sum")
).reset_index()

inst_agg2["AVG_PAYMENT_DELAY"] = inst_agg2["sum_delay"] / (inst_agg2["count_delay"] + 1e-9)
inst_agg2["MAX_PAYMENT_DELAY"] = inst_agg2["max_delay"]
inst_agg2["LATE_PAYMENT_RATIO"] = inst_agg2["sum_late"] / (inst_agg2["count_inst"] + 1e-9)
inst_agg2["AVG_PAYMENT_DIFF"] = inst_agg2["sum_diff"] / (inst_agg2["count_inst"] + 1e-9)
inst_agg2["UNDERPAID_RATIO"] = inst_agg2["sum_underpaid"] / (inst_agg2["count_inst"] + 1e-9)

inst_agg2["INST_LATE_RATIO_3M"] = inst_agg2["sum_late_3m"] / (inst_agg2["sum_in_3m"] + 1e-9)
inst_agg2["INST_LATE_RATIO_6M"] = inst_agg2["sum_late_6m"] / (inst_agg2["sum_in_6m"] + 1e-9)
inst_agg2["INST_LATE_RATIO_12M"] = inst_agg2["sum_late_12m"] / (inst_agg2["sum_in_12m"] + 1e-9)

inst_agg2["mean_X"] = inst_agg2["sum_X"] / inst_agg2["count_inst"]
inst_agg2["mean_Y_ratio"] = inst_agg2["sum_Y_ratio"] / inst_agg2["count_inst"]
inst_agg2["mean_XY_ratio"] = inst_agg2["sum_XY_ratio"] / inst_agg2["count_inst"]
inst_agg2["mean_X2"] = inst_agg2["sum_X2"] / inst_agg2["count_inst"]
inst_agg2["INST_PAY_RATIO_SLOPE"] = (inst_agg2["mean_XY_ratio"] - inst_agg2["mean_X"] * inst_agg2["mean_Y_ratio"]) / (inst_agg2["mean_X2"] - inst_agg2["mean_X"]**2 + 1e-9)
inst_agg2["INST_PAY_RATIO_SLOPE"] = np.where(inst_agg2["count_inst"] < 2, 0, inst_agg2["INST_PAY_RATIO_SLOPE"])

inst_agg2["mean_Y_delay"] = inst_agg2["sum_Y_delay"] / inst_agg2["count_inst"]
inst_agg2["mean_XY_delay"] = inst_agg2["sum_XY_delay"] / inst_agg2["count_inst"]
inst_agg2["INST_DELAY_SLOPE"] = (inst_agg2["mean_XY_delay"] - inst_agg2["mean_X"] * inst_agg2["mean_Y_delay"]) / (inst_agg2["mean_X2"] - inst_agg2["mean_X"]**2 + 1e-9)
inst_agg2["INST_DELAY_SLOPE"] = np.where(inst_agg2["count_inst"] < 2, 0, inst_agg2["INST_DELAY_SLOPE"])

inst_agg2 = inst_agg2.drop(columns=["sum_delay", "count_delay", "max_delay", "sum_late", "sum_diff", "sum_underpaid", "sum_late_3m", "sum_in_3m", "sum_late_6m", "sum_in_6m", "sum_late_12m", "sum_in_12m", "sum_X", "sum_Y_ratio", "sum_XY_ratio", "sum_X2", "sum_Y_delay", "sum_XY_delay", "mean_X", "mean_Y_ratio", "mean_XY_ratio", "mean_X2", "mean_Y_delay", "mean_XY_delay"])

inst_recent_combined = pd.concat(inst_recent_chunks, ignore_index=True)
del inst_recent_chunks
gc.collect()

inst_sorted = inst_recent_combined.sort_values(["SK_ID_CURR", "DAYS_INSTALMENT"], ascending=[True, False])
del inst_recent_combined
gc.collect()

inst_ranked = inst_sorted.groupby("SK_ID_CURR").head(5)
del inst_sorted
gc.collect()

inst_recent_agg = inst_ranked.groupby("SK_ID_CURR").agg(
    RECENT_LATE_RATIO=("IS_LATE", "mean"),
    RECENT_AVG_DELAY=("PAYMENT_DELAY", "mean"),
).reset_index()
del inst_ranked
gc.collect()

inst_agg2 = inst_agg2.merge(inst_recent_agg, on="SK_ID_CURR", how="left")
inst_agg2["LATE_RATIO_TREND"] = inst_agg2["RECENT_LATE_RATIO"] - inst_agg2["LATE_PAYMENT_RATIO"]

del inst_recent_agg
gc.collect()
print(f"Installments aggregations complete: {inst_agg2.shape}")
'''))

cells.append(md("### Previous Applications Aggregations"))
cells.append(code('''
prev_agg2_chunks = []
prev_recency_chunks = []
refused_only_chunks = []

for chunk in read_csv_in_chunks(DATA_DIR / "previous_application.csv", usecols=["SK_ID_CURR", "AMT_CREDIT", "NAME_CONTRACT_STATUS", "AMT_APPLICATION", "DAYS_DECISION"]):
    chunk_agg = chunk.groupby("SK_ID_CURR").agg(
        PREV_APP_COUNT=("SK_ID_CURR", "size"),
        PREV_SUM_CREDIT=("AMT_CREDIT", "sum"),
        PREV_COUNT_CREDIT=("AMT_CREDIT", "count")
    ).reset_index()
    
    chunk["_refused"] = (chunk["NAME_CONTRACT_STATUS"] == "Refused").astype(int)
    refused_chunk = (
        chunk.groupby("SK_ID_CURR")["_refused"]
        .agg(PREV_REFUSED_COUNT="sum", PREV_REFUSED_RATIO_SUM="sum", PREV_REFUSED_RATIO_COUNT="count")
        .reset_index()
    )
    
    credit_app_src = chunk[chunk["AMT_APPLICATION"] > 0].copy()
    credit_app_src["_ratio"] = credit_app_src["AMT_CREDIT"] / credit_app_src["AMT_APPLICATION"]
    credit_app_ratio_chunk = (
        credit_app_src.groupby("SK_ID_CURR")["_ratio"]
        .agg(sum_ratio="sum", count_ratio="count")
        .reset_index()
    )
    
    most_recent_chunk = (
        chunk.groupby("SK_ID_CURR")["DAYS_DECISION"].min()
        .rename("PREV_MOST_RECENT_DAYS")
        .reset_index()
    )
    
    chunk_merged = chunk_agg.merge(refused_chunk, on="SK_ID_CURR", how="left")
    chunk_merged = chunk_merged.merge(credit_app_ratio_chunk, on="SK_ID_CURR", how="left")
    chunk_merged = chunk_merged.merge(most_recent_chunk, on="SK_ID_CURR", how="left")
    
    prev_agg2_chunks.append(chunk_merged)
    
    prev_rec_chunk = chunk[["SK_ID_CURR", "DAYS_DECISION", "NAME_CONTRACT_STATUS"]].copy()
    prev_rec_chunk = prev_rec_chunk.sort_values(["SK_ID_CURR", "DAYS_DECISION"], ascending=[True, False])
    prev_rec_chunk = prev_rec_chunk.groupby("SK_ID_CURR").head(3)
    prev_recency_chunks.append(prev_rec_chunk)
    
    refused_only_chunk = chunk[chunk["NAME_CONTRACT_STATUS"] == "Refused"][["SK_ID_CURR", "DAYS_DECISION"]].copy()
    if not refused_only_chunk.empty:
        refused_only_chunk = refused_only_chunk.groupby("SK_ID_CURR")["DAYS_DECISION"].max().reset_index()
        refused_only_chunks.append(refused_only_chunk)
    
    del chunk, chunk_agg, refused_chunk, credit_app_src, credit_app_ratio_chunk, most_recent_chunk, chunk_merged, prev_rec_chunk, refused_only_chunk
    gc.collect()

prev_agg2 = pd.concat(prev_agg2_chunks, ignore_index=True)
del prev_agg2_chunks
gc.collect()

prev_agg2 = prev_agg2.groupby("SK_ID_CURR").agg(
    PREV_APP_COUNT=("PREV_APP_COUNT", "sum"),
    PREV_SUM_CREDIT=("PREV_SUM_CREDIT", "sum"),
    PREV_COUNT_CREDIT=("PREV_COUNT_CREDIT", "sum"),
    PREV_REFUSED_COUNT=("PREV_REFUSED_COUNT", "sum"),
    PREV_REFUSED_RATIO_SUM=("PREV_REFUSED_RATIO_SUM", "sum"),
    PREV_REFUSED_RATIO_COUNT=("PREV_REFUSED_RATIO_COUNT", "sum"),
    sum_ratio=("sum_ratio", "sum"),
    count_ratio=("count_ratio", "sum"),
    PREV_MOST_RECENT_DAYS=("PREV_MOST_RECENT_DAYS", "min")
).reset_index()

prev_agg2["PREV_AVG_CREDIT"] = prev_agg2["PREV_SUM_CREDIT"] / (prev_agg2["PREV_COUNT_CREDIT"] + 1e-9)
prev_agg2["PREV_REFUSED_RATIO"] = prev_agg2["PREV_REFUSED_RATIO_SUM"] / (prev_agg2["PREV_REFUSED_RATIO_COUNT"] + 1e-9)
prev_agg2["PREV_CREDIT_APP_RATIO"] = prev_agg2["sum_ratio"] / (prev_agg2["count_ratio"] + 1e-9)

prev_agg2 = prev_agg2.drop(columns=["PREV_SUM_CREDIT", "PREV_COUNT_CREDIT", "PREV_REFUSED_RATIO_SUM", "PREV_REFUSED_RATIO_COUNT", "sum_ratio", "count_ratio"])

prev_rec_combined = pd.concat(prev_recency_chunks, ignore_index=True)
del prev_recency_chunks
gc.collect()

prev_sorted = prev_rec_combined.sort_values(["SK_ID_CURR", "DAYS_DECISION"], ascending=[True, False])
del prev_rec_combined
gc.collect()

prev_recent3 = prev_sorted.groupby("SK_ID_CURR").head(3).copy()
del prev_sorted
gc.collect()

prev_recent3["is_refused"] = (prev_recent3["NAME_CONTRACT_STATUS"] == "Refused").astype(int)
prev_recent3_agg = prev_recent3.groupby("SK_ID_CURR")["is_refused"].mean().rename("PREV_REFUSED_RATIO_RECENT3").reset_index()
del prev_recent3
gc.collect()

prev_refused_combined = pd.concat(refused_only_chunks, ignore_index=True)
del refused_only_chunks
gc.collect()

prev_last_refused = (-prev_refused_combined.groupby("SK_ID_CURR")["DAYS_DECISION"].max()).rename("PREV_DAYS_SINCE_LAST_REFUSED").reset_index()
del prev_refused_combined
gc.collect()

prev_agg2 = prev_agg2.merge(prev_recent3_agg, on="SK_ID_CURR", how="left")
prev_agg2 = prev_agg2.merge(prev_last_refused, on="SK_ID_CURR", how="left")

del prev_recent3_agg, prev_last_refused
gc.collect()
print(f"Previous application aggregations complete: {prev_agg2.shape}")
'''))

cells.append(md("## 3. Merge All Aggregations and Perform Interaction Feature Engineering"))
cells.append(code('''
# Join aggregations onto base app dataset
bureau_overlap = ["BUREAU_MAX_OVERDUE", "BUREAU_AVG_CREDIT", "BUREAU_LOAN_COUNT", "PREV_APP_COUNT", "AVG_PAYMENT_DELAY", "MAX_PAYMENT_DELAY", "PREV_AVG_CREDIT"]
app_clean = app.drop(columns=[c for c in bureau_overlap if c in app.columns])

final = app_clean.merge(bureau_agg2, on="SK_ID_CURR", how="left")
del bureau_agg2, app_clean
gc.collect()

final = final.merge(prev_agg2, on="SK_ID_CURR", how="left")
del prev_agg2
gc.collect()

final = final.merge(inst_agg2, on="SK_ID_CURR", how="left")
del inst_agg2
gc.collect()

final = final.merge(pos_agg, on="SK_ID_CURR", how="left")
del pos_agg
gc.collect()

final = final.merge(cc_agg, on="SK_ID_CURR", how="left")
del cc_agg
gc.collect()

# Feature Engineering: Classic interaction ratios
ext_sources = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
final["EXT_SOURCE_MEAN"] = final[ext_sources].mean(axis=1)
final["EXT_SOURCE_STD"] = final[ext_sources].std(axis=1).fillna(0)
final["EXT_SOURCE_MIN"] = final[ext_sources].min(axis=1)
final["EXT_SOURCE_MAX"] = final[ext_sources].max(axis=1)
final["EXT_SOURCE_PROD"] = final["EXT_SOURCE_1"].fillna(1) * final["EXT_SOURCE_2"].fillna(1) * final["EXT_SOURCE_3"].fillna(1)
final["EXT_SOURCE_1_2_MULT"] = final["EXT_SOURCE_1"].fillna(1) * final["EXT_SOURCE_2"].fillna(1)
final["EXT_SOURCE_2_3_MULT"] = final["EXT_SOURCE_2"].fillna(1) * final["EXT_SOURCE_3"].fillna(1)
final["EXT_SOURCE_1_3_MULT"] = final["EXT_SOURCE_1"].fillna(1) * final["EXT_SOURCE_3"].fillna(1)

final["INCOME_PER_PERSON"] = final["AMT_INCOME_TOTAL"] / (final["CNT_FAM_MEMBERS"] + 1e-9)
final["CHILDREN_RATIO"] = final["CNT_CHILDREN"] / (final["CNT_FAM_MEMBERS"] + 1e-9)
final["DAYS_EMPLOYED_PERCENT"] = final["DAYS_EMPLOYED"] / (final["DAYS_BIRTH"] + 1e-9)
final["INCOME_CREDIT_PERC"] = final["AMT_INCOME_TOTAL"] / (final["AMT_CREDIT"] + 1e-9)
final["DAYS_LAST_PHONE_CHANGE_BIRTH_RATIO"] = final["DAYS_LAST_PHONE_CHANGE"] / (final["DAYS_BIRTH"] + 1e-9)
final["DAYS_REGISTRATION_BIRTH_RATIO"] = final["DAYS_REGISTRATION"] / (final["DAYS_BIRTH"] + 1e-9)
final["CAR_TO_BIRTH_RATIO"] = final["OWN_CAR_AGE"] / (final["DAYS_BIRTH"] + 1e-9)
final["CAR_TO_EMPLOYED_RATIO"] = final["OWN_CAR_AGE"] / (final["DAYS_EMPLOYED"] + 1e-9)

# Define classic ratios from basic cleaning
final["CREDIT_INCOME_RATIO"] = final["AMT_CREDIT"] / (final["AMT_INCOME_TOTAL"] + 1)
final["ANNUITY_INCOME_RATIO"] = final["AMT_ANNUITY"] / (final["AMT_INCOME_TOTAL"] + 1)
final["CREDIT_TERM"] = final["AMT_CREDIT"] / (final["AMT_ANNUITY"] + 1)
final["AGE_YEARS"] = -final["DAYS_BIRTH"] / 365.0

# Fill null values for engineered features with sensible defaults
zero_fill_cols = [
    "BUREAU_LOAN_COUNT", "BUREAU_ACTIVE_COUNT", "PREV_APP_COUNT", "PREV_REFUSED_COUNT", "PREV_REFUSED_RATIO",
    "POS_COUNT", "POS_DPD_RATIO", "CC_COUNT", "CC_OVER_LIMIT_RATIO", "LATE_PAYMENT_RATIO", "UNDERPAID_RATIO", "RECENT_LATE_RATIO",
    "BUREAU_RECENCY_WEIGHTED_OVERDUE", "BUREAU_DEBT_DAYS_SLOPE", "BUREAU_ACTIVE_CLOSED_RATIO",
    "INST_LATE_RATIO_3M", "INST_LATE_RATIO_6M", "INST_LATE_RATIO_12M", "INST_PAY_RATIO_SLOPE", "INST_DELAY_SLOPE",
    "POS_DPD_SLOPE", "POS_COMPLETION_RATE_DIFF", "CC_UTILIZATION_TREND_6M", "CC_DRAWINGS_DIFF_6M", "PREV_REFUSED_RATIO_RECENT3"
]
existing_zero_fill = [c for c in zero_fill_cols if c in final.columns]
final[existing_zero_fill] = final[existing_zero_fill].fillna(0)
final["PREV_DAYS_SINCE_LAST_REFUSED"] = final["PREV_DAYS_SINCE_LAST_REFUSED"].fillna(99999)

print(f"Final merged dataset: {final.shape}")
'''))

cells.append(md("## 4. Feature Selection"))
cells.append(md('''We apply 4 methods of feature selection to identify the most robust predictors:
1. **Correlation Analysis:** Keeps features with target correlation >= 0.02.
2. **XGBoost Feature Importance:** Keeps top features cumulative 90% importances.
3. **Recursive Feature Elimination (RFE):** Iteratively eliminates weak features with Logistic Regression.
4. **Lasso (L1) Regularization:** Keeps features with non-zero weights under L1 penalty.

We keep features selected by at least 2 methods.'''))

cells.append(code('''
# One-hot encode categorical features before feature selection
categorical_cols = [
    'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
    'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
    'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE', 'WEEKDAY_APPR_PROCESS_START', 'ORGANIZATION_TYPE',
    'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE'
]
df_encoded = pd.get_dummies(final, columns=[c for c in categorical_cols if c in final.columns], dummy_na=True)
import re
df_encoded = df_encoded.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '_', str(x)))

# Perform selection on a sample of the dataset to keep it fast
fs_sample = df_encoded.sample(frac=0.25, random_state=42)
exclude_cols = ["SK_ID_CURR", "TARGET"]
candidate_cols = [c for c, dtype in zip(fs_sample.columns, fs_sample.dtypes) if c not in exclude_cols and pd.api.types.is_numeric_dtype(dtype)]

X_fs = fs_sample[candidate_cols].copy().dropna(axis=1, how="all")
y_fs = fs_sample["TARGET"]
candidate_cols = X_fs.columns.tolist()

# Median-impute remaining nulls for feature selection methods
X_fs = X_fs.fillna(X_fs.median())
print(f"Candidates considered for selection: {len(candidate_cols)}")
'''))

cells.append(md("### 1. Correlation Analysis Selection"))
cells.append(code('''
corr_with_target = X_fs.corrwith(y_fs).abs().sort_values(ascending=False)
corr_selected = set(corr_with_target[corr_with_target >= 0.02].index)
print(f"Correlation selected: {len(corr_selected)}")
'''))

cells.append(md("### 2. XGBoost Feature Importance Selection"))
cells.append(code('''
from xgboost import XGBClassifier
xgb_fs = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, eval_metric="logloss")
xgb_fs.fit(X_fs, y_fs)
importances = pd.Series(xgb_fs.feature_importances_, index=candidate_cols).sort_values(ascending=False)
cum_importance = importances.cumsum() / importances.sum()
xgb_selected = set(cum_importance[cum_importance <= 0.90].index)
print(f"XGBoost selected: {len(xgb_selected)}")
'''))

cells.append(md("### 3. Recursive Feature Elimination Selection"))
cells.append(code('''
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_fs_scaled = scaler.fit_transform(X_fs)

rfe = RFE(
    estimator=LogisticRegression(max_iter=1000, class_weight="balanced"),
    n_features_to_select=min(30, len(candidate_cols)),
    step=15
)
rfe.fit(X_fs_scaled, y_fs)
rfe_selected = set(col for col, keep in zip(candidate_cols, rfe.support_) if keep)
print(f"RFE selected: {len(rfe_selected)}")
'''))

cells.append(md("### 4. Lasso Selection"))
cells.append(code('''
from sklearn.linear_model import LogisticRegression as LassoLogReg
lasso = LassoLogReg(penalty="l1", solver="liblinear", C=0.05, class_weight="balanced", max_iter=1000, random_state=42)
lasso.fit(X_fs_scaled, y_fs)
lasso_coefs = pd.Series(lasso.coef_[0], index=candidate_cols)
lasso_selected = set(lasso_coefs[lasso_coefs.abs() > 1e-4].index)
print(f"Lasso selected: {len(lasso_selected)}")
'''))

cells.append(md("### Combine Selections"))
cells.append(code('''
comparison = pd.DataFrame({"feature": candidate_cols})
comparison["Correlation"] = comparison["feature"].isin(corr_selected)
comparison["XGBoost"] = comparison["feature"].isin(xgb_selected)
comparison["RFE"] = comparison["feature"].isin(rfe_selected)
comparison["Lasso"] = comparison["feature"].isin(lasso_selected)
comparison["Selected_Count"] = comparison[["Correlation", "XGBoost", "RFE", "Lasso"]].sum(axis=1)

comparison = comparison.sort_values("Selected_Count", ascending=False)
display(comparison.head(30))

final_features = comparison[comparison["Selected_Count"] >= 2]["feature"].tolist()
print(f"Total features selected (>= 2 methods): {len(final_features)}")
'''))

cells.append(md("## 5. Dimensionality Reduction (PCA & LDA)"))
cells.append(code('''
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt

# Extract and scale final features
X_final = X_fs[final_features]
scaler_final = StandardScaler()
X_final_scaled = scaler_final.fit_transform(X_final)

# Run PCA
pca = PCA()
X_pca = pca.fit_transform(X_final_scaled)
cum_var = np.cumsum(pca.explained_variance_ratio_)

plt.figure(figsize=(8,4))
plt.plot(cum_var, marker="o", color="darkblue")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance")
plt.grid()
plt.show()

n_components_95 = np.argmax(cum_var >= 0.95) + 1
print(f"PCA components explaining 95% variance: {n_components_95}")

# Run LDA
lda = LinearDiscriminantAnalysis(n_components=1)
X_lda = lda.fit_transform(X_final_scaled, y_fs)

plt.figure(figsize=(8,3))
plt.scatter(X_lda[y_fs==0], np.zeros(sum(y_fs==0)), alpha=0.3, label="Non-Default", color="green")
plt.scatter(X_lda[y_fs==1], np.zeros(sum(y_fs==1)), alpha=0.3, label="Default", color="red")
plt.legend()
plt.title("LDA 1D Projection")
plt.show()
'''))

cells.append(md("## 6. Persist Outputs"))
cells.append(code('''
import json

# Save features selection metadata
comparison.to_csv(PROCESSED_DIR / "feature_selection_results.csv", index=False)
print("Saved:", PROCESSED_DIR / "feature_selection_results.csv")

# Save final feature names as list
with open(MODELS_DIR / "final_features.json", "w") as f:
    json.dump(final_features, f)
print("Saved:", MODELS_DIR / "final_features.json")

# Save final merged dataset
df_encoded.to_parquet(PROCESSED_DIR / "model_dataset.parquet", index=False)
print("Saved:", PROCESSED_DIR / "model_dataset.parquet")
print(f"Parquet shape: {df_encoded.shape}")
'''))

save(cells, "02_DataCleaning_FeatureEngineering_FeatureSelection.ipynb", "02")
