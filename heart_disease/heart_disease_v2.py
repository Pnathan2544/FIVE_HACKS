"""
Heart Disease Prediction - V2 (F2-Score Optimized)
====================================================
F2 = (1+4) * precision * recall / (4*precision + recall)
F2 heavily weights RECALL — we need to catch as many true positives as possible.
Strategy: lower thresholds, higher scale_pos_weight, recall-biased models.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, fbeta_score, roc_auc_score,
    classification_report, precision_score, recall_score
)
from scipy.stats import rankdata
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings("ignore")


def f2_score(y_true, y_pred):
    """F2 score — recall weighted 2x more than precision."""
    return fbeta_score(y_true, y_pred, beta=2)


# ============================================================
# 1. LOAD DATA
# ============================================================
print("=" * 60)
print("1. LOADING DATA")
print("=" * 60)

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sample_sub = pd.read_csv("sample_submission.csv")

TARGET = "History of HeartDisease or Attack"

train = train.dropna(subset=[TARGET]).reset_index(drop=True)
y = (train[TARGET] == "Yes").astype(int)

print(f"Train: {train.shape}, Test: {test.shape}")
print(f"Positive rate: {y.mean():.4f} ({y.sum()} / {len(y)})")

train_ids = train["ID"]
test_ids = test["ID"]
train_feat = train.drop(columns=["ID", TARGET])
test_feat = test.drop(columns=["ID"])

combined = pd.concat([train_feat, test_feat], axis=0).reset_index(drop=True)
n_train = len(train_feat)

# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 60)
print("2. FEATURE ENGINEERING")
print("=" * 60)

# --- Binary Yes/No columns ---
binary_cols = [
    "High Blood Pressure", "Told High Cholesterol", "Cholesterol Checked",
    "Smoked 100+ Cigarettes", "Diagnosed Stroke", "Diagnosed Diabetes",
    "Leisure Physical Activity", "Heavy Alcohol Consumption",
    "Health Care Coverage", "Doctor Visit Cost Barrier",
    "Difficulty Walking", "Vegetable or Fruit Intake (1+ per Day)"
]
for col in binary_cols:
    combined[col] = combined[col].map({"Yes": 1, "No": 0})

combined["Sex"] = combined["Sex"].map({"Male": 1, "Female": 0})

# --- Ordinal encoding ---
health_order = {"Very Poor": 0, "Poor": 1, "Fair": 2, "Good": 3, "Very Good": 4, "Excellent": 5}
combined["General Health"] = combined["General Health"].map(health_order)

edu_order = {
    "Never attended school": 0, "Elementary": 1, "Some high school": 2,
    "High school graduate": 3, "Some college or technical school": 4, "College graduate": 5
}
combined["Education Level"] = combined["Education Level"].map(edu_order)

income_map = {
    "Less than $10,000": 0,
    "($10,000 to less than $15,000": 1, "$10,000 to less than $15,000": 1,
    "$15,000 to less than $20,000": 2, "$20,000 to less than $25,000": 3,
    "$25,000 to less than $35,000": 4, "$35,000 to less than $50,000": 5,
    "$50,000 to less than $75,000": 6, "$75,000 or more": 7
}
combined["Income Level"] = combined["Income Level"].map(income_map)

# --- DOMAIN-DRIVEN FEATURES ---

# Risk factor count
risk_cols = ["High Blood Pressure", "Told High Cholesterol", "Smoked 100+ Cigarettes",
             "Diagnosed Stroke", "Diagnosed Diabetes", "Difficulty Walking"]
combined["Risk Factor Count"] = combined[risk_cols].sum(axis=1)

# Protective factor count
protect_cols = ["Leisure Physical Activity", "Health Care Coverage",
                "Vegetable or Fruit Intake (1+ per Day)"]
combined["Protective Factor Count"] = combined[protect_cols].sum(axis=1)

# Net risk
combined["Net Risk Score"] = combined["Risk Factor Count"] - combined["Protective Factor Count"]

# BMI categories
combined["BMI_Obese"] = (combined["Body Mass Index"] >= 30).astype(float)
combined["BMI_Overweight"] = ((combined["Body Mass Index"] >= 25) & (combined["Body Mass Index"] < 30)).astype(float)

# Age groups
combined["Age_Senior"] = (combined["Age"] >= 60).astype(int)
combined["Age_Elderly"] = (combined["Age"] >= 75).astype(int)

# Key interactions
combined["Age_x_HighBP"] = combined["Age"] * combined["High Blood Pressure"]
combined["Age_x_HighChol"] = combined["Age"] * combined["Told High Cholesterol"]
combined["Age_x_Diabetes"] = combined["Age"] * combined["Diagnosed Diabetes"]
combined["Age_x_BMI"] = combined["Age"] * combined["Body Mass Index"]
combined["Age_x_Smoking"] = combined["Age"] * combined["Smoked 100+ Cigarettes"]
combined["Age_x_RiskCount"] = combined["Age"] * combined["Risk Factor Count"]

combined["BMI_x_HighBP"] = combined["Body Mass Index"] * combined["High Blood Pressure"]
combined["BMI_x_Diabetes"] = combined["Body Mass Index"] * combined["Diagnosed Diabetes"]

# Composites
combined["BP_Chol_Combo"] = combined["High Blood Pressure"] + combined["Told High Cholesterol"]
combined["Cardio_Risk"] = (combined["High Blood Pressure"] + combined["Told High Cholesterol"] +
                           combined["Diagnosed Diabetes"] + combined["Diagnosed Stroke"])

# Poor health
combined["Poor_Health"] = (combined["General Health"] <= 1).astype(float)

# Age squared
combined["Age_Squared"] = combined["Age"] ** 2

# Healthcare barrier
combined["Healthcare_Barrier"] = ((combined["Health Care Coverage"] == 0) |
                                   (combined["Doctor Visit Cost Barrier"] == 1)).astype(float)

print(f"Total features: {combined.shape[1]}")

# Split back
X_train = combined.iloc[:n_train].copy()
X_test = combined.iloc[n_train:].reset_index(drop=True).copy()

# ============================================================
# 3. TARGET ENCODING (CV-safe)
# ============================================================
print("\n" + "=" * 60)
print("3. TARGET ENCODING")
print("=" * 60)

te_cols = ["General Health", "Education Level", "Income Level", "Age"]
te_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=99)

for col in te_cols:
    te_col_name = f"{col}_TE"
    X_train[te_col_name] = 0.0
    global_mean = y.mean()

    for tr_idx, val_idx in te_skf.split(X_train, y):
        means = y.iloc[tr_idx].groupby(X_train[col].iloc[tr_idx]).mean()
        X_train.loc[X_train.index[val_idx], te_col_name] = (
            X_train[col].iloc[val_idx].map(means).fillna(global_mean)
        )

    full_means = y.groupby(X_train[col]).mean()
    X_test[te_col_name] = X_test[col].map(full_means).fillna(global_mean)

feature_names = X_train.columns.tolist()
print(f"Features after target encoding: {len(feature_names)}")

# ============================================================
# 4. MODEL TRAINING - 10-Fold CV
# ============================================================
print("\n" + "=" * 60)
print("4. MODEL TRAINING - 10-FOLD CV")
print("=" * 60)

N_SPLITS = 10
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

neg_count = (y == 0).sum()
pos_count = (y == 1).sum()
scale_pos = neg_count / pos_count

# For F2 optimization, we want even MORE recall bias
# Use higher scale_pos_weight to push models toward recall
scale_pos_boosted = scale_pos * 1.5  # Extra recall push
print(f"Base scale_pos_weight: {scale_pos:.2f}")
print(f"Boosted scale_pos_weight: {scale_pos_boosted:.2f}")

# Storage
oof_lgb = np.zeros(len(X_train))
oof_xgb = np.zeros(len(X_train))
oof_cat = np.zeros(len(X_train))
test_lgb = np.zeros(len(X_test))
test_xgb = np.zeros(len(X_test))
test_cat = np.zeros(len(X_test))

# Also train with multiple seeds for robustness
SEEDS = [42, 123, 2024]

# --- LightGBM ---
print("\n--- LightGBM (multi-seed) ---")
oof_lgb_all = np.zeros(len(X_train))
test_lgb_all = np.zeros(len(X_test))

for seed in SEEDS:
    oof_lgb_seed = np.zeros(len(X_train))
    test_lgb_seed = np.zeros(len(X_test))
    skf_seed = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

    lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 127,
        "max_depth": 8,
        "min_child_samples": 80,
        "subsample": 0.75,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "scale_pos_weight": scale_pos_boosted,
        "min_gain_to_split": 0.01,
        "verbose": -1,
        "n_jobs": -1,
        "random_state": seed,
    }

    for fold, (train_idx, val_idx) in enumerate(skf_seed.split(X_train, y)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(**lgb_params, n_estimators=5000)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)],
        )
        oof_lgb_seed[val_idx] = model.predict_proba(X_val)[:, 1]
        test_lgb_seed += model.predict_proba(X_test)[:, 1] / N_SPLITS

    auc = roc_auc_score(y, oof_lgb_seed)
    print(f"  Seed {seed} OOF AUC: {auc:.5f}")
    oof_lgb_all += oof_lgb_seed / len(SEEDS)
    test_lgb_all += test_lgb_seed / len(SEEDS)

oof_lgb = oof_lgb_all
test_lgb = test_lgb_all
lgb_auc = roc_auc_score(y, oof_lgb)
print(f"  LightGBM Multi-Seed OOF AUC: {lgb_auc:.5f}")

# --- XGBoost ---
print("\n--- XGBoost (multi-seed) ---")
oof_xgb_all = np.zeros(len(X_train))
test_xgb_all = np.zeros(len(X_test))

for seed in SEEDS:
    oof_xgb_seed = np.zeros(len(X_train))
    test_xgb_seed = np.zeros(len(X_test))
    skf_seed = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "learning_rate": 0.03,
        "max_depth": 8,
        "min_child_weight": 80,
        "subsample": 0.75,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "scale_pos_weight": scale_pos_boosted,
        "gamma": 0.01,
        "tree_method": "hist",
        "random_state": seed,
        "n_jobs": -1,
        "verbosity": 0,
    }

    for fold, (train_idx, val_idx) in enumerate(skf_seed.split(X_train, y)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = xgb.XGBClassifier(**xgb_params, n_estimators=5000, early_stopping_rounds=150)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        oof_xgb_seed[val_idx] = model.predict_proba(X_val)[:, 1]
        test_xgb_seed += model.predict_proba(X_test)[:, 1] / N_SPLITS

    auc = roc_auc_score(y, oof_xgb_seed)
    print(f"  Seed {seed} OOF AUC: {auc:.5f}")
    oof_xgb_all += oof_xgb_seed / len(SEEDS)
    test_xgb_all += test_xgb_seed / len(SEEDS)

oof_xgb = oof_xgb_all
test_xgb = test_xgb_all
xgb_auc = roc_auc_score(y, oof_xgb)
print(f"  XGBoost Multi-Seed OOF AUC: {xgb_auc:.5f}")

# --- CatBoost ---
print("\n--- CatBoost (multi-seed) ---")
oof_cat_all = np.zeros(len(X_train))
test_cat_all = np.zeros(len(X_test))

for seed in SEEDS:
    oof_cat_seed = np.zeros(len(X_train))
    test_cat_seed = np.zeros(len(X_test))
    skf_seed = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(skf_seed.split(X_train, y)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = CatBoostClassifier(
            iterations=5000,
            learning_rate=0.03,
            depth=8,
            l2_leaf_reg=5.0,
            subsample=0.75,
            colsample_bylevel=0.7,
            auto_class_weights="Balanced",
            eval_metric="AUC",
            early_stopping_rounds=150,
            random_seed=seed,
            verbose=0,
        )
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val))

        oof_cat_seed[val_idx] = model.predict_proba(X_val)[:, 1]
        test_cat_seed += model.predict_proba(X_test)[:, 1] / N_SPLITS

    auc = roc_auc_score(y, oof_cat_seed)
    print(f"  Seed {seed} OOF AUC: {auc:.5f}")
    oof_cat_all += oof_cat_seed / len(SEEDS)
    test_cat_all += test_cat_seed / len(SEEDS)

oof_cat = oof_cat_all
test_cat = test_cat_all
cat_auc = roc_auc_score(y, oof_cat)
print(f"  CatBoost Multi-Seed OOF AUC: {cat_auc:.5f}")

# ============================================================
# 5. ENSEMBLE + F2-OPTIMIZED THRESHOLD
# ============================================================
print("\n" + "=" * 60)
print("5. ENSEMBLE + F2 THRESHOLD OPTIMIZATION")
print("=" * 60)

# Weighted average ensemble
total_auc = lgb_auc + xgb_auc + cat_auc
w_lgb = lgb_auc / total_auc
w_xgb = xgb_auc / total_auc
w_cat = cat_auc / total_auc
print(f"Weights -> LGB: {w_lgb:.4f}, XGB: {w_xgb:.4f}, CAT: {w_cat:.4f}")

oof_weighted = w_lgb * oof_lgb + w_xgb * oof_xgb + w_cat * oof_cat
test_weighted = w_lgb * test_lgb + w_xgb * test_xgb + w_cat * test_cat

# Rank average ensemble
oof_rank = (rankdata(oof_lgb) + rankdata(oof_xgb) + rankdata(oof_cat)) / 3
test_rank = (rankdata(test_lgb) + rankdata(test_xgb) + rankdata(test_cat)) / 3
oof_rank = (oof_rank - oof_rank.min()) / (oof_rank.max() - oof_rank.min())
test_rank = (test_rank - test_rank.min()) / (test_rank.max() - test_rank.min())

# Simple average
oof_simple = (oof_lgb + oof_xgb + oof_cat) / 3
test_simple = (test_lgb + test_xgb + test_cat) / 3

# Evaluate all ensemble methods with F2-optimized threshold
ensembles = {
    "Weighted Avg": (oof_weighted, test_weighted),
    "Rank Avg": (oof_rank, test_rank),
    "Simple Avg": (oof_simple, test_simple),
}

best_ensemble_name = None
best_ensemble_f2 = 0
best_ensemble_thresh = 0.5

for name, (oof, test_p) in ensembles.items():
    auc = roc_auc_score(y, oof)
    # Find best F2 threshold
    best_t, best_f2 = 0.5, 0
    for t in np.arange(0.03, 0.80, 0.002):
        preds = (oof >= t).astype(int)
        f2 = f2_score(y, preds)
        if f2 > best_f2:
            best_f2 = f2
            best_t = t

    preds = (oof >= best_t).astype(int)
    prec = precision_score(y, preds)
    rec = recall_score(y, preds)
    print(f"  {name}: AUC={auc:.5f} | Best F2={best_f2:.5f} @ thresh={best_t:.3f} | P={prec:.3f} R={rec:.3f}")

    if best_f2 > best_ensemble_f2:
        best_ensemble_f2 = best_f2
        best_ensemble_name = name
        best_ensemble_thresh = best_t

print(f"\n>>> Best ensemble: {best_ensemble_name} with F2={best_ensemble_f2:.5f}")

oof_final, test_final = ensembles[best_ensemble_name]

# Detailed F2 threshold analysis around optimum
print("\n--- F2 Threshold Sensitivity ---")
for t in np.arange(max(0.01, best_ensemble_thresh - 0.05),
                    best_ensemble_thresh + 0.06, 0.01):
    preds = (oof_final >= t).astype(int)
    f2 = f2_score(y, preds)
    prec = precision_score(y, preds)
    rec = recall_score(y, preds)
    yes_pct = preds.mean() * 100
    print(f"  thresh={t:.3f} | F2={f2:.5f} | P={prec:.3f} | R={rec:.3f} | Yes%={yes_pct:.1f}%")

# Final OOF report with best threshold
print(f"\n--- OOF Report (F2-optimized thresh={best_ensemble_thresh:.3f}) ---")
oof_preds = (oof_final >= best_ensemble_thresh).astype(int)
print(classification_report(y, oof_preds, target_names=["No", "Yes"]))
print(f"F2 Score: {f2_score(y, oof_preds):.5f}")
print(f"AUC:      {roc_auc_score(y, oof_final):.5f}")

# ============================================================
# 6. GENERATE SUBMISSIONS
# ============================================================
print("\n" + "=" * 60)
print("6. GENERATE SUBMISSIONS")
print("=" * 60)

# Primary: F2-optimized threshold
test_preds = (test_final >= best_ensemble_thresh).astype(int)
test_labels = pd.Series(test_preds).map({1: "Yes", 0: "No"})
submission = pd.DataFrame({"ID": test_ids, TARGET: test_labels.values})
submission.to_csv("submission_v2_f2opt.csv", index=False)
yes_pct = test_preds.mean() * 100
print(f"  submission_v2_f2opt.csv (thresh={best_ensemble_thresh:.3f}) -> Yes: {yes_pct:.1f}%")

# Also generate slightly more aggressive (lower threshold = more recall)
for delta in [-0.03, -0.02, -0.01, 0.01, 0.02]:
    t = best_ensemble_thresh + delta
    preds = (test_final >= t).astype(int)
    labels = pd.Series(preds).map({1: "Yes", 0: "No"})
    sub = pd.DataFrame({"ID": test_ids, TARGET: labels.values})
    fname = f"submission_v2_thresh{t:.3f}.csv"
    sub.to_csv(fname, index=False)
    yes_pct = preds.mean() * 100
    print(f"  {fname} -> Yes: {yes_pct:.1f}%")

# Save probabilities
prob_sub = pd.DataFrame({"ID": test_ids, "probability": test_final})
prob_sub.to_csv("submission_v2_probabilities.csv", index=False)

print("\n" + "=" * 60)
print(f"DONE! Primary submission: submission_v2_f2opt.csv")
print(f"OOF F2 Score: {best_ensemble_f2:.5f}")
print("=" * 60)
