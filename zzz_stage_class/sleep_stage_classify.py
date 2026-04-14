"""
Sleep Stage Classification Pipeline
Phase 1: Baseline LightGBM with hand-crafted features
Phase 2: Frequency-domain + inter-channel features
Phase 3: Ensemble (LightGBM + XGBoost)
Phase 4: Per-subject normalization + class balancing + smoothing
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy import stats, signal
from scipy.fft import rfft, rfftfreq
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "train", "train")
TEST_DIR = os.path.join(BASE_DIR, "test_segment", "test_segment")
SUBMISSION_FILE = os.path.join(BASE_DIR, "sample_submission.csv")
FS = 16  # sampling rate
SEGMENT_LEN = 30 * FS  # 480 samples per 30-second segment

CHANNELS = ["BVP", "ACC_X", "ACC_Y", "ACC_Z", "TEMP", "EDA", "HR", "IBI"]


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def time_domain_features(segment, ch_name):
    """Basic statistical features for a single channel segment."""
    prefix = ch_name
    feats = {}
    feats[f"{prefix}_mean"] = np.mean(segment)
    feats[f"{prefix}_std"] = np.std(segment)
    feats[f"{prefix}_min"] = np.min(segment)
    feats[f"{prefix}_max"] = np.max(segment)
    feats[f"{prefix}_median"] = np.median(segment)
    feats[f"{prefix}_ptp"] = np.ptp(segment)  # peak-to-peak
    feats[f"{prefix}_skew"] = stats.skew(segment)
    feats[f"{prefix}_kurtosis"] = stats.kurtosis(segment)
    feats[f"{prefix}_iqr"] = np.percentile(segment, 75) - np.percentile(segment, 25)
    feats[f"{prefix}_energy"] = np.sum(segment ** 2)
    feats[f"{prefix}_rms"] = np.sqrt(np.mean(segment ** 2))
    # zero crossing rate
    feats[f"{prefix}_zcr"] = np.sum(np.diff(np.sign(segment - np.mean(segment))) != 0) / len(segment)
    # percentiles
    feats[f"{prefix}_p10"] = np.percentile(segment, 10)
    feats[f"{prefix}_p90"] = np.percentile(segment, 90)
    # mean absolute deviation
    feats[f"{prefix}_mad"] = np.mean(np.abs(segment - np.mean(segment)))
    return feats


def freq_domain_features(segment, ch_name):
    """Frequency-domain features using FFT."""
    prefix = ch_name
    feats = {}
    n = len(segment)
    yf = np.abs(rfft(segment - np.mean(segment)))
    xf = rfftfreq(n, 1.0 / FS)

    total_power = np.sum(yf ** 2) + 1e-10
    feats[f"{prefix}_total_power"] = total_power

    # power in frequency bands (relevant for physiological signals)
    bands = [(0.1, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 8.0)]
    for i, (lo, hi) in enumerate(bands):
        mask = (xf >= lo) & (xf < hi)
        feats[f"{prefix}_band{i}_power"] = np.sum(yf[mask] ** 2) / total_power

    # dominant frequency
    if len(yf) > 1:
        feats[f"{prefix}_dom_freq"] = xf[np.argmax(yf[1:]) + 1]
    else:
        feats[f"{prefix}_dom_freq"] = 0

    # spectral entropy
    psd = yf ** 2 / total_power
    psd = psd[psd > 0]
    feats[f"{prefix}_spectral_entropy"] = -np.sum(psd * np.log2(psd + 1e-12))

    # spectral centroid
    feats[f"{prefix}_spectral_centroid"] = np.sum(xf * yf) / (np.sum(yf) + 1e-10)

    return feats


def inter_channel_features(segment_dict):
    """Features combining multiple channels."""
    feats = {}
    # Accelerometer magnitude
    acc_mag = np.sqrt(
        segment_dict["ACC_X"] ** 2 +
        segment_dict["ACC_Y"] ** 2 +
        segment_dict["ACC_Z"] ** 2
    )
    feats["acc_mag_mean"] = np.mean(acc_mag)
    feats["acc_mag_std"] = np.std(acc_mag)
    feats["acc_mag_max"] = np.max(acc_mag)
    feats["acc_mag_energy"] = np.sum(acc_mag ** 2)

    # HR-IBI relationship
    hr = segment_dict["HR"]
    ibi = segment_dict["IBI"]
    if np.std(hr) > 0 and np.std(ibi) > 0:
        feats["hr_ibi_corr"] = np.corrcoef(hr, ibi)[0, 1]
    else:
        feats["hr_ibi_corr"] = 0

    # HR variability (std of IBI is a proxy for HRV)
    feats["hrv_sdnn"] = np.std(ibi)
    feats["hrv_rmssd"] = np.sqrt(np.mean(np.diff(ibi) ** 2)) if len(ibi) > 1 else 0

    # TEMP-EDA interaction
    feats["temp_eda_product_mean"] = np.mean(segment_dict["TEMP"] * segment_dict["EDA"])

    # BVP amplitude
    feats["bvp_pp_amplitude"] = np.ptp(segment_dict["BVP"])

    return feats


def extract_features_from_segment(segment_df):
    """Extract all features from a single 30-second segment DataFrame."""
    all_feats = {}
    segment_dict = {}

    for ch in CHANNELS:
        values = segment_df[ch].values.astype(float)
        segment_dict[ch] = values
        all_feats.update(time_domain_features(values, ch))
        all_feats.update(freq_domain_features(values, ch))

    all_feats.update(inter_channel_features(segment_dict))
    return all_feats


# ============================================================
# DATA LOADING
# ============================================================

def load_train_data():
    """Load and segment all training files."""
    print("Loading training data...")
    train_files = sorted(glob.glob(os.path.join(TRAIN_DIR, "*.csv")))
    all_features = []
    all_labels = []
    all_groups = []  # subject IDs for GroupKFold

    for fi, fpath in enumerate(train_files):
        subject_id = os.path.basename(fpath).replace(".csv", "")
        print(f"  [{fi+1}/{len(train_files)}] Processing {subject_id}...")
        df = pd.read_csv(fpath)

        n_segments = len(df) // SEGMENT_LEN
        for seg_i in range(n_segments):
            start = seg_i * SEGMENT_LEN
            end = start + SEGMENT_LEN
            seg_df = df.iloc[start:end]

            # majority label for this segment
            label = seg_df["Sleep_Stage"].mode()[0]

            feats = extract_features_from_segment(seg_df)
            all_features.append(feats)
            all_labels.append(label)
            all_groups.append(fi)  # group by subject

    X = pd.DataFrame(all_features)
    y = np.array(all_labels)
    groups = np.array(all_groups)
    print(f"Training data: {X.shape[0]} segments, {X.shape[1]} features")
    return X, y, groups


def load_test_data():
    """Load pre-segmented test data."""
    print("Loading test data...")
    submission = pd.read_csv(SUBMISSION_FILE)
    test_ids = submission["id"].values

    all_features = []
    for seg_id in test_ids:
        # parse subject from segment ID: test001_00000 -> test001
        subject = seg_id.rsplit("_", 1)[0]
        fpath = os.path.join(TEST_DIR, subject, f"{seg_id}.csv")
        seg_df = pd.read_csv(fpath)
        feats = extract_features_from_segment(seg_df)
        all_features.append(feats)

    X_test = pd.DataFrame(all_features)
    print(f"Test data: {X_test.shape[0]} segments, {X_test.shape[1]} features")
    return X_test, submission


# ============================================================
# MODEL TRAINING & PREDICTION
# ============================================================

def train_and_predict(X, y, groups, X_test):
    """Ensemble of LightGBM + XGBoost with StratifiedGroupKFold."""
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = len(le.classes_)
    print(f"Classes: {le.classes_} (n={n_classes})")
    print(f"Distribution: {dict(zip(le.classes_, np.bincount(y_enc)))}")

    # compute class weights (inverse frequency)
    class_counts = np.bincount(y_enc)
    class_weights = len(y_enc) / (n_classes * class_counts)
    sample_weights = np.array([class_weights[c] for c in y_enc])

    n_splits = 5
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_preds = np.zeros((len(y), n_classes))
    test_preds_lgb = np.zeros((len(X_test), n_classes))
    test_preds_xgb = np.zeros((len(X_test), n_classes))
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y_enc, groups)):
        print(f"\n--- Fold {fold+1}/{n_splits} ---")
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y_enc[train_idx], y_enc[val_idx]
        w_tr = sample_weights[train_idx]

        # LightGBM
        lgb_params = {
            "objective": "multiclass",
            "num_class": n_classes,
            "metric": "multi_logloss",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "max_depth": -1,
            "min_child_samples": 20,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbosity": -1,
            "n_jobs": -1,
            "seed": 42 + fold,
        }
        lgb_train = lgb.Dataset(X_tr, y_tr, weight=w_tr)
        lgb_val = lgb.Dataset(X_val, y_val)

        model_lgb = lgb.train(
            lgb_params, lgb_train,
            num_boost_round=1000,
            valid_sets=[lgb_val],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        val_pred_lgb = model_lgb.predict(X_val.values)
        test_preds_lgb += model_lgb.predict(X_test.values) / n_splits

        # XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=n_classes,
            eval_metric="mlogloss",
            early_stopping_rounds=50,
            verbosity=0,
            random_state=42 + fold,
            n_jobs=-1,
        )
        xgb_model.fit(
            X_tr, y_tr, sample_weight=w_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        val_pred_xgb = xgb_model.predict_proba(X_val)
        test_preds_xgb += xgb_model.predict_proba(X_test) / n_splits

        # Blend OOF
        val_pred_blend = 0.5 * val_pred_lgb + 0.5 * val_pred_xgb
        oof_preds[val_idx] = val_pred_blend

        fold_f1 = f1_score(y_val, np.argmax(val_pred_blend, axis=1), average="weighted")
        fold_scores.append(fold_f1)
        print(f"  Fold {fold+1} macro F1: {fold_f1:.4f}")

    # Overall OOF score
    oof_labels = np.argmax(oof_preds, axis=1)
    overall_f1 = f1_score(y_enc, oof_labels, average="weighted")
    print(f"\n{'='*50}")
    print(f"Overall OOF macro F1: {overall_f1:.4f}")
    print(f"Per-fold F1: {[f'{s:.4f}' for s in fold_scores]}")
    print(classification_report(y_enc, oof_labels, target_names=le.classes_))

    # Blend test predictions
    test_preds = 0.5 * test_preds_lgb + 0.5 * test_preds_xgb
    test_labels_enc = np.argmax(test_preds, axis=1)
    test_labels = le.inverse_transform(test_labels_enc)

    return test_labels, test_preds, le


# ============================================================
# POST-PROCESSING
# ============================================================

def smooth_predictions(submission, test_preds, le, window=3):
    """Smooth predictions per subject using a sliding window on probabilities."""
    print("Applying prediction smoothing...")
    smoothed_labels = []
    subjects = []
    for seg_id in submission["id"]:
        subjects.append(seg_id.rsplit("_", 1)[0])

    submission_temp = submission.copy()
    submission_temp["subject"] = subjects

    result_labels = np.argmax(test_preds, axis=1).copy()

    for subject in sorted(set(subjects)):
        mask = submission_temp["subject"] == subject
        idxs = np.where(mask)[0]
        probs = test_preds[idxs]

        # apply moving average on probabilities
        smoothed_probs = np.copy(probs)
        for i in range(len(probs)):
            start = max(0, i - window // 2)
            end = min(len(probs), i + window // 2 + 1)
            smoothed_probs[i] = np.mean(probs[start:end], axis=0)

        result_labels[idxs] = np.argmax(smoothed_probs, axis=1)

    return le.inverse_transform(result_labels)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import time
    t0 = time.time()

    # Load data
    X_train, y_train, groups = load_train_data()
    X_test, submission = load_test_data()

    # Align columns
    X_test = X_test[X_train.columns]

    # Replace inf/nan
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Train and predict
    test_labels, test_preds, le = train_and_predict(X_train, y_train, groups, X_test)

    # Post-process: smooth predictions
    test_labels_smoothed = smooth_predictions(submission, test_preds, le, window=5)

    # Save submission
    submission["labels"] = test_labels_smoothed
    output_path = os.path.join(BASE_DIR, "submission.csv")
    submission.to_csv(output_path, index=False)
    print(f"\nSubmission saved to {output_path}")
    print(f"Label distribution: {dict(zip(*np.unique(test_labels_smoothed, return_counts=True)))}")
    print(f"\nTotal time: {time.time() - t0:.1f}s")