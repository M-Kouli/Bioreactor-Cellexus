"""
Multi-Label Random Forest Species Classification Pipeline
=========================================================

This script implements a complete machine learning pipeline for predicting species presence
at geographical sites using environmental features from both terrestrial and marine sources.

Key Features:
- Multi-label classification for ~500 species
- Handles missing values through imputation
- Feature correlation pruning (threshold: 0.85)
- Hyperparameter tuning with RandomizedSearchCV
- Spatial cross-validation using GroupKFold
- Comprehensive evaluation metrics (F1-macro, MCC, AUC-ROC, AUC-PR)
- Fast Mode: Optional stratified sampling for quick experimentation

Fast Mode Configuration:
- Set FAST_MODE = True to use a subset of training data (default: 30%)
- Uses stratified sampling to maintain species distribution
- Ensures all species are represented in the subset
- Significantly reduces training time for hyperparameter tuning
- Final model (Section 8) always uses full training data
- Recommended for initial experimentation and debugging
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for thread-safety with parallel processing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score, average_precision_score, make_scorer
from sklearn.base import clone
import joblib
import warnings
warnings.filterwarnings('ignore')
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib

# Set random seed for reproducibility
SEED = 399786328
np.random.seed(SEED)

# ============================================================================
# FAST MODE CONFIGURATION
# ============================================================================
# Enable FAST_MODE for quick experimentation with subset of data
# Set to True to train on a smaller, stratified sample that includes all species
FAST_MODE = True
SUBSET_FRACTION = 0.3  # Use 30% of training data (ensures species coverage)

if FAST_MODE:
    print("\n" + "!" * 80)
    print("FAST MODE ENABLED")
    print(f"Training will use {SUBSET_FRACTION*100:.0f}% of data (stratified by species presence)")
    print("Set FAST_MODE = False for full training")
    print("!" * 80)

print("=" * 80)
print("Multi-Label Random Forest Species Classification Pipeline")
print("=" * 80)

# ============================================================================
# 1. DATA LOADING
# ============================================================================

print("\n[1] Loading data from pickle files...")

try:
    import pickle
    with open("train_df_with_bioclim.pkl", "rb") as f:
        train_df = pickle.load(f)
    with open("test_df_with_bioclim.pkl", "rb") as f:
        test_df = pickle.load(f)
    print(f"✓ Training data loaded: {train_df.shape}")
    print(f"✓ Test data loaded: {test_df.shape}")
except Exception as e:
    print(f"✗ Error loading pickle files: {e}")
    print("\nTrying .npz files as fallback...")
    try:
        train_npz = np.load("train_df_with_bioclim.npz", allow_pickle=True)
        test_npz = np.load("test_df_with_bioclim.npz", allow_pickle=True)
        
        train_arr = train_npz["train_df"]
        test_arr = test_npz["test_df"]
        
        if train_arr.ndim == 0:
            train_df = train_arr.item()
            test_df = test_arr.item()
        elif isinstance(train_arr, np.ndarray) and train_arr.dtype == object:
            if train_arr.shape == ():
                train_df = train_arr.item()
                test_df = test_arr.item()
            elif train_arr.shape == (1,):
                train_df = train_arr[0]
                test_df = test_arr[0]
            else:
                train_df = pd.DataFrame(train_arr)
                test_df = pd.DataFrame(test_arr)
        else:
            train_df = pd.DataFrame(train_arr)
            test_df = pd.DataFrame(test_arr)
        
        print(f"✓ Training data loaded from .npz: {train_df.shape}")
        print(f"✓ Test data loaded from .npz: {test_df.shape}")
    except Exception as e2:
        print(f"✗ All loading methods failed")
        print(f"  Pickle error: {e}")
        print(f"  NPZ error: {e2}")
        raise

# Identify which columns are features vs species
# Species columns are typically integers, features are strings
string_cols = [col for col in train_df.columns if isinstance(col, str)]
numeric_cols = [col for col in train_df.columns if isinstance(col, (int, np.integer))]

print(f"  String columns: {len(string_cols)}")
print(f"  Numeric columns (likely species): {len(numeric_cols)}")

# Define expected feature columns
expected_features = [
    'temp_mean_1991_2024', 'temp_std_1991_2024',
    'land_mask', 'lon_sin', 'lon_cos', 'lat_sin', 'lat_cos',
    'marine_flag'
]

# Add bioclim precipitation features
expected_features += [f"wc2.1_10m_bio_{n}" for n in range(12, 20)]

# Add elevation
expected_features += [col for col in string_cols if 'elev' in col.lower()]

# Add monthly-derived summaries
expected_features += [
    'srad_mean', 'srad_min', 'srad_max', 'srad_std',
    'wind_mean', 'wind_min', 'wind_max', 'wind_std',
    'vapr_mean', 'vapr_min', 'vapr_max', 'vapr_std'
]

# Add Bio-ORACLE marine features
marine_features = ['bo_temp_mean', 'bo_sal_mean', 'bo_curr_mean',
                   'bo_chl_mean', 'bo_pp_mean', 'bo_depth']
expected_features += marine_features

# Filter to only columns that actually exist
feat_cols = [col for col in expected_features if col in train_df.columns]

# If no features found in expected names, use all string columns except special ones
if len(feat_cols) == 0:
    print("  Warning: No expected features found, using all string columns as features")
    exclude_special = {'site_id', 'latitude', 'longitude', 'species_id'}
    feat_cols = [col for col in string_cols if col not in exclude_special]

# Species columns are the remaining columns (typically integers)
exclude_cols = set(feat_cols + ['site_id', 'latitude', 'longitude'])
species_cols = [col for col in train_df.columns if col not in exclude_cols]

print(f"  Detected {len(feat_cols)} feature columns")
print(f"  Detected {len(species_cols)} species columns")
if len(feat_cols) > 0:
    print(f"  Sample features: {feat_cols[:5]}")
if len(species_cols) > 0:
    print(f"  Sample species: {species_cols[:5]}")

print(f"✓ Number of features: {len(feat_cols)}")
print(f"✓ Number of species: {len(species_cols)}")

# Build X and Y matrices
X_train = train_df[feat_cols].copy()
Y_train = train_df[species_cols].copy().values

X_test = test_df[feat_cols].copy()
Y_test = test_df[species_cols].copy().values

print(f"✓ X_train shape: {X_train.shape}")
print(f"✓ Y_train shape: {Y_train.shape}")
print(f"✓ X_test shape: {X_test.shape}")
print(f"✓ Y_test shape: {Y_test.shape}")

# ============================================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================

print("\n[2] Exploratory Data Analysis")
print("-" * 80)

# 2.1 Basic structure
print("\n2.1 Dataset Structure:")
print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"\nFirst 5 rows of features:")
print(train_df[feat_cols].head())
print(f"\nFirst 5 rows, first 10 species:")
print(train_df[species_cols].iloc[:5, :10])

# 2.2 Land vs Marine composition
print("\n2.2 Land vs Marine Composition:")
if 'marine_flag' in train_df.columns:
    print("\nTraining set marine_flag distribution:")
    print(train_df["marine_flag"].value_counts(normalize=True))
    print("\nTest set marine_flag distribution:")
    print(test_df["marine_flag"].value_counts(normalize=True))

if 'land_mask' in train_df.columns:
    print("\nTraining set land_mask distribution (0 vs >0):")
    print((train_df["land_mask"] == 0).value_counts(normalize=True))
    print("\nTest set land_mask distribution (0 vs >0):")
    print((test_df["land_mask"] == 0).value_counts(normalize=True))

# 2.3 Missingness analysis
print("\n2.3 Missing Values Analysis:")
if len(feat_cols) > 0:
    missing_train = X_train.isna().mean().sort_values(ascending=False)
    print("\nTop 30 features with missing values:")
    print(missing_train.head(30))

    # Plot missingness only if there are features with missing values
    if len(missing_train) > 0 and missing_train.head(20).sum() > 0:
        plt.figure(figsize=(12, 6))
        missing_train.head(20).plot(kind='barh')
        plt.xlabel('Fraction Missing')
        plt.title('Top 20 Features by Missing Values (Training Set)')
        plt.tight_layout()
        plt.savefig('missing_values_plot.png', dpi=150, bbox_inches='tight')
        print("✓ Saved missingness plot to 'missing_values_plot.png'")
        plt.close()
    else:
        print("  No missing values detected in features")
else:
    print("  No features available for missingness analysis")

# 2.4 Feature distributions
print("\n2.4 Feature Distributions:")
sample_features = ['temp_mean_1991_2024', 'land_mask']
if 'srad_mean' in feat_cols:
    sample_features.append('srad_mean')
if 'wind_mean' in feat_cols:
    sample_features.append('wind_mean')

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for idx, feat in enumerate(sample_features[:4]):
    if feat in X_train.columns:
        ax = axes[idx]
        X_train[feat].hist(bins=50, alpha=0.7, ax=ax)
        ax.set_xlabel(feat)
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of {feat}')

plt.tight_layout()
plt.savefig('feature_distributions.png', dpi=150, bbox_inches='tight')
print("✓ Saved feature distributions plot to 'feature_distributions.png'")
plt.close()

# 2.5 Species prevalence
print("\n2.5 Species Prevalence Analysis:")
species_prevalence = Y_train.mean(axis=0)
prevalence_series = pd.Series(species_prevalence, index=species_cols).sort_values(ascending=False)

print("\nTop 20 most prevalent species:")
print(prevalence_series.head(20))
print("\nTop 20 least prevalent species:")
print(prevalence_series.tail(20))

plt.figure(figsize=(12, 6))
prevalence_series.hist(bins=50)
plt.xlabel('Species Prevalence')
plt.ylabel('Number of Species')
plt.title('Distribution of Species Prevalence')
plt.tight_layout()
plt.savefig('species_prevalence.png', dpi=150, bbox_inches='tight')
print("✓ Saved species prevalence plot to 'species_prevalence.png'")
plt.close()

# ============================================================================
# 3. PEARSON CORRELATION AND FEATURE PRUNING
# ============================================================================

print("\n[3] Pearson Correlation Analysis and Feature Pruning")
print("-" * 80)

print("\nComputing correlation matrix...")
corr = X_train.corr(method="pearson")

corr_matrix = corr.abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Find features to drop (correlation > 0.85)
to_drop = [
    column
    for column in upper.columns
    if any(upper[column] > 0.85)
]

print(f"\nFeatures with correlation > 0.85: {len(to_drop)}")
if to_drop:
    print("\nHighly correlated features to drop:")
    for col in to_drop:
        corr_pairs = upper[col][upper[col] > 0.85]
        for pair_col, corr_val in corr_pairs.items():
            print(f"  {col} <-> {pair_col}: {corr_val:.3f}")

# Define pruned feature list
pruned_feat_cols = [c for c in feat_cols if c not in to_drop]
print(f"\n✓ Original features: {len(feat_cols)}")
print(f"✓ Pruned features: {len(pruned_feat_cols)}")
print(f"✓ Features removed: {len(feat_cols) - len(pruned_feat_cols)}")

# Rebuild with pruned features
X_train_pruned = train_df[pruned_feat_cols].copy()
X_test_pruned = test_df[pruned_feat_cols].copy()

print(f"✓ X_train_pruned shape: {X_train_pruned.shape}")
print(f"✓ X_test_pruned shape: {X_test_pruned.shape}")

# Save pruned feature list
with open('pruned_features.txt', 'w') as f:
    f.write('\n'.join(pruned_feat_cols))
print("✓ Saved pruned feature list to 'pruned_features.txt'")

# ============================================================================
# 4. BASELINE TRAIN/VALIDATION SPLIT
# ============================================================================

print("\n[4] Creating Baseline Train/Validation Split (80/20)")
print("-" * 80)

train_idx, val_idx = train_test_split(
    train_df.index,
    test_size=0.2,
    random_state=SEED,
    shuffle=True,
)

X_tr = X_train_pruned.loc[train_idx].values
Y_tr = Y_train[train_idx]

X_val = X_train_pruned.loc[val_idx].values
Y_val = Y_train[val_idx]

print(f"✓ Training set: {X_tr.shape[0]} samples")
print(f"✓ Validation set: {X_val.shape[0]} samples")

# ============================================================================
# 4.1 FAST MODE: STRATIFIED SAMPLING FOR QUICK EXPERIMENTATION
# ============================================================================

if FAST_MODE:
    print(f"\n[4.1] FAST MODE: Creating Stratified Subset ({SUBSET_FRACTION*100:.0f}%)")
    print("-" * 80)
    
    # Calculate species presence for stratification
    species_presence = (Y_tr.sum(axis=1) > 0).astype(int)  # At least one species present
    
    # Calculate how many samples with/without species to keep
    n_samples_target = int(len(Y_tr) * SUBSET_FRACTION)
    
    # Use stratified sampling to maintain species distribution
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, train_size=SUBSET_FRACTION, random_state=SEED)
    
    for subset_idx, _ in sss.split(X_tr, species_presence):
        X_tr_subset = X_tr[subset_idx]
        Y_tr_subset = Y_tr[subset_idx]
        break
    
    # Report species coverage
    original_species_with_data = (Y_tr.sum(axis=0) > 0).sum()
    subset_species_with_data = (Y_tr_subset.sum(axis=0) > 0).sum()
    species_coverage = (subset_species_with_data / original_species_with_data) * 100
    
    print(f"✓ Original training samples: {X_tr.shape[0]}")
    print(f"✓ Subset training samples: {X_tr_subset.shape[0]} ({SUBSET_FRACTION*100:.0f}%)")
    print(f"✓ Species with data in original: {original_species_with_data}/{Y_tr.shape[1]}")
    print(f"✓ Species with data in subset: {subset_species_with_data}/{Y_tr.shape[1]}")
    print(f"✓ Species coverage: {species_coverage:.1f}%")
    
    # Replace training data with subset
    X_tr = X_tr_subset
    Y_tr = Y_tr_subset
    
    print(f"\n✓ Fast mode enabled: Using {X_tr.shape[0]} samples for training")
    print("  (Full data will be used for final model in Section 8)")

# ============================================================================
# 5. MODEL: MULTI-OUTPUT RANDOM FOREST WITH RANDOMIZEDSEARCHCV
# ============================================================================

print("\n[5] Building Multi-Output Random Forest with Hyperparameter Tuning")
print("-" * 80)

# Build base pipeline
base_rf = RandomForestClassifier(
    n_jobs=1,
    random_state=SEED,
)

multi_rf = MultiOutputClassifier(
    estimator=base_rf,
    n_jobs=-1,
)

pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="mean", add_indicator=True)),
    ("clf", multi_rf),
])

# Define improved hyperparameter search space
# More comprehensive and balanced search space for Random Forest
param_distributions = {
    "clf__estimator__n_estimators": [100,200,500],  # Wider range including smaller models
    "clf__estimator__max_depth": [20, 30, None],  # More granular depth control
    "clf__estimator__min_samples_split": [2, 5, 10],  # Prevent overfitting
    "clf__estimator__min_samples_leaf": [1, 2, 4],  # Leaf size control
    "clf__estimator__max_features": ["sqrt", "log2", 0.5],  # Feature sampling strategies
    "clf__estimator__bootstrap": [True],  # Keep True for stability
    "clf__estimator__class_weight": ["balanced"],  # Handle imbalanced species
}

# Define scorer
f1_macro_scorer = make_scorer(f1_score, average="macro", zero_division=0)

print("\nImproved Hyperparameter Search Space:")
for param, values in param_distributions.items():
    print(f"  {param.replace('clf__estimator__', '')}: {values}")

print("\nStarting RandomizedSearchCV (cv=3, n_iter=10)...")
print("This may take a while...")
progress_bar = tqdm(total=10*3, desc="CV fits", unit="fit")
rand_search = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=param_distributions,
    n_iter=8,  
    scoring=f1_macro_scorer,
    cv=3,
    random_state=SEED,
    n_jobs=4,
    verbose=2,
    refit=True,
)

with tqdm_joblib(progress_bar) as _:
    rand_search.fit(X_tr, Y_tr)

progress_bar.close()

print("\n✓ Hyperparameter tuning complete!")
print(f"Best parameters: {rand_search.best_params_}")
print(f"Best CV F1-macro: {rand_search.best_score_:.4f}")

# Save detailed search results
results_df = pd.DataFrame(rand_search.cv_results_)
results_df.to_csv('randomized_search_results_full.csv', index=False)
print("✓ Saved full search results to 'randomized_search_results_full.csv'")

# Create a cleaner summary table with key metrics for each combination
summary_cols = [
    'rank_test_score',
    'mean_test_score',
    'std_test_score',
    'mean_fit_time',
    'mean_score_time',
    'param_clf__estimator__n_estimators',
    'param_clf__estimator__max_depth',
    'param_clf__estimator__min_samples_split',
    'param_clf__estimator__min_samples_leaf',
    'param_clf__estimator__max_features',
    'param_clf__estimator__class_weight',
]

summary_df = results_df[summary_cols].copy()
summary_df.columns = [
    'rank', 'f1_macro_mean', 'f1_macro_std', 'fit_time_mean', 'score_time_mean',
    'n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf',
    'max_features', 'class_weight'
]
summary_df = summary_df.sort_values('rank')
summary_df.to_csv('hyperparameter_search_summary.csv', index=False)
print("✓ Saved summary to 'hyperparameter_search_summary.csv'")

print("\nTop 10 Hyperparameter Combinations:")
print(summary_df.head(10).to_string(index=False))

# ============================================================================
# 6. VALIDATION METRICS
# ============================================================================

print("\n[6] Computing Validation Metrics")
print("-" * 80)

best_model = rand_search.best_estimator_

# Predictions on validation set
Y_val_pred = best_model.predict(X_val)
val_f1_macro = f1_score(Y_val, Y_val_pred, average="macro", zero_division=0)

# MCC (mean over species)
mcc_values = []
for j in range(Y_val.shape[1]):
    y_true_j = Y_val[:, j]
    y_pred_j = Y_val_pred[:, j]
    if len(np.unique(y_true_j)) < 2:
        continue
    mcc_values.append(matthews_corrcoef(y_true_j, y_pred_j))
mcc_mean = np.mean(mcc_values) if mcc_values else np.nan

# AUC-ROC (mean over species)
proba_list = best_model.predict_proba(X_val)
proba_pos = np.column_stack([p[:, 1] for p in proba_list])

auc_roc_values = []
for j in range(Y_val.shape[1]):
    y_true_j = Y_val[:, j]
    if len(np.unique(y_true_j)) < 2:
        continue
    auc_roc_values.append(roc_auc_score(y_true_j, proba_pos[:, j]))
auc_roc_mean = np.mean(auc_roc_values) if auc_roc_values else np.nan

# AUC-PR (mean over species)
auc_pr_values = []
for j in range(Y_val.shape[1]):
    y_true_j = Y_val[:, j]
    if len(np.unique(y_true_j)) < 2:
        continue
    auc_pr_values.append(average_precision_score(y_true_j, proba_pos[:, j]))
auc_pr_mean = np.mean(auc_pr_values) if auc_pr_values else np.nan

print(f"\nBest Model Validation Metrics:")
print(f"  F1-macro: {val_f1_macro:.4f}")
print(f"  MCC (mean over species): {mcc_mean:.4f}")
print(f"  ROC AUC (mean over species): {auc_roc_mean:.4f}")
print(f"  PR AUC (mean over species): {auc_pr_mean:.4f}")

# Compute detailed metrics for each hyperparameter combination
print("\n[6.1] Computing Validation Metrics for All Hyperparameter Combinations")
print("-" * 80)
print("Evaluating all tested combinations on validation set...")

detailed_metrics = []
for idx in range(len(rand_search.cv_results_['params'])):
    params = rand_search.cv_results_['params'][idx]
    cv_score = rand_search.cv_results_['mean_test_score'][idx]
    cv_std = rand_search.cv_results_['std_test_score'][idx]
    rank = rand_search.cv_results_['rank_test_score'][idx]
    
    # Create model with these specific parameters
    test_rf = RandomForestClassifier(
        n_estimators=params['clf__estimator__n_estimators'],
        max_depth=params['clf__estimator__max_depth'],
        min_samples_split=params['clf__estimator__min_samples_split'],
        min_samples_leaf=params['clf__estimator__min_samples_leaf'],
        max_features=params['clf__estimator__max_features'],
        class_weight=params['clf__estimator__class_weight'],
        bootstrap=params['clf__estimator__bootstrap'],
        n_jobs=1,
        random_state=SEED,
    )
    
    test_multi_rf = MultiOutputClassifier(test_rf, n_jobs=-1)
    test_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean", add_indicator=True)),
        ("clf", test_multi_rf),
    ])
    
    # Fit on training set
    test_pipe.fit(X_tr, Y_tr)
    
    # Evaluate on validation set
    Y_val_pred_test = test_pipe.predict(X_val)
    f1_val = f1_score(Y_val, Y_val_pred_test, average="macro", zero_division=0)
    
    # MCC
    mcc_vals_test = []
    for j in range(Y_val.shape[1]):
        y_true_j = Y_val[:, j]
        y_pred_j = Y_val_pred_test[:, j]
        if len(np.unique(y_true_j)) < 2:
            continue
        mcc_vals_test.append(matthews_corrcoef(y_true_j, y_pred_j))
    mcc_val = np.mean(mcc_vals_test) if mcc_vals_test else np.nan
    
    # AUC-ROC
    proba_list_test = test_pipe.predict_proba(X_val)
    proba_pos_test = np.column_stack([p[:, 1] for p in proba_list_test])
    
    auc_roc_vals_test = []
    for j in range(Y_val.shape[1]):
        y_true_j = Y_val[:, j]
        if len(np.unique(y_true_j)) < 2:
            continue
        auc_roc_vals_test.append(roc_auc_score(y_true_j, proba_pos_test[:, j]))
    auc_roc_val = np.mean(auc_roc_vals_test) if auc_roc_vals_test else np.nan
    
    # AUC-PR
    auc_pr_vals_test = []
    for j in range(Y_val.shape[1]):
        y_true_j = Y_val[:, j]
        if len(np.unique(y_true_j)) < 2:
            continue
        auc_pr_vals_test.append(average_precision_score(y_true_j, proba_pos_test[:, j]))
    auc_pr_val = np.mean(auc_pr_vals_test) if auc_pr_vals_test else np.nan
    
    detailed_metrics.append({
        'rank': rank,
        'n_estimators': params['clf__estimator__n_estimators'],
        'max_depth': params['clf__estimator__max_depth'],
        'min_samples_split': params['clf__estimator__min_samples_split'],
        'min_samples_leaf': params['clf__estimator__min_samples_leaf'],
        'max_features': params['clf__estimator__max_features'],
        'class_weight': params['clf__estimator__class_weight'],
        'cv_f1_mean': cv_score,
        'cv_f1_std': cv_std,
        'val_f1_macro': f1_val,
        'val_mcc_mean': mcc_val,
        'val_auc_roc_mean': auc_roc_val,
        'val_auc_pr_mean': auc_pr_val,
    })
    
    print(f"  Evaluated combination {idx+1}/{len(rand_search.cv_results_['params'])}", end='\r')

print("\n✓ All combinations evaluated on validation set")

# Save detailed metrics
detailed_metrics_df = pd.DataFrame(detailed_metrics).sort_values('rank')
detailed_metrics_df.to_csv('all_combinations_validation_metrics.csv', index=False)
print("✓ Saved detailed metrics to 'all_combinations_validation_metrics.csv'")

print("\nTop 5 Combinations by Validation F1-macro:")
top_5_val = detailed_metrics_df.nlargest(5, 'val_f1_macro')[
    ['rank', 'n_estimators', 'max_depth', 'val_f1_macro', 'val_mcc_mean', 'val_auc_roc_mean', 'val_auc_pr_mean']
]
print(top_5_val.to_string(index=False))

# Save best model
joblib.dump(best_model, 'best_model_baseline.joblib')
print("\n✓ Saved best model to 'best_model_baseline.joblib'")

# ============================================================================
# 7. SPATIAL BLOCKING CROSS-VALIDATION
# ============================================================================

print("\n[7] Spatial Blocking Cross-Validation")
print("-" * 80)

# Extract best hyperparameters
best_params = rand_search.best_params_

rf_params = {
    "n_estimators": best_params["clf__estimator__n_estimators"],
    "max_depth": best_params["clf__estimator__max_depth"],
    "min_samples_split": best_params["clf__estimator__min_samples_split"],
    "min_samples_leaf": best_params["clf__estimator__min_samples_leaf"],
    "max_features": best_params["clf__estimator__max_features"],
    "n_jobs": 1,
    "random_state": SEED,
}

print(f"\nUsing best RF parameters: {rf_params}")

spatial_rf = RandomForestClassifier(**rf_params)
spatial_multi_rf = MultiOutputClassifier(spatial_rf, n_jobs=-1)

spatial_model = Pipeline([
    ("imputer", SimpleImputer(strategy="mean", add_indicator=True)),
    ("clf", spatial_multi_rf),
])

# Define spatial blocks (5-degree grid)
print("\nCreating spatial blocks (5-degree grid)...")
lat = train_df["latitude"].values if "latitude" in train_df.columns else np.zeros(len(train_df))
lon = train_df["longitude"].values if "longitude" in train_df.columns else np.zeros(len(train_df))

lat_bins = np.floor(lat / 5.0)
lon_bins = np.floor(lon / 5.0)

spatial_blocks = (lat_bins.astype(int) * 1000 + lon_bins.astype(int))

print(f"✓ Number of unique spatial blocks: {len(np.unique(spatial_blocks))}")

# GroupKFold with 3 splits
gkf = GroupKFold(n_splits=3)

X_full = X_train_pruned.values
Y_full = Y_train
groups = spatial_blocks

f1_scores = []
mcc_scores = []
auc_roc_scores = []
auc_pr_scores = []

print("\nRunning spatial cross-validation...")

for fold, (train_idx_sp, test_idx_sp) in enumerate(gkf.split(X_full, Y_full, groups)):
    print(f"\n--- Spatial CV Fold {fold+1}/3 ---")
    
    X_tr_sp, X_te_sp = X_full[train_idx_sp], X_full[test_idx_sp]
    Y_tr_sp, Y_te_sp = Y_full[train_idx_sp], Y_full[test_idx_sp]
    
    print(f"Train samples: {X_tr_sp.shape[0]}, Test samples: {X_te_sp.shape[0]}")
    
    model_fold = clone(spatial_model)
    model_fold.fit(X_tr_sp, Y_tr_sp)
    
    Y_pred_sp = model_fold.predict(X_te_sp)
    f1_sp = f1_score(Y_te_sp, Y_pred_sp, average="macro", zero_division=0)
    
    # MCC mean
    mcc_vals = []
    for j in range(Y_te_sp.shape[1]):
        y_true_j = Y_te_sp[:, j]
        y_pred_j = Y_pred_sp[:, j]
        if len(np.unique(y_true_j)) < 2:
            continue
        mcc_vals.append(matthews_corrcoef(y_true_j, y_pred_j))
    mcc_sp = np.mean(mcc_vals) if mcc_vals else np.nan
    
    # AUC-ROC mean
    proba_list_sp = model_fold.predict_proba(X_te_sp)
    proba_pos_sp = np.column_stack([p[:, 1] for p in proba_list_sp])
    
    auc_roc_vals = []
    for j in range(Y_te_sp.shape[1]):
        y_true_j = Y_te_sp[:, j]
        if len(np.unique(y_true_j)) < 2:
            continue
        auc_roc_vals.append(roc_auc_score(y_true_j, proba_pos_sp[:, j]))
    auc_roc_sp = np.mean(auc_roc_vals) if auc_roc_vals else np.nan
    
    # AUC-PR mean
    auc_pr_vals = []
    for j in range(Y_te_sp.shape[1]):
        y_true_j = Y_te_sp[:, j]
        if len(np.unique(y_true_j)) < 2:
            continue
        auc_pr_vals.append(average_precision_score(y_true_j, proba_pos_sp[:, j]))
    auc_pr_sp = np.mean(auc_pr_vals) if auc_pr_vals else np.nan
    
    print(f"Fold F1-macro: {f1_sp:.4f}, MCC-mean: {mcc_sp:.4f}, AUC-ROC: {auc_roc_sp:.4f}, AUC-PR: {auc_pr_sp:.4f}")
    
    f1_scores.append(f1_sp)
    mcc_scores.append(mcc_sp)
    auc_roc_scores.append(auc_roc_sp)
    auc_pr_scores.append(auc_pr_sp)
    
    # Save fold model
    joblib.dump(model_fold, f'spatial_model_fold_{fold+1}.joblib')
    print(f"✓ Saved fold model to 'spatial_model_fold_{fold+1}.joblib'")

print("\n" + "=" * 80)
print("Spatial CV Summary:")
print("=" * 80)
print(f"F1-macro: mean={np.nanmean(f1_scores):.4f}, std={np.nanstd(f1_scores):.4f}")
print(f"MCC-mean: mean={np.nanmean(mcc_scores):.4f}, std={np.nanstd(mcc_scores):.4f}")
print(f"AUC-ROC: mean={np.nanmean(auc_roc_scores):.4f}, std={np.nanstd(auc_roc_scores):.4f}")
print(f"AUC-PR: mean={np.nanmean(auc_pr_scores):.4f}, std={np.nanstd(auc_pr_scores):.4f}")

# Save spatial CV results
spatial_results = pd.DataFrame({
    'fold': range(1, 4),
    'f1_macro': f1_scores,
    'mcc_mean': mcc_scores,
    'auc_roc_mean': auc_roc_scores,
    'auc_pr_mean': auc_pr_scores
})
spatial_results.to_csv('spatial_cv_results.csv', index=False)
print("\n✓ Saved spatial CV results to 'spatial_cv_results.csv'")

# ============================================================================
# 8. FINAL MODEL TRAINING ON FULL DATASET
# ============================================================================

print("\n[8] Training Final Model on Full Training Dataset")
print("-" * 80)

final_model = clone(spatial_model)
print("Fitting final model...")
final_model.fit(X_full, Y_full)

# Save final model
joblib.dump(final_model, 'final_model_full_train.joblib')
print("✓ Saved final model to 'final_model_full_train.joblib'")

# ============================================================================
# 9. FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("PIPELINE COMPLETE - SUMMARY")
print("=" * 80)

print("\nFiles created:")
print("  • best_model_baseline.joblib - Best model from RandomizedSearchCV")
print("  • final_model_full_train.joblib - Final model trained on all training data")
print("  • spatial_model_fold_1.joblib, 2, 3 - Models from spatial CV folds")
print("  • pruned_features.txt - List of features after correlation pruning")
print("  • randomized_search_results_full.csv - Full hyperparameter search results")
print("  • hyperparameter_search_summary.csv - Clean summary of search results")
print("  • all_combinations_validation_metrics.csv - Validation metrics for each combination")
print("  • spatial_cv_results.csv - Spatial cross-validation results")
print("  • missing_values_plot.png - Visualization of missing data")
print("  • feature_distributions.png - Feature distribution plots")
print("  • species_prevalence.png - Species prevalence histogram")

print("\nKey Metrics:")
print(f"  Features used: {len(pruned_feat_cols)}")
print(f"  Species classified: {len(species_cols)}")
print(f"  Best CV F1-macro: {rand_search.best_score_:.4f}")
print(f"  Validation F1-macro: {val_f1_macro:.4f}")
print(f"  Spatial CV F1-macro: {np.nanmean(f1_scores):.4f} ± {np.nanstd(f1_scores):.4f}")
print(f"  Spatial CV MCC-mean: {np.nanmean(mcc_scores):.4f} ± {np.nanstd(mcc_scores):.4f}")
print(f"  Spatial CV AUC-ROC: {np.nanmean(auc_roc_scores):.4f} ± {np.nanstd(auc_roc_scores):.4f}")
print(f"  Spatial CV AUC-PR: {np.nanmean(auc_pr_scores):.4f} ± {np.nanstd(auc_pr_scores):.4f}")

print("\n" + "=" * 80)
print("Script execution completed successfully!")
print("=" * 80)