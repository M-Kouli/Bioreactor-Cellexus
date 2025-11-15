from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import pandas as pd
from collections import Counter

# set random seed

# seed = int(os.environ.get("GLOBAL_SEED", random.SystemRandom().randint(0, 2**32 - 1)))
# print(seed)
seed = 1054254479

# Set 4 more seeds for analysis using random number generator once, save manually for reuse every time 
#SEED2 = int(os.environ.get("GLOBAL_SEED",random.SystemRandom().randint(0,2**32-1)))
SEED2 = 2854737905
#SEED3 = int(os.environ.get("GLOBAL_SEED",random.SystemRandom().randint(0,2**32-1)))
SEED3 = 1701380585
#SEED4 = int(os.environ.get("GLOBAL_SEED",random.SystemRandom().randint(0,2**32-1)))
SEED4 = 4044111372
#SEED5 = int(os.environ.get("GLOBAL_SEED",random.SystemRandom().randint(0,2**32-1)))
SEED5 = 2419978774


train = pd.read_csv("pseudo_species_train.tsv", sep = "\t")
val = pd.read_csv("pseudo_species_val.tsv", sep = "\t")

test = pd.read_csv("test.tsv", sep = "\t")

#features selection
feature_cols = ["lon_sin","lon_cos", "lat_sin", "lat_cos", "temp_mean_1991_2024", "temp_std_1991_2024"]
label_cols = ["presence"]

#creating dfs for training and validaation
x_train = train[feature_cols]
y_train = train[label_cols]

x_val = val[feature_cols]
y_val = val[label_cols]

np.random.seed(seed) # makes code repeatable, using seed previously initialised

random_species = np.random.choice(train['species_id'].unique(), size=100, replace=False)

y_train_array = y_train.values.ravel() # convert to 1D array 


# # Random Forest hyperparameters
# param_distributions = {
#     'n_estimators': [200, 400, 600], # number of trees in the forest
#     'max_depth': [10, 20, 30, None], # maximum depth of trees, None means nodes expand until all leaves are pure
#     'min_samples_split': [2, 5, 10], # minimum samples required to split an internal node
#     'min_samples_leaf': [1, 2, 4], # minimum samples required to be at a leaf node
#     'max_features': ['sqrt', 'log2', 0.5], # number of features to consider when looking for best split
#     'max_samples': [0.5, 0.7, 0.9], # fraction of samples to use for each tree (bootstrap sampling)
#     'bootstrap': [True], # whether to use bootstrap samples
#     'class_weight': ['balanced', 'balanced_subsample', None] # handle class imbalance
# }


# results_rf = [] # empty list to fill in with for loop


# for species in random_species: # repeat classifier for each of n species selected
#     rf = RandomForestClassifier( # create base Random Forest classifier
#         random_state=seed,
#         n_jobs=1  # per-tree parallelism
#     )

#     search = RandomizedSearchCV( # cross validation
#         rf,
#         param_distributions=param_distributions,
#         n_iter=8, # testing 8 random combinations
#         scoring='f1_macro',  # scoring system
#         n_jobs=-1, # use all processors available for CV
#         cv=2 # cross validation twice
#     )

#     search.fit(x_train, y_train_array) # fit to training data

#     best_model = search.best_estimator_
#     preds = best_model.predict(x_val)  # predict on validation data
#     f1 = f1_score(y_val, preds) # calculate f1 score for validation data

#     results_rf.append({
#         "species_id": species,
#         "best_params": search.best_params_,
#         "val_f1": f1
#     })


# results_rf_df = results_rf.copy()
# results_rf_df = pd.DataFrame(results_rf_df)

# best_params_list = results_rf_df["best_params"].tolist()
# counts = Counter([frozenset(p.items()) for p in best_params_list])  # convert dicts to hashable frozensets

# most_common_params = dict(max(counts.items(), key=lambda x: x[1])[0])

# print("Most common best hyperparameters:", most_common_params)

# Model training using best hyperparameters from search
# print("Base Model")
# model = RandomForestClassifier(
#     n_estimators= 100, # number of trees in the forest
#     max_depth= None, # maximum depth of trees, None means nodes expand until all leaves are pure
#     min_samples_split= 10, # minimum samples required to split an internal node
#     min_samples_leaf= 4, # minimum samples required to be at a leaf node
#     max_features= 'sqrt', # number of features to consider when looking for best split
#     bootstrap= True, # whether to use bootstrap samples
#     class_weight= 'balanced', # handle class imbalance,  # Unpack best hyperparameters
#     n_jobs=1,  # per-tree parallelism
#     random_state=seed,
# )

# multi_model = MultiOutputClassifier(estimator=model, n_jobs=20)
# print("Fitting Model")
# multi_model.fit(x_train, y_train)


# #predict probabilities
# train_probs_list = multi_model.predict_proba(x_train)
# val_probs_list = multi_model.predict_proba(x_val)

# #combine probabilities into 1 array each
# train_probs = np.column_stack([p[:,1] for p in train_probs_list])
# val_probs = np.column_stack([p[:,1] for p in val_probs_list])


# #convert to presence (1) or absence (0) using 0.5 as a threshold
# train_predictions = (train_probs > 0.45).astype(int)
# val_predictions = (val_probs > 0.45).astype(int) 


# # F1 macro
# lgb_f1_macro_train = f1_score(y_train.values, train_predictions, average='macro')
# lgb_f1_macro_val = f1_score(y_val.values, val_predictions, average='macro')
# print("F1 train Macro:", lgb_f1_macro_train)
# print("F1 Val Macro:", lgb_f1_macro_val)

# # Matthews correlation coefficient 
# train_mccs = [matthews_corrcoef(y_train.iloc[:, i], train_predictions[:, i]) for i in range(len(label_cols))]
# val_mccs = [matthews_corrcoef(y_val.iloc[:, i], val_predictions[:, i]) for i in range(len(label_cols))]
# species_mccs = pd.DataFrame({
#     'species': label_cols,
#     'train_MCC': train_mccs,
#     'val_MCC': val_mccs
# })
# mean_train_mcc = np.nanmean(train_mccs)
# mean_val_mcc = np.nanmean(val_mccs)
# print("Mean train MCC:", mean_train_mcc)
# print("Mean val MCC:", mean_val_mcc)

# #AUC PR
# train_auc_pr = average_precision_score(y_train.values, train_probs, average='macro')
# val_auc_pr = average_precision_score(y_val.values, val_probs, average='macro')
# print("Train AUC PR (macro):", train_auc_pr)
# print("Val AUC PR (macro):", val_auc_pr)

# #AUC-ROC
# train_auc_roc = roc_auc_score(y_train.values, train_probs, average='macro')
# val_auc_roc = roc_auc_score(y_val.values, val_probs, average='macro')
# print("Train AUC ROC (macro):", train_auc_roc)
# print("Val AUC ROC (macro):", val_auc_roc)


results_rows = []          # overall metrics per seed & split
species_results_rows = []  # per-species metrics per seed (on test set)
#Repeat with each of the 5 seeds using best hyperparameters
for s in [seed, SEED2, SEED3, SEED4, SEED5]:
    print("Looping over", s)
    model = RandomForestClassifier(
        n_estimators= 100, # number of trees in the forest
        max_depth= None, # maximum depth of trees, None means nodes expand until all leaves are pure
        min_samples_split= 10, # minimum samples required to split an internal node
        min_samples_leaf= 4, # minimum samples required to be at a leaf node
        max_features= 'sqrt', # number of features to consider when looking for best split
        bootstrap= True, # whether to use bootstrap samples
        class_weight= 'balanced', # handle class imbalance,  # Unpack best hyperparameters
        n_jobs=1,  # per-tree parallelism
        random_state=s,
    )

    multi_model = MultiOutputClassifier(estimator=model, n_jobs=-1)
    print("Fitting Model")
    multi_model.fit(x_train, y_train)


    #predict probabilities
    train_probs_list = multi_model.predict_proba(x_train)
    val_probs_list = multi_model.predict_proba(x_val)

    #combine probabilities into 1 array each
    train_probs = np.column_stack([p[:,1] for p in train_probs_list])
    val_probs = np.column_stack([p[:,1] for p in val_probs_list])


    #convert to presence (1) or absence (0) using 0.5 as a threshold
    train_predictions = (train_probs > 0.45).astype(int)
    val_predictions = (val_probs > 0.45).astype(int) 


    # F1 macro
    lgb_f1_macro_train = f1_score(y_train.values, train_predictions, average='macro')
    lgb_f1_macro_val = f1_score(y_val.values, val_predictions, average='macro')
    print("F1 train Macro:", lgb_f1_macro_train)
    print("F1 Val Macro:", lgb_f1_macro_val)

    # Matthews correlation coefficient 
    train_mccs = [matthews_corrcoef(y_train.iloc[:, i], train_predictions[:, i]) for i in range(len(label_cols))]
    val_mccs = [matthews_corrcoef(y_val.iloc[:, i], val_predictions[:, i]) for i in range(len(label_cols))]
    species_mccs = pd.DataFrame({
        'species': label_cols,
        'train_MCC': train_mccs,
        'val_MCC': val_mccs
    })
    mean_train_mcc = np.nanmean(train_mccs)
    mean_val_mcc = np.nanmean(val_mccs)
    print("Mean train MCC:", mean_train_mcc)
    print("Mean val MCC:", mean_val_mcc)

    #AUC PR
    train_auc_pr = average_precision_score(y_train.values, train_probs, average='macro')
    val_auc_pr = average_precision_score(y_val.values, val_probs, average='macro')
    print("Train AUC PR (macro):", train_auc_pr)
    print("Val AUC PR (macro):", val_auc_pr)

    #AUC-ROC
    train_auc_roc = roc_auc_score(y_train.values, train_probs, average='macro')
    val_auc_roc = roc_auc_score(y_val.values, val_probs, average='macro')
    print("Train AUC ROC (macro):", train_auc_roc)
    print("Val AUC ROC (macro):", val_auc_roc)

    # Define test features
    test_feature_cols = ["lon_sin","lon_cos", "lat_sin", "lat_cos", "temp_mean_1991_2024", "temp_std_1991_2024"]
    species_cols = test.columns.difference(test_feature_cols)

    # overall rows (one per split)
    results_rows.append({
        "seed": s,
        "split": "train",
        "f1_macro": lgb_f1_macro_train,
        "mean_mcc": mean_train_mcc,
        "auc_pr":   train_auc_pr,
        "auc_roc":  train_auc_roc,
    })
    results_rows.append({
        "seed": s,
        "split": "val",
        "f1_macro": lgb_f1_macro_val,
        "mean_mcc": mean_val_mcc,
        "auc_pr":   val_auc_pr,
        "auc_roc":  val_auc_roc,
    })

#     for species in species_cols:
#         print(species)
#         x_test = test[test_feature_cols]
#         y_test = test[species]
#         test_probs_list = multi_model.predict_proba(x_test)

#         #combine probabilities into 1 array each
#         test_probs = np.column_stack([p[:,1] for p in test_probs_list])

#         test_predictions = (test_probs > 0.45).astype(int) 

#         f1_macro_test = f1_score(y_test.values, test_predictions, average='macro')
#         print("F1 train Macro:", f1_macro_test)

#         test_mccs = [matthews_corrcoef(y_test.iloc[:, i], test_predictions[:, i]) for i in range(len(label_cols))]
#         mean_test_mcc = np.nanmean(test_mccs)
#         print("Mean train MCC:", mean_test_mcc)

#         test_auc_pr = average_precision_score(y_test.values, test_probs, average='macro')
#         print("AUC PR (macro):", test_auc_pr)

#         test_auc_roc = roc_auc_score(y_test.values, test_probs, average='macro')
#         print("AUC ROC (macro):", test_auc_roc)

#         species_results_rows.append({
#             "seed": s,
#             "species": species_name,
#             "f1": f1_macro_test,
#             "mcc": mean_test_mcc,
#             "auc_pr": test_auc_pr,
#             "auc_roc": test_auc_roc,
#         })


results_df = pd.DataFrame(results_rows)
species_results_df = pd.DataFrame(species_results_rows)

print("Overall results:")
print(results_df.head())

print("Per-species results:")
print(species_results_df.head())

# Example summaries:

# Mean over seeds per split (overall)
overall_summary = (
    results_df
    .groupby("split")[["f1_macro", "mean_mcc", "auc_pr", "auc_roc"]]
    .mean()
)
print("Overall summary across seeds:")
print(overall_summary)

# # Mean per species on test across seeds
# per_species_test_summary = (
#     species_results_df[species_results_df["split"].isna() == True]  # if no split column there, ignore this
# )
# # if you want split in per-species too, you can add it above; otherwise:
# per_species_test_summary = (
#     species_results_df
#     .groupby("species")[["f1", "mcc", "auc_pr", "auc_roc"]]
#     .mean()
#     .sort_values("f1", ascending=False)
# )
# print("Per-species mean (test, across seeds):")
# print(per_species_test_summary.head())
