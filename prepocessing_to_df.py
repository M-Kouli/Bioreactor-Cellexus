import numpy as np
import pandas as pd
import os
import random
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import BallTree

# Current inefficient test build remove warning from stack 
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# Set seed using random number generator once, save manually for reuse every time 
#SEED = int(os.environ.get("GLOBAL_SEED",random.SystemRandom().randint(0,2**32-1)))

SEED = 399786328

# Load the .npz file directly (no extraction needed)
z_tr = np.load("species_train_with_climate.npz", allow_pickle=True)
z_te = np.load("species_test_with_climate.npz", allow_pickle=True)

# Make training dataframe for modelling
train_locs   = z_tr["train_locs"]            # (N, 2) [lat, lon]
train_ids    = z_tr["train_ids"]             # (N,)
taxon_ids_tr = z_tr["taxon_ids"]             # (500,)
# Dictionary mapping taxon_id to taxon_name
taxon_nams_tr= np.asarray(z_tr["taxon_names"]).astype(str)
taxon_id_to_name = dict(zip(taxon_ids_tr, taxon_nams_tr))
# Training shape 
N = train_locs.shape[0]
# Creating training dataframe
df_train = pd.DataFrame({
    "latitude":  train_locs[:, 0],
    "longitude": train_locs[:, 1],
    "species_id": train_ids,
})
# Add per-row feature arrays (same length as train_locs)
skip_tr = {"train_locs", "train_ids", "taxon_ids", "taxon_names"}
for k in z_tr.files:
    if k in skip_tr:
        continue
    arr = z_tr[k]
    if isinstance(arr, np.ndarray) and arr.shape == (N,):
        df_train[k] = arr
    elif isinstance(arr, np.ndarray) and arr.shape == (N, 1):
        df_train[k] = arr[:, 0]

# Optional checl of training dataframe shape
#print("First print df_train shape:", df_train.head())


#CREATING TEST DATA
test_locs     = z_te["test_locs"]            # (M, 2)
# z_te['test_pos_inds'] is a list of lists, where each list corresponds to 
# the indices in test_locs where a given species is present, it can be assumed 
# that they are not present in the other locations 
# test_pos_inds corresponds to the number of species after checking the shape of it.
# so number of species is equal to the taxon_ids.
test_pos_inds = dict(zip(z_te['taxon_ids'], z_te['test_pos_inds']))    

# Getting test rows
M = test_locs.shape[0]
print(M)
# Base test DF
df_test = pd.DataFrame({
    "latitude":  test_locs[:, 0],
    "longitude": test_locs[:, 1],
})

# Add per-row feature arrays (same length as test_locs)
skip_te = {"test_locs", "taxon_ids", "test_pos_inds"}
for k in z_te.files:
    if k in skip_te:
        continue
    arr = z_te[k]
    if isinstance(arr, np.ndarray) and arr.shape == (M,):
        df_test[k] = arr
    elif isinstance(arr, np.ndarray) and arr.shape == (M, 1):
        df_test[k] = arr[:, 0]

# print("df_test shape: \n", df_test.head())

#Data Encoding for Longitude and Latitude function (*corrected version - TW)
def encode_coords(df):
    df['lon_sin'] = np.sin((np.pi * df['longitude']) / 180)
    df['lon_cos'] = np.cos((np.pi * df['longitude']) / 180)
    df['lat_sin'] = np.sin((np.pi * df['latitude']) / 180)
    df['lat_cos'] = np.cos((np.pi * df['latitude']) / 180)
    return df
# Apply encoding to train dataframe
df_train = encode_coords(df_train)

# Create and specify feature columns
feat_cols = ['temp_mean_1991_2024', 'temp_std_1991_2024', 
             'land_mask', 'lon_sin', 'lon_cos', 'lat_sin', 'lat_cos'
            ]

# Split the training data into train and validation
species_train, species_val = sklearn.model_selection.train_test_split(df_train, test_size=0.2, random_state = SEED)

# Save train and validation files to tsv (optional)
species_train.to_csv("presence_species_train.tsv",sep="\t",encoding="utf-8",index=False)
species_val.to_csv("presence_species_val.tsv",sep="\t",encoding="utf-8",index=False)

# Generate pseudo-absence data from background sightings (of other species)
np.random.seed(SEED)
n_absence_per_species = 500
max_distance = 100
R = 6371
pseudoabsence_dataframes = []

coordinates = np.radians(df_train[["latitude", "longitude"]].to_numpy())
radius = max_distance / R
tree = BallTree(coordinates, metric = "haversine")
                                   
# Sort into presence (from iNaturalist) and background (other species iNaturalist)
for species in df_train["species_id"].unique():
    presence_df = df_train[df_train["species_id"] == species].copy()
    background_df = df_train[df_train["species_id"] != species].copy()

    # Convert to radians for haversine
    presence_loc = np.radians(presence_df[["latitude", "longitude"]].to_numpy())

    # Ball tree
    neighbor_lists = tree.query_radius(presence_loc, r=radius)

    neighbor_index = np.unique(np.concatenate(neighbor_lists))
    
    presence_idx = presence_df.index.to_numpy()
    pseudoabsences_range_geo = np.setdiff1d(neighbor_index, presence_idx)
    pseudoabsences_range_geo = [i for i in pseudoabsences_range_geo if i in background_df.index]

    # Setting a temperature range to sample from within 
    temp_min, temp_max = presence_df["temp_mean_1991_2024"].quantile([0.1, 0.9])
    precip_min, precip_max = presence_df["annual_precip_mm"].quantile([0.1, 0.9])
    
    pseudoabsences_range_env = background_df[(background_df["temp_mean_1991_2024"].between(temp_min, temp_max)) &
                                    (background_df["annual_precip_mm"].between(precip_min, precip_max))].index
    
    # Combining geographical and environmental ranges on pseudoabsence sampling
    pseudoabsences_range = list(set(pseudoabsences_range_geo).intersection(pseudoabsences_range_env))
    # Sampling pseudoabsence sites from set range without replacement (if not enough, take any available)
    if len(pseudoabsences_range) >= n_absence_per_species:
        pseudoabsences_sample = np.random.choice(
            pseudoabsences_range,
            size = n_absence_per_species,
            replace = False
        )
    else:
        pseudoabsences_sample = pseudoabsences_range

    # Labelling presence datapoints 
    presence_df["presence"] = 1
    # Labelling psuedoabsence datapoints
    pseudoabsence_df = background_df.loc[pseudoabsences_sample].copy()
    pseudoabsence_df["species_id"] = species
    pseudoabsence_df["presence"] = 0

    # Combine into pseudoabsences list
    df_presence_absence = pd.concat([presence_df, pseudoabsence_df], ignore_index=True)
    pseudoabsence_dataframes.append(df_presence_absence)

# Combine all of the species dataframes made in the loop into pseudoabsence dataframe
pseudoabsence_train = pd.concat(pseudoabsence_dataframes, ignore_index = True)

# Split the training data (WITH PSEUDOABSENCE) into train and validation
pseudo_species_train, pseudo_species_val = sklearn.model_selection.train_test_split(pseudoabsence_train, test_size=0.2, random_state = SEED, stratify=pseudoabsence_train["species_id"])

# Save pseudoabsence train and validation files to tsv (optional)
pseudo_species_train.to_csv("pseudo_species_train.tsv",sep="\t",encoding="utf-8",index=False)
pseudo_species_val.to_csv("pseudo_species_val.tsv",sep="\t",encoding="utf-8",index=False)

# # Make test dataframe for final test purposes
# # Build Y: site × species 0/1 matrix
# # Specify all species present, same as the given code for sp = np.random.choice(species)
# species_cols = z_te["taxon_ids"]  # same order as in z_te

# # Create a dataframe with the co-ords present
# y = pd.DataFrame(test_locs, columns=["longitude", "latitude"])

# # Loop over the species in the test and extract their location they are present in
# for sp in species_cols:
#     # Set the species as 0 for all locations first, this will also add the species ID as a column
#     y[sp] = 0 
#     # From the given code test_inds_pos is the locations where the selected species is present
#     test_inds_pos = test_pos_inds[sp]
#     # Locate in the dataframe created above where the species is present and assign at that column a value of 1
#     y.loc[test_inds_pos, sp] = 1 
#     # # This is just to confirm for an example "sp 17050" to make sure the dataframe is capturing the location correctly
#     # if sp == 17050:
#     #     pres = test_locs[test_inds_pos, 1], test_locs[test_inds_pos, 0]
#     #     print("Lon",pres[0])
#     #     print("Lat",pres[1])
#     #     print(len(pres[0]))
        
# # Test with the same sp 17050 from above to compare
# # sp = 17050
# # print(y.loc[y[sp] == 1, ["longitude", "latitude", sp]])
# #print(len(y))

# # This is just dropping the percipitation columns, remove the line below, once the percipitation data is removed from the .npz
# df_test = df_test.drop(columns=["annual_precip_mm", "precip_wettest_month_mm", "precip_driest_month_mm"])

# # Select the columns, drop the lon and lat, this is because the co-ords are in the same order
# y = y[species_cols]

# # Join the df_test with the dataframe built above
# df_test = pd.concat([df_test, y], axis=1)

# # # Confirm that it looks sensible
# # print(df_test.head())

# # Now apply encoding to test dataframe
# df_test = encode_coords(df_test)

# #Grouping by site_id for multilabeling
# df_test["site_id"] = df_test.groupby(
#     ["lon_sin", "lon_cos", "lat_sin", "lat_cos"], sort=False
# ).ngroup()

# # Build X same as above
# X_test = (df_test.groupby("site_id")[feat_cols].first()
#        .sort_index())

# # Build Y, as we have already done major part of it above
# Y_test = (
#     df_test.groupby("site_id")[species_cols]
#            .max()          # 1 if species present at any row for that site
#            .sort_index()
# )

# # Combine X and Y into one big DF
# test_df = pd.concat([X_test, Y_test], axis=1)

# # Save test to tsv (optional)
# test_df.to_csv("test.tsv",sep="\t",encoding="utf-8",index=False)


# # Check with above that the data persisted and that there arent any errors, check the num of rows
# sp = 17050
# print(test_df.loc[test_df[sp] == 1, ['temp_mean_1991_2024', 'temp_std_1991_2024', 'land_mask', 'lon_sin', 'lon_cos', 'lat_sin', 'lat_cos',sp]])

# Set 9 more seeds for analysis using random number generator once, save manually for reuse every time 
#SEED2 = int(os.environ.get("GLOBAL_SEED",random.SystemRandom().randint(0,2**32-1)))
SEED2 = 2854737905
#SEED3 = int(os.environ.get("GLOBAL_SEED",random.SystemRandom().randint(0,2**32-1)))
SEED3 = 1701380585
#SEED4 = int(os.environ.get("GLOBAL_SEED",random.SystemRandom().randint(0,2**32-1)))
SEED4 = 4044111372
#SEED5 = int(os.environ.get("GLOBAL_SEED",random.SystemRandom().randint(0,2**32-1)))
SEED5 = 2419978774
#SEED6 = int(os.environ.get("GLOBAL_SEED",random.SystemRandom().randint(0,2**32-1)))
SEED6 = 4047028074
#SEED7 = int(os.environ.get("GLOBAL_SEED",random.SystemRandom().randint(0,2**32-1)))
SEED7 = 2180401918
#SEED8 = int(os.environ.get("GLOBAL_SEED",random.SystemRandom().randint(0,2**32-1)))
SEED8 = 2162869241
#SEED9 = int(os.environ.get("GLOBAL_SEED",random.SystemRandom().randint(0,2**32-1)))
SEED9 = 1462129709

